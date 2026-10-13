"""Validační testy pro FÁZI 2: streaming 30s/2s chunking + Silero VAD prefiltr.

Formát výstupu pro pipeline (List[str] wav chunků + offset matematika
idx*step v merge) se nemění — viz test_merge_offsets_30_2.
"""
import numpy as np
import soundfile as sf
from unittest.mock import patch

import pytest

SR = 16000


def _tone_wav(path, seconds, freq=440.0, silence_ranges=()):
    """Sinus + volitelné úseky ticha [(start_s, end_s)]. int16 mono 16kHz."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    sig = (8000 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    for s, e in silence_ranges:
        sig[int(s * SR):int(e * SR)] = 0
    sf.write(str(path), sig, SR, subtype="PCM_16", format="WAV")
    return path


@pytest.fixture
def stream_config(app_config):
    app_config.audio.chunk_size = 30
    app_config.audio.chunk_overlap_sec = 2
    app_config.audio.chunk_threshold_mb = 0  # vždy split
    return app_config


def _proc(config):
    from src.audio.processor import AudioProcessor
    return AudioProcessor(config)


def test_chunk_math_30_2(stream_config, tmp_path):
    """65s -> 3 chunky [0-30][28-58][56-65], 16k mono PCM_16."""
    wav = _tone_wav(tmp_path / "a.wav", 65)
    with patch("faster_whisper.vad.get_speech_timestamps",
               return_value=[{"start": 0, "end": 30 * SR}]):
        chunks = _proc(stream_config).split_audio_file(str(wav))
    assert len(chunks) == 3
    durs = [sf.info(c).duration for c in chunks]
    assert durs[0] == pytest.approx(30.0, abs=0.05)
    assert durs[1] == pytest.approx(30.0, abs=0.05)
    assert durs[2] == pytest.approx(9.0, abs=0.05)
    for c in chunks:
        info = sf.info(c)
        assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"


def test_overlap_continuity(stream_config, tmp_path):
    """Poslední 2s chunku0 == první 2s chunku1 (stejné zdrojové samply)."""
    wav = _tone_wav(tmp_path / "b.wav", 65)
    with patch("faster_whisper.vad.get_speech_timestamps",
               return_value=[{"start": 0, "end": 30 * SR}]):
        c0, c1, _ = _proc(stream_config).split_audio_file(str(wav))
    d0, _ = sf.read(c0, dtype="int16")
    d1, _ = sf.read(c1, dtype="int16")
    assert len(d0) == 30 * SR and len(d1) == 30 * SR
    assert (d0[-2 * SR:] == d1[:2 * SR]).all()


def test_vad_silent_placeholder(stream_config, tmp_path):
    """VAD=[] -> 0.5s silence placeholder (merge indexy drží)."""
    wav = _tone_wav(tmp_path / "c.wav", 65)
    with patch("faster_whisper.vad.get_speech_timestamps", return_value=[]):
        chunks = _proc(stream_config).split_audio_file(str(wav))
    assert len(chunks) == 3
    for c in chunks:
        assert sf.info(c).duration == pytest.approx(0.5, abs=0.05)
        data, _ = sf.read(c, dtype="int16")
        assert (data == 0).all()


def test_real_silero_silence_is_zero(stream_config):
    """Reálné Silero v6: čisté ticho -> ratio 0.0 (žádné halucinace v tichu)."""
    zeros = np.zeros(30 * SR, dtype=np.float32)
    ratio, sec = _proc(stream_config)._vad_speech_ratio(zeros, SR)
    assert ratio == 0.0 and sec == 0.0


def test_forbid_10s_chunks(stream_config, tmp_path):
    """chunk<=10s -> ValueError (WER +3.5%); 20s jen warning."""
    wav = _tone_wav(tmp_path / "d.wav", 65)
    proc = _proc(stream_config)
    stream_config.audio.chunk_size = 10
    with pytest.raises(ValueError, match="WER"):
        proc.split_audio_file(str(wav))
    stream_config.audio.chunk_size = 20
    with patch("faster_whisper.vad.get_speech_timestamps",
               return_value=[{"start": 0, "end": 20 * SR}]):
        chunks = proc.split_audio_file(str(wav))  # bez výjimky
    assert len(chunks) == 4  # ceil(65/20)


def test_no_pydub_full_load(stream_config, tmp_path):
    """Důkaz streamingu: pydub--from_file padá, split stejně projde (PyAV cesta)."""
    wav = _tone_wav(tmp_path / "e.wav", 65)
    with patch("pydub.AudioSegment.from_file", side_effect=AssertionError("no full-load")), \
         patch("faster_whisper.vad.get_speech_timestamps",
               return_value=[{"start": 0, "end": 30 * SR}]):
        chunks = _proc(stream_config).split_audio_file(str(wav))
    assert len(chunks) == 3


def test_small_file_passthrough(app_config, tmp_path):
    """Pod prahem -> [originál], beze změny."""
    from conftest import make_wav
    wav = make_wav(tmp_path / "f.wav", seconds=0.5)
    assert _proc(app_config).split_audio_file(str(wav)) == [str(wav)]


def test_merge_offsets_30_2(stream_config):
    """Časová matematika merge s 30/2: overlap-duplikát ven, zbytek s offsety."""
    proc = _proc(stream_config)
    chunks = [
        {"text": "x", "language": "cs", "segments": [
            {"start": 0.0, "end": 29.5, "text": "uhelna veta jedna"},
            {"start": 29.0, "end": 29.9, "text": "uhelna veta dva"},
        ]},
        {"text": "x", "language": "cs", "segments": [
            {"start": 0.5, "end": 1.5, "text": "uhelna veta dva"},  # overlap-dup (28.5-29.5)
            {"start": 3.0, "end": 5.0, "text": "uhelna veta tri"},
        ]},
    ]
    out = proc.merge_transcription_results(chunks)
    texts = [s["text"] for s in out["segments"]]
    assert texts == ["uhelna veta jedna", "uhelna veta dva", "uhelna veta tri"]
    assert [(s["start"], s["end"]) for s in out["segments"]] == [
        (0.0, 29.5), (29.0, 29.9), (31.0, 33.0)]
