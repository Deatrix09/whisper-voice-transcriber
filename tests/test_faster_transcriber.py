"""Validační testy pro FÁZI 1: faster-whisper transcriber.

Stub část (bez vah, bez GPU): signatura volání, singleton, VAD/greedy
parametry, převod generátoru, compute_type mapování, graceful missing.
Real část: tiny smoke test (skip při chybějícím balíčku/síti).
"""
import sys
import types
from unittest.mock import patch

import pytest
import torch


class _FakeSeg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _FakeInfo:
    language = "cs"
    language_probability = 0.99


def _fake_fw_module(calls):
    mod = types.ModuleType("faster_whisper")

    class FakeModel:
        def __init__(self, model_size, device="auto", compute_type="default", **kw):
            calls["construct"].append(
                {"model_size": model_size, "device": device,
                 "compute_type": compute_type, "kw": kw})

        def transcribe(self, audio, **kw):
            calls["transcribe"].append({"audio": audio, "kw": kw})
            segs = iter([_FakeSeg(0.0, 1.5, " ahoj svete "),
                         _FakeSeg(1.5, 3.0, "druhy segment")])
            return segs, _FakeInfo()

    mod.WhisperModel = FakeModel
    return mod


@pytest.fixture
def fw_stub(monkeypatch):
    calls = {"construct": [], "transcribe": []}
    monkeypatch.setitem(sys.modules, "faster_whisper", _fake_fw_module(calls))
    import src.transcription.transcriber as T
    monkeypatch.setattr(T, "WhisperModel", sys.modules["faster_whisper"].WhisperModel)
    monkeypatch.setattr(T.Transcriber, "check_ffmpeg", lambda self: True)
    return T, calls


@pytest.fixture
def turbo_config(app_config):
    app_config.model.name = "large-v3-turbo"
    return app_config


def test_construct_once_with_turbo_cpu_int8(fw_stub, turbo_config, tmp_path):
    T, calls = fw_stub
    from conftest import make_wav
    wav = make_wav(tmp_path / "t.wav", seconds=0.5)
    tr = T.Transcriber(turbo_config)
    out = tr.transcribe(str(wav))
    assert len(calls["construct"]) == 1
    c = calls["construct"][0]
    assert c["model_size"] == "large-v3-turbo"
    assert c["device"] == "cpu"  # tohle prostředí nemá CUDA
    assert c["compute_type"] == "int8"
    assert out["language"] == "cs"
    assert out["text"] == "ahoj svete druhy segment"
    assert [(s["start"], s["end"]) for s in out["segments"]] == [(0.0, 1.5), (1.5, 3.0)]
    # singleton: druhý transcribe už nestaví
    tr.transcribe(str(wav))
    assert len(calls["construct"]) == 1
    assert len(calls["transcribe"]) == 2


def test_greedy_vad_options(fw_stub, turbo_config, tmp_path):
    T, calls = fw_stub
    from conftest import make_wav
    wav = make_wav(tmp_path / "g.wav", seconds=0.5)
    T.Transcriber(turbo_config).transcribe(str(wav))
    kw = calls["transcribe"][0]["kw"]
    assert kw["beam_size"] == 1
    assert kw["condition_on_previous_text"] is False
    assert kw["vad_filter"] is True
    assert set(kw["vad_parameters"]) == {"threshold", "min_speech_duration_ms",
                                         "min_silence_duration_ms", "speech_pad_ms"}
    assert kw["without_timestamps"] is False
    assert kw["word_timestamps"] is False
    assert kw["task"] == turbo_config.transcription.task  # passthrough z configu
    assert "language" not in kw  # "" -> auto-detect


def test_language_passed_when_set(fw_stub, turbo_config, tmp_path):
    T, calls = fw_stub
    turbo_config.model.language = "cs"
    from conftest import make_wav
    wav = make_wav(tmp_path / "l.wav", seconds=0.5)
    T.Transcriber(turbo_config).transcribe(str(wav))
    assert calls["transcribe"][0]["kw"]["language"] == "cs"


def test_cuda_compute_type_and_empty_cache(fw_stub, turbo_config, tmp_path):
    T, calls = fw_stub
    turbo_config.performance.device = "cuda"  # uživatel chce GPU + je dostupné
    from conftest import make_wav
    wav = make_wav(tmp_path / "c.wav", seconds=0.5)
    with patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch.cuda, "empty_cache") as ec, \
         patch.object(torch.cuda, "memory_allocated", return_value=0), \
         patch.object(torch.cuda, "memory_reserved", return_value=0), \
         patch.object(torch.cuda, "mem_get_info", return_value=(1, 2)):
        T.Transcriber(turbo_config).transcribe(str(wav))
    assert calls["construct"][0]["device"] == "cuda"
    assert calls["construct"][0]["compute_type"] == "int8_float16"
    ec.assert_called()


def test_missing_package_hint(fw_stub, turbo_config):
    T, _ = fw_stub
    T.WhisperModel = None
    tr = T.Transcriber(turbo_config)
    with pytest.raises(RuntimeError, match="faster-whisper"):
        tr.initialize_model()


def test_chunk_loop_no_reload(fw_stub, turbo_config, tmp_path):
    """2 chunky -> 2× transcribe, 1× konstrukce, 1× merge."""
    T, calls = fw_stub
    from conftest import make_wav
    wav = make_wav(tmp_path / "m.wav", seconds=0.5)
    tr = T.Transcriber(turbo_config)
    from src.audio.processor import AudioProcessor
    with patch.object(AudioProcessor, "split_audio_file",
                      return_value=[str(wav), str(wav)]), \
         patch.object(AudioProcessor, "merge_transcription_results",
                      return_value={"text": "m", "segments": [], "language": "cs"}) as mg, \
         patch.object(AudioProcessor, "cleanup", return_value=None):
        out = tr.transcribe(str(wav))
    assert len(calls["construct"]) == 1
    assert len(calls["transcribe"]) == 2
    mg.assert_called_once()
    assert out["text"] == "m"


def test_options_match_real_api(app_config):
    """Regrese: klíče options musí existovat v faster_whisper API.

    Chytilo by `logprob_threshold` vs `log_prob_threshold` (TypeError až za běhu).
    """
    fw = pytest.importorskip("faster_whisper")
    import inspect
    import src.transcription.transcriber as T
    params = set(inspect.signature(fw.WhisperModel.transcribe).parameters)
    opts = T.Transcriber(app_config)._transcribe_options()
    unknown = set(opts) - params - {"vad_parameters"}  # vad_parameters je dict, API ho bere
    assert "vad_parameters" in params
    assert not unknown, f"neznámé klíče pro faster-whisper: {unknown}"


def test_real_tiny_smoke(tmp_path):
    """End-to-end tiny na CPU (stáhne ~75MB při prvním běhu). Skip bez sítě/balíčku."""
    fw = pytest.importorskip("faster_whisper")
    from conftest import make_wav
    wav = make_wav(tmp_path / "tiny.wav", seconds=2.0)
    try:
        model = fw.WhisperModel("tiny", device="cpu", compute_type="int8")
    except Exception as e:
        pytest.skip(f"tiny nelze stáhnout/nahrát: {e}")
    segs, info = model.transcribe(str(wav), beam_size=1, vad_filter=True,
                                  condition_on_previous_text=False,
                                  without_timestamps=False)
    out = list(segs)  # inference běží až tady
    assert isinstance(out, list)
    assert info.language is not None
