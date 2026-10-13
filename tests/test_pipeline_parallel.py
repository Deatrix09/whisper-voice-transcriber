"""Validační testy pro Fix E (L06/L08): paralela + bisect merge + logger levels.

1. Paralela: diarizace a transkripce se překrývají v čase (důkaz ThreadPoolu).
2. Fallbacky: num_workers<2 -> sekvenčně; disabled -> žádný ThreadPool.
3. Výjimky z obou větví probublají.
4. Merge parita: bisect verze == původní O(T*D^2) na náhodných datech + tie-case.
"""
import random
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _pipe(app_config, name="test-parallel"):
    from src.core.pipeline import Pipeline
    from src.utils.logger import setup_logger
    pipe = Pipeline.__new__(Pipeline)
    pipe.config = app_config
    pipe.logger = setup_logger(name, "WARNING", None)
    pipe.audio_processor = MagicMock()
    import torch
    pipe.audio_processor.load_audio.return_value = (torch.empty(0), 16000, None)
    pipe.diarizer = MagicMock()
    pipe.transcriber = MagicMock()
    return pipe


def test_parallel_overlap(app_config, tmp_path):
    """Důkaz překryvu: obě větve 0.4 s -> paralelně < 0.75 s a intervaly se kryjí."""
    from conftest import make_wav
    wav = make_wav(tmp_path / "p.wav", seconds=0.5)
    app_config.diarization.enabled = True
    app_config.performance.num_workers = 4
    pipe = _pipe(app_config, "test-par-overlap")

    marks = {}

    def fake_diar(path):
        import threading
        marks["diar_start"] = time.perf_counter()
        marks["diar_thread"] = threading.get_ident()
        time.sleep(0.4)
        marks["diar_end"] = time.perf_counter()
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5, "duration": 0.5}]

    def fake_tr(path):
        import threading
        marks["tr_start"] = time.perf_counter()
        marks["tr_thread"] = threading.get_ident()
        time.sleep(0.4)
        marks["tr_end"] = time.perf_counter()
        return {"text": "hello", "segments": [{"start": 0.0, "end": 0.5, "text": "hello"}]}

    pipe.diarizer.process.side_effect = fake_diar
    pipe.transcriber.transcribe.side_effect = fake_tr
    t0 = time.perf_counter()
    results = pipe.process_file(str(wav))
    wall = time.perf_counter() - t0

    assert results[0]["speaker"] == "SPEAKER_00"
    assert marks["diar_thread"] != marks["tr_thread"]
    assert marks["diar_start"] < marks["tr_end"] and marks["tr_start"] < marks["diar_end"]
    assert wall < 0.75, f"sekvenční? wall={wall:.2f}s"


def test_sequential_fallback_when_workers_lt2(app_config, tmp_path):
    """num_workers=1 -> žádný ThreadPool, pořadí diar -> transcribe jako dřív."""
    from conftest import make_wav
    wav = make_wav(tmp_path / "s.wav", seconds=0.5)
    app_config.diarization.enabled = True
    app_config.performance.num_workers = 1
    pipe = _pipe(app_config, "test-par-seq")
    order = []
    pipe.diarizer.process.side_effect = lambda p: order.append("diar") or []
    pipe.transcriber.transcribe.side_effect = (
        lambda p: order.append("tr") or {"text": "", "segments": []}
    )
    with patch("concurrent.futures.ThreadPoolExecutor",
               side_effect=AssertionError("nesmí vzniknout")):
        pipe.process_file(str(wav))
    assert order == ["diar", "tr"]


def test_no_threadpool_when_disabled(app_config, tmp_path):
    from conftest import make_wav
    wav = make_wav(tmp_path / "d.wav", seconds=0.5)
    app_config.diarization.enabled = False
    pipe = _pipe(app_config, "test-par-dis")
    pipe.transcriber.transcribe.return_value = {
        "text": "hi", "segments": [{"start": 0.0, "end": 0.5, "text": "hi"}]
    }
    with patch("concurrent.futures.ThreadPoolExecutor",
               side_effect=AssertionError("nesmí vzniknout")):
        results = pipe.process_file(str(wav))
    assert results[0]["speaker"] == "UNKNOWN"
    pipe.diarizer.process.assert_not_called()


def test_exceptions_propagate(app_config, tmp_path):
    from conftest import make_wav
    wav = make_wav(tmp_path / "e.wav", seconds=0.5)
    app_config.diarization.enabled = True
    # pád diarizace
    pipe = _pipe(app_config, "test-par-exc1")
    pipe.diarizer.process.side_effect = RuntimeError("diar boom")
    pipe.transcriber.transcribe.return_value = {"text": "", "segments": []}
    with pytest.raises(RuntimeError):
        pipe.process_file(str(wav))
    # pád transkripce
    pipe2 = _pipe(app_config, "test-par-exc2")
    pipe2.diarizer.process.return_value = []
    pipe2.transcriber.transcribe.side_effect = RuntimeError("tr boom")
    with pytest.raises(RuntimeError):
        pipe2.process_file(str(wav))


def _ref_merge(diarization_results, transcription_result, pipe):
    """Doslovná kopie PŮVODNÍHO algoritmu (full scan + sum přes celé D)."""
    merged_results = []
    prev_segment_text = None
    for segment in transcription_result["segments"]:
        segment_start = segment["start"]
        segment_end = segment["end"]
        segment_text = segment["text"].strip()
        if prev_segment_text and pipe._is_similar_text(segment_text, prev_segment_text):
            continue
        prev_segment_text = segment_text
        matching_speakers = [
            d["speaker"] for d in diarization_results
            if max(segment_start, d["start"]) < min(segment_end, d["end"])
        ]
        if not matching_speakers:
            speaker = "UNKNOWN"
        elif len(matching_speakers) > 1:
            speaker = max(
                matching_speakers,
                key=lambda s: sum(
                    min(segment_end, d["end"]) - max(segment_start, d["start"])
                    for d in diarization_results
                    if d["speaker"] == s
                )
            )
        else:
            speaker = matching_speakers[0]
        if not segment_text or (segment_end - segment_start < 0.5 and len(segment_text.split()) <= 1):
            continue
        if pipe._is_repetitive_content(segment_text):
            continue
        merged_results.append({
            "start": segment_start, "end": segment_end, "speaker": speaker,
            "text": segment_text, "duration": segment_end - segment_start,
        })
    return merged_results


def test_merge_parity_randomized(app_config):
    """Bisect merge == původní merge na 200 náhodných segmentech včetně tie."""
    rng = random.Random(42)
    pipe = _pipe(app_config, "test-par-parity")
    speakers = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    diar = []
    t = 0.0
    for _ in range(200):
        dur = rng.choice([0.4, 0.8, 1.5, 3.0])
        diar.append({"speaker": rng.choice(speakers), "start": t,
                     "end": t + dur, "duration": dur})
        t += dur * rng.choice([0.5, 0.9, 1.0])  # i překryvy
    rng.shuffle(diar)  # záměrně rozházené pořadí
    words = ["hello", "world", "test", "speech", "audio", "lorem ipsum dolor"]
    tr_segs = []
    t = 0.0
    for i in range(120):
        dur = rng.choice([0.3, 0.6, 1.2, 2.5])
        txt = " ".join(rng.choice(words) for _ in range(rng.randint(1, 6)))
        if i % 17 == 0:
            txt = "bla bla bla bla"  # repetitive filtr
        tr_segs.append({"start": t, "end": t + dur, "text": txt})
        t += dur
    tr = {"text": "x", "segments": tr_segs}
    # cílený tie: dva speakeři se STEJNÝM překryvem
    diar += [
        {"speaker": "TIE_A", "start": 1000.0, "end": 1002.0, "duration": 2.0},
        {"speaker": "TIE_B", "start": 1000.0, "end": 1002.0, "duration": 2.0},
    ]
    tr["segments"].append({"start": 1000.0, "end": 1002.0, "text": "tie breaker segment here"})

    new = pipe.merge_diarization_with_transcription(diar, tr)
    ref = _ref_merge(diar, tr, pipe)
    assert new == ref


def test_merge_empty_diarization(app_config):
    pipe = _pipe(app_config, "test-par-empty")
    tr = {"text": "hi", "segments": [{"start": 0.0, "end": 1.0, "text": "hi there"}]}
    out = pipe.merge_diarization_with_transcription([], tr)
    assert out[0]["speaker"] == "UNKNOWN"
