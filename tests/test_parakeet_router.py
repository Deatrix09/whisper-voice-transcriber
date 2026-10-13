# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Validační testy pro FÁZI 5: parakeet engine + router.

NeMo tu není -> stub `nemo.collections.asr` (reálné zůstávají chunk/merge
kontrakt, offsety, fallbacky). faster-whisper se pro router testy stubuje,
aby se nic nestahovalo.
"""
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def _nemo_stub(captured, hyps=None, fail_init=False, bf16_fail=False):
    nemo = types.ModuleType("nemo")
    collections = types.ModuleType("nemo.collections")
    asr = types.ModuleType("nemo.collections.asr")

    class FakeASRModel:
        @classmethod
        def from_pretrained(cls, model_name):
            captured["model_name"] = model_name
            if fail_init:
                raise RuntimeError("fake weights boom")
            return cls()

        def eval(self):
            captured["eval"] = True
            return self

        def cuda(self):
            captured["cuda"] = True
            return self

        def to(self, *a, **k):
            captured.setdefault("to", []).append((a, k))
            if k.get("dtype") == torch.bfloat16 and bf16_fail:
                raise RuntimeError("no bf16")
            return self

        def transcribe(self, files, batch_size=1, timestamps=False):
            captured["transcribe"] = {"files": list(files),
                                      "batch_size": batch_size,
                                      "timestamps": timestamps}
            return hyps if hyps is not None else []

    asr.models = SimpleNamespace(ASRModel=FakeASRModel)
    collections.asr = asr
    nemo.collections = collections
    return {"nemo": nemo, "nemo.collections": collections,
            "nemo.collections.asr": asr}


def _hyp(text, segments, words=None):
    return SimpleNamespace(text=text, timestamp={"segment": segments,
                                                 "word": words or []})


def _fw_stub(calls):
    mod = types.ModuleType("faster_whisper")

    class FakeModel:
        def __init__(self, model_size, device="auto", compute_type="default", **kw):
            calls.append({"model_size": model_size, "device": device,
                          "compute_type": compute_type})

        def transcribe(self, audio, **kw):
            return iter([]), SimpleNamespace(language="en")

    mod.WhisperModel = FakeModel
    return mod


@pytest.fixture
def pk_config(app_config):
    app_config.diarization.enabled = False
    app_config.asr.engine = "parakeet"
    app_config.asr.batch_size = 8
    return app_config


def _parakeet(pk_config):
    from src.transcription.parakeet_transcriber import ParakeetTranscriber
    return ParakeetTranscriber(pk_config)


# ---------- init ----------

def test_init_cuda_bf16(pk_config):
    pk_config.performance.device = "cuda"
    captured = {}
    with patch.dict(sys.modules, _nemo_stub(captured)), \
         patch.object(torch.cuda, "is_available", return_value=True):
        t = _parakeet(pk_config)
        t.check_ffmpeg = lambda: True
        t.initialize_model()
    assert captured["model_name"] == "nvidia/parakeet-tdt-0.6b-v3"
    assert captured.get("cuda") is True
    # .to(torch.bfloat16) pozičně (nn.Module API) - stub nahrává (args, kwargs)
    assert any(list(a) == [torch.bfloat16] or k.get("dtype") == torch.bfloat16
               for a, k in captured.get("to", []))
    assert t.device == "cuda"


def test_init_cpu(pk_config):
    captured = {}
    with patch.dict(sys.modules, _nemo_stub(captured)):
        t = _parakeet(pk_config)
        t.check_ffmpeg = lambda: True
        t.initialize_model()
    assert "cuda" not in captured  # žádný .cuda() na CPU
    assert t.device == "cpu"


def test_bf16_fallback_fp32(pk_config):
    pk_config.performance.device = "cuda"
    captured = {}
    with patch.dict(sys.modules, _nemo_stub(captured, bf16_fail=True)), \
         patch.object(torch.cuda, "is_available", return_value=True):
        t = _parakeet(pk_config)
        t.check_ffmpeg = lambda: True
        t.initialize_model()  # bez výjimky, fp32
    assert t.device == "cuda"


def test_missing_nemo(app_config, monkeypatch):
    from src.transcription.parakeet_transcriber import ParakeetTranscriber
    for m in ("nemo", "nemo.collections", "nemo.collections.asr"):
        monkeypatch.setitem(sys.modules, m, None)
    with pytest.raises(RuntimeError, match="NeMo"):
        ParakeetTranscriber(app_config).initialize_model()


# ---------- transcribe ----------

def test_batched_single_call_with_offsets(pk_config, tmp_path):
    """2 chunky -> JEDEN transcribe call (batch) + merge offsety 28s."""
    from conftest import make_wav
    pk_config.audio.chunk_size = 30
    pk_config.audio.chunk_overlap_sec = 2
    f1 = make_wav(tmp_path / "c1.wav", seconds=0.5)
    f2 = make_wav(tmp_path / "c2.wav", seconds=0.5)
    hyps = [_hyp("alfa beta", [{"start": 0.0, "end": 0.4, "segment": "alfa beta"}]),
            # chunk1 segment mimo overlap (lokální 3.0 -> globální 31.0); lokální
            # 0-2s by merge správně zahodil jako overlap-regionu (28.x <= 30)
            _hyp("gama delta", [{"start": 3.0, "end": 3.4, "segment": "gama delta"}])]
    captured = {}
    with patch.dict(sys.modules, _nemo_stub(captured, hyps)):
        t = _parakeet(pk_config)
        t.check_ffmpeg = lambda: True
        from src.audio.processor import AudioProcessor
        with patch.object(AudioProcessor, "split_audio_file",
                          return_value=[str(f1), str(f2)]), \
             patch.object(AudioProcessor, "cleanup", return_value=None):
            out = t.transcribe(str(f1))
    tc = captured["transcribe"]
    assert tc["files"] == [str(f1), str(f2)]  # jedna dávka
    assert tc["batch_size"] == 8
    assert tc["timestamps"] is True
    assert [(s["start"], s["end"], s["text"]) for s in out["segments"]] == [
        (0.0, 0.4, "alfa beta"), (31.0, 31.4, "gama delta")]


def test_segment_fallback_chunk_span(pk_config, tmp_path):
    """Bez parseovatelných segmentů -> 1 segment přes chunk (offsety drží)."""
    from conftest import make_wav
    wav = make_wav(tmp_path / "s.wav", seconds=0.5)
    hyps = [_hyp("pouhy text", [])]
    with patch.dict(sys.modules, _nemo_stub({}, hyps)):
        t = _parakeet(pk_config)
        t.check_ffmpeg = lambda: True
        out = t.transcribe(str(wav))
    assert len(out["segments"]) == 1
    assert out["segments"][0]["text"] == "pouhy text"
    assert out["segments"][0]["end"] == pytest.approx(0.5, abs=0.05)


def test_align_words(pk_config, tmp_path):
    from conftest import make_wav
    wav = make_wav(tmp_path / "w.wav", seconds=0.5)
    hyps = [_hyp("ahoj svete",
                 [{"start": 0.0, "end": 0.4, "segment": "ahoj svete"}],
                 [{"start": 0.0, "end": 0.2, "word": "ahoj"},
                  {"start": 0.2, "end": 0.4, "word": "svete"}])]
    with patch.dict(sys.modules, _nemo_stub({}, hyps)):
        t = _parakeet(pk_config)
        t.check_ffmpeg = lambda: True
        out = t.transcribe(str(wav))
        assert "words" not in out  # default False
        pk_config.asr.align_words = True
        t2 = _parakeet(pk_config)
        t2.check_ffmpeg = lambda: True
        out2 = t2.transcribe(str(wav))
    assert [w["text"] for w in out2["words"]] == ["ahoj", "svete"]


# ---------- router ----------

def _router_fw_stub(monkeypatch, calls):
    import src.transcription.transcriber as T
    monkeypatch.setitem(sys.modules, "faster_whisper", _fw_stub(calls))
    monkeypatch.setattr(T, "WhisperModel", sys.modules["faster_whisper"].WhisperModel)
    monkeypatch.setattr(T.Transcriber, "check_ffmpeg", lambda self: True)


def test_router_engine_direct(app_config, monkeypatch):
    from src.transcription.parakeet_transcriber import create_asr_transcriber, \
        ParakeetTranscriber
    from src.transcription.transcriber import Transcriber
    calls = []
    _router_fw_stub(monkeypatch, calls)
    app_config.asr.engine = "faster-whisper"
    t = create_asr_transcriber(app_config)
    assert isinstance(t, Transcriber) and not isinstance(t, ParakeetTranscriber)
    assert "nemo" not in sys.modules


def test_router_parakeet_and_fallback(app_config, monkeypatch):
    from src.transcription.parakeet_transcriber import create_asr_transcriber, \
        ParakeetTranscriber
    from src.transcription.transcriber import Transcriber
    _router_fw_stub(monkeypatch, [])
    app_config.asr.engine = "parakeet"
    with patch.dict(sys.modules, _nemo_stub({})):
        t = create_asr_transcriber(app_config)
    assert isinstance(t, ParakeetTranscriber)
    # rozbité NeMo -> graceful fallback (žádný raise)
    with patch.dict(sys.modules, _nemo_stub({}, fail_init=True)):
        t2 = create_asr_transcriber(app_config)
    assert type(t2) is Transcriber


def test_router_auto(app_config, monkeypatch):
    from src.transcription.parakeet_transcriber import create_asr_transcriber, \
        ParakeetTranscriber, PARAKEET_LANGS, supports_language
    from src.transcription.transcriber import Transcriber
    assert len(PARAKEET_LANGS) == 25 and "cs" in PARAKEET_LANGS
    assert supports_language("cs") and not supports_language("zh")
    assert not supports_language("") and not supports_language(None)
    _router_fw_stub(monkeypatch, [])
    app_config.asr.engine = "auto"
    app_config.model.language = "cs"
    with patch.dict(sys.modules, _nemo_stub({})):
        assert isinstance(create_asr_transcriber(app_config), ParakeetTranscriber)
    app_config.model.language = "zh"  # mimo 25 -> turbo, nemo se ani neimportuje
    for m in list(sys.modules):
        if m == "nemo" or m.startswith("nemo."):
            del sys.modules[m]
    assert type(create_asr_transcriber(app_config)) is Transcriber
    assert "nemo" not in sys.modules
    app_config.model.language = ""  # auto-detect -> bezpečný default turbo
    assert type(create_asr_transcriber(app_config)) is Transcriber


def test_release_compat():
    from types import SimpleNamespace
    from src.core.pipeline import Pipeline
    pipe = SimpleNamespace(transcriber=SimpleNamespace(model=object()),
                           diarizer=SimpleNamespace(diarizer=None))
    Pipeline.release_model_resources(pipe)
    assert pipe.transcriber.model is None
