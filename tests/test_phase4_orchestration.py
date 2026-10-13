"""Validační testy pro FÁZI 4: asr.* config, compute validace, batch, cleanup.

faster-whisper se stubuje (stejný vzor jako FÁZE 1); CT2 validace běží
proti reálnému ctranslate2 (ověřená množina typů).
"""
import sys
import types
from unittest.mock import patch
import os

import pytest
import torch


class _FakeSeg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _FakeInfo:
    language = "cs"


def _fw_stub(calls, fail_first=False):
    mod = types.ModuleType("faster_whisper")

    class FakeModel:
        calls_n = 0

        def __init__(self, model_size, device="auto", compute_type="default", **kw):
            FakeModel.calls_n += 1
            calls["construct"].append(
                {"model_size": model_size, "device": device, "compute_type": compute_type})
            if fail_first and FakeModel.calls_n == 1:
                raise RuntimeError("fake load boom")

        def transcribe(self, audio, **kw):
            calls["transcribe"].append(kw)
            return iter([_FakeSeg(0.0, 1.0, "text")]), _FakeInfo()

    mod.WhisperModel = FakeModel
    FakeModel.calls_n = 0
    return mod


@pytest.fixture
def tw_stub(monkeypatch):
    calls = {"construct": [], "transcribe": []}
    import src.transcription.transcriber as T
    monkeypatch.setitem(sys.modules, "faster_whisper", _fw_stub(calls))
    monkeypatch.setattr(T, "WhisperModel", sys.modules["faster_whisper"].WhisperModel)
    monkeypatch.setattr(T.Transcriber, "check_ffmpeg", lambda self: True)
    return T, calls


# ---------- config ----------

def test_asr_defaults_old_yaml(tmp_path):
    """Staré YAML bez asr: sekce -> defaulty, nic se nerozbije."""
    import yaml
    from src.utils.config import AppConfig
    import os
    repo_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config", "config.yaml")
    with open(repo_cfg, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    del data["asr"]
    p = tmp_path / "old.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = AppConfig.from_yaml(str(p))
    assert cfg.asr.engine == "faster-whisper"
    assert cfg.asr.model == "large-v3-turbo"
    assert cfg.asr.compute_type is None
    assert cfg.asr.batch_size == 4
    assert cfg.asr.align_words is False


def test_asr_parsed_new_yaml():
    import os
    from src.utils.config import AppConfig
    repo_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config", "config.yaml")
    cfg = AppConfig.from_yaml(repo_cfg)
    assert cfg.asr.engine == "faster-whisper"
    assert cfg.asr.model == "large-v3-turbo"
    assert cfg.asr.batch_size == 4


def test_appconfig_without_asr_kwarg(app_config):
    assert app_config.asr.model == "large-v3-turbo"  # default_factory


# ---------- transcriber resolution ----------

def test_model_asr_priority(tw_stub, app_config, tmp_path):
    T, calls = tw_stub
    from conftest import make_wav
    wav = make_wav(tmp_path / "m.wav", seconds=0.5)
    app_config.model.name = "base"
    app_config.asr.model = "tiny"
    T.Transcriber(app_config).transcribe(str(wav))
    assert calls["construct"][0]["model_size"] == "tiny"
    app_config.asr.model = ""
    T.Transcriber(app_config).transcribe(str(wav))
    assert calls["construct"][-1]["model_size"] == "base"  # fallback model.name


def test_compute_auto_cpu(tw_stub, app_config, tmp_path):
    T, calls = tw_stub
    from conftest import make_wav
    wav = make_wav(tmp_path / "c.wav", seconds=0.5)
    app_config.asr.compute_type = None
    T.Transcriber(app_config).transcribe(str(wav))
    assert calls["construct"][0]["compute_type"] == "int8"  # CPU auto


def test_compute_explicit_and_unsupported(tw_stub, app_config, tmp_path):
    T, calls = tw_stub
    from conftest import make_wav
    wav = make_wav(tmp_path / "e.wav", seconds=0.5)
    # explicit validní pro CPU (CT2 množina) -> respektováno
    app_config.asr.compute_type = "int8_float32"
    T.Transcriber(app_config).transcribe(str(wav))
    assert calls["construct"][-1]["compute_type"] == "int8_float32"
    # bf16 v CT2 4.8.3 NEEXISTUJE -> warn + fallback int8 (CPU)
    app_config.asr.compute_type = "int8_bfloat16"
    T.Transcriber(app_config).transcribe(str(wav))
    assert calls["construct"][-1]["compute_type"] == "int8"


def test_compute_cuda_mapping(app_config, tmp_path, monkeypatch):
    import src.transcription.transcriber as T
    from unittest.mock import patch
    from conftest import make_wav
    wav = make_wav(tmp_path / "g.wav", seconds=0.5)
    calls = {"construct": [], "transcribe": []}
    monkeypatch.setattr(T, "WhisperModel", _fw_stub(calls).WhisperModel)
    monkeypatch.setattr(T.Transcriber, "check_ffmpeg", lambda self: True)
    app_config.performance.device = "cuda"
    app_config.asr.compute_type = None
    with patch.object(torch.cuda, "is_available", return_value=True):
        T.Transcriber(app_config).transcribe(str(wav))
    assert calls["construct"][0]["device"] == "cuda"
    assert calls["construct"][0]["compute_type"] == "int8_float16"


def test_load_fallback_retry(app_config, tmp_path, monkeypatch):
    import src.transcription.transcriber as T
    from conftest import make_wav
    wav = make_wav(tmp_path / "r.wav", seconds=0.5)
    calls = {"construct": [], "transcribe": []}
    monkeypatch.setattr(T, "WhisperModel", _fw_stub(calls, fail_first=True).WhisperModel)
    monkeypatch.setattr(T.Transcriber, "check_ffmpeg", lambda self: True)
    app_config.asr.compute_type = "float32"  # validní, ale load 1. pokus padne
    T.Transcriber(app_config).transcribe(str(wav))
    assert [c["compute_type"] for c in calls["construct"]] == ["float32", "int8"]


def test_align_words_flag(tw_stub, app_config, tmp_path):
    T, calls = tw_stub
    from conftest import make_wav
    wav = make_wav(tmp_path / "a.wav", seconds=0.5)
    assert calls == {"construct": [], "transcribe": []}
    T.Transcriber(app_config).transcribe(str(wav))
    assert calls["transcribe"][0]["word_timestamps"] is False
    app_config.asr.align_words = True
    T.Transcriber(app_config).transcribe(str(wav))
    assert calls["transcribe"][-1]["word_timestamps"] is True


def test_engine_fallback(tw_stub, app_config, tmp_path):
    T, calls = tw_stub
    from conftest import make_wav
    wav = make_wav(tmp_path / "p.wav", seconds=0.5)
    app_config.asr.engine = "parakeet"  # router až FÁZE 5
    tr = T.Transcriber(app_config)
    assert tr._resolve_engine() == "parakeet"
    tr.transcribe(str(wav))  # bez výjimky, faster-whisper fallback
    assert len(calls["construct"]) == 1


# ---------- pyannote batch ----------

def _pa_stub_chain(monkeypatch, calls):
    pa = types.ModuleType("pyannote")
    audio = types.ModuleType("pyannote.audio")
    core = types.ModuleType("pyannote.audio.core")
    task = types.ModuleType("pyannote.audio.core.task")

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kw):
            self = cls()
            self.set_attrs = {}
            return self

        def instantiate(self, params):
            pass

        def __setattr__(self, k, v):
            if k in ("segmentation_batch_size", "embedding_batch_size"):
                calls[k] = v
            object.__setattr__(self, k, v)

        def to(self, device):
            return self

    audio.Pipeline = FakePipeline
    pa.audio = audio
    audio.core = core
    core.task = task
    task.Specifications = task.Problem = task.Resolution = object
    monkeypatch.setitem(sys.modules, "pyannote", pa)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core", core)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.task", task)


def test_pyannote_batch_applied(app_config, monkeypatch):
    calls = {}
    _pa_stub_chain(monkeypatch, calls)
    app_config.asr.batch_size = 4
    from src.diarization.pyannote_diarizer import PyAnnoteDiarizer
    d = PyAnnoteDiarizer(app_config)
    d.initialize()
    assert calls == {"segmentation_batch_size": 4, "embedding_batch_size": 4}


def test_pyannote_batch_default_untouched(app_config, monkeypatch):
    calls = {}
    _pa_stub_chain(monkeypatch, calls)
    app_config.asr.batch_size = 1
    from src.diarization.pyannote_diarizer import PyAnnoteDiarizer
    PyAnnoteDiarizer(app_config).initialize()
    assert calls == {}


# ---------- pipeline per-file cleanup + memory ----------

def _pipe4(app_config, name):
    from src.core.pipeline import Pipeline
    from src.utils.logger import setup_logger
    pipe = Pipeline.__new__(Pipeline)
    pipe.config = app_config
    pipe.logger = setup_logger(name, "WARNING", None)
    return pipe


def test_process_file_finally_cleanup(app_config, tmp_path):
    from conftest import make_wav
    from unittest.mock import MagicMock
    import torch as _t
    wav = make_wav(tmp_path / "f.wav", seconds=0.5)
    converted = tmp_path / "f.wav"  # existuje; cleanup ho přes converted_file_path smaže
    app_config.diarization.enabled = False
    pipe = _pipe4(app_config, "test-p4-ok")
    from src.audio.processor import AudioProcessor
    pipe.audio_processor = AudioProcessor(app_config)
    pipe.audio_processor.converted_file_path = str(converted)
    pipe.audio_processor.load_audio = lambda *a, **k: (_t.empty(0), 16000, str(converted))
    pipe.diarizer = MagicMock()
    pipe.transcriber = MagicMock()
    pipe.transcriber.transcribe.return_value = {
        "text": "hi", "segments": [{"start": 0.0, "end": 0.5, "text": "hi"}]}
    pipe.process_file(str(wav))
    assert not os.path.exists(str(converted))  # finally cleanup proběhl


def test_process_file_cleanup_on_error(app_config, tmp_path):
    from conftest import make_wav
    from unittest.mock import MagicMock
    import os
    import torch as _t
    wav = make_wav(tmp_path / "g.wav", seconds=0.5)
    converted = tmp_path / "g.wav"
    app_config.diarization.enabled = False
    pipe = _pipe4(app_config, "test-p4-err")
    from src.audio.processor import AudioProcessor
    pipe.audio_processor = AudioProcessor(app_config)
    pipe.audio_processor.converted_file_path = str(converted)
    pipe.audio_processor.load_audio = lambda *a, **k: (_t.empty(0), 16000, str(converted))
    pipe.diarizer = MagicMock()
    pipe.transcriber = MagicMock()
    pipe.transcriber.transcribe.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        pipe.process_file(str(wav))
    assert not os.path.exists(str(converted))  # cleanup i při pádu


def test_memory_summary_no_crash(app_config):
    pipe = _pipe4(app_config, "test-p4-mem")
    assert pipe._log_memory_summary("test") is None
