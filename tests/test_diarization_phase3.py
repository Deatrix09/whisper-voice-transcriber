"""Validační testy pro FÁZI 3: community-1, Sortformer, Nemotron-3, factory.

Těžké knihovny (pyannote modely, NeMo, transformers) se nestahují:
pyannote Pipeline se patchuje, nemo/transformers stubují v sys.modules.
"""
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


# ---------- shared helpers ----------

def _ann(tracks):
    """Fake pyannote Annotation: tracks=[(start, end, label)]."""
    class FakeAnn:
        def itertracks(self, yield_label=False):
            for s, e, lab in tracks:
                yield SimpleNamespace(start=s, end=e), None, lab
    return FakeAnn()


class _FakeOut:
    def __init__(self, exclusive, regular):
        self.exclusive_speaker_diarization = exclusive
        self.speaker_diarization = regular


def _pyannote_config(app_config, model, num_speakers=None, token="tok"):
    app_config.diarization.enabled = True
    app_config.diarization.module = "pyannote"
    app_config.diarization.num_speakers = num_speakers
    app_config.diarization.pyannote.model = model
    app_config.diarization.pyannote.auth_token = token
    return app_config


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Stub celého pyannote řetězce (torchaudio tu není, real import padá)."""
    import src.diarization.pyannote_diarizer  # noqa: F401 (light top-level)
    calls = {}

    class FakePipeline:
        def __init__(self):
            self.instantiate_params = None
            self.calls = []

        @classmethod
        def from_pretrained(cls, model_id, **kw):
            calls["from"] = {"model_id": model_id, "kw": kw}
            return cls()

        def instantiate(self, params):
            self.instantiate_params = params
            calls["instantiate"] = params

        def to(self, device):
            calls["to"] = str(device)
            return self

        def __call__(self, wav, **kw):
            self.calls.append({"wav": wav, "kw": kw})
            calls["call"] = self.calls[-1]
            return _FakeOut(
                _ann([(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]),
                _ann([(0.0, 5.0, "SPEAKER_00")]),
            )

    pa = types.ModuleType("pyannote")
    audio = types.ModuleType("pyannote.audio")
    core = types.ModuleType("pyannote.audio.core")
    task = types.ModuleType("pyannote.audio.core.task")
    audio.Pipeline = FakePipeline
    pa.audio = audio
    audio.core = core
    core.task = task
    task.Specifications = task.Problem = task.Resolution = object
    monkeypatch.setitem(sys.modules, "pyannote", pa)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core", core)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.task", task)
    return calls


# ---------- base helpers ----------

def test_normalize_speaker_label():
    from src.diarization.base_diarizer import BaseDiarizer as B
    assert B.normalize_speaker_label("speaker_0") == "SPEAKER_00"
    assert B.normalize_speaker_label("SPEAKER_1") == "SPEAKER_01"
    assert B.normalize_speaker_label("SPEAKER_00") == "SPEAKER_00"
    assert B.normalize_speaker_label("UNKNOWN") == "UNKNOWN"


def test_resolve_overlaps():
    from src.diarization.base_diarizer import BaseDiarizer as B
    segs = [
        {"speaker": "A", "start": 0.0, "end": 2.0, "duration": 2.0},
        {"speaker": "B", "start": 1.0, "end": 3.0, "duration": 2.0},
        {"speaker": "B", "start": 0.5, "end": 1.0, "duration": 0.5},  # uvnitř -> drop
    ]
    out = B.resolve_overlaps(segs)
    assert [(s["start"], s["end"]) for s in out] == [(0.0, 2.0), (2.0, 3.0)]
    # unie časů zachována
    assert sum(s["duration"] for s in out) == pytest.approx(3.0)
    # idempotence
    assert B.resolve_overlaps(out) == out


# ---------- pyannote community-1 ----------

def test_community_uses_token_and_minmax(app_config, tmp_path, fake_pipeline, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    _pyannote_config(app_config, "pyannote/speaker-diarization-community-1",
                     num_speakers=2, token="tok123")
    from conftest import make_wav
    from src.diarization.pyannote_diarizer import PyAnnoteDiarizer
    wav = make_wav(tmp_path / "c.wav", seconds=0.5)
    out = PyAnnoteDiarizer(app_config).process(str(wav))
    assert fake_pipeline["from"]["kw"] == {"token": "tok123"}  # NE use_auth_token
    assert "clustering" not in fake_pipeline["instantiate"]  # jen segmentace (4.x klíče)
    assert fake_pipeline["call"]["kw"] == {"min_speakers": 2, "max_speakers": 2}
    # exclusive anotace preferována (2 segmenty, ne 1 dlouhý regular)
    assert [(s["start"], s["end"]) for s in out] == [(0.0, 1.0), (1.0, 2.0)]


def test_legacy_31_path(app_config, tmp_path, fake_pipeline, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    _pyannote_config(app_config, "pyannote/speaker-diarization-3.1",
                     num_speakers=3, token="tok9")
    from conftest import make_wav
    from src.diarization.pyannote_diarizer import PyAnnoteDiarizer
    wav = make_wav(tmp_path / "l.wav", seconds=0.5)
    PyAnnoteDiarizer(app_config).process(str(wav))
    assert fake_pipeline["from"]["kw"] == {"use_auth_token": "tok9"}
    assert fake_pipeline["call"]["kw"] == {"num_speakers": 3}
    assert "clustering" in fake_pipeline["instantiate"]


def test_no_token_clear_error(app_config, monkeypatch):
    _pyannote_config(app_config, "pyannote/speaker-diarization-community-1", token="")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    from src.diarization.pyannote_diarizer import PyAnnoteDiarizer
    with pytest.raises(ValueError, match="token"):
        PyAnnoteDiarizer(app_config).initialize()


def test_token_env_fallback(app_config, tmp_path, fake_pipeline, monkeypatch):
    _pyannote_config(app_config, "pyannote/speaker-diarization-community-1", token="")
    monkeypatch.setenv("HF_TOKEN", "env-tok")
    from conftest import make_wav
    from src.diarization.pyannote_diarizer import PyAnnoteDiarizer
    wav = make_wav(tmp_path / "e.wav", seconds=0.5)
    PyAnnoteDiarizer(app_config).process(str(wav))
    assert fake_pipeline["from"]["kw"] == {"token": "env-tok"}


# ---------- sortformer ----------

def _nemo_stub(captured, raw):
    nemo = types.ModuleType("nemo")
    collections = types.ModuleType("nemo.collections")
    asr = types.ModuleType("nemo.collections.asr")
    models = types.ModuleType("nemo.collections.asr.models")

    class FakeSortformer:
        @classmethod
        def from_pretrained(cls, model_id):
            captured["model_id"] = model_id
            return cls()

        def eval(self):
            return self

        def to(self, device):
            captured["to"] = str(device)
            return self

        def diarize(self, audio, batch_size=1):
            captured["batch"] = batch_size
            captured["audio"] = list(audio)
            return raw

    models.SortformerEncLabelModel = FakeSortformer
    asr.models = models
    collections.asr = asr
    nemo.collections = collections
    return {"nemo": nemo, "nemo.collections": collections,
            "nemo.collections.asr": asr, "nemo.collections.asr.models": models}


def _sortformer_config(app_config):
    app_config.diarization.enabled = True
    app_config.diarization.module = "sortformer"
    return app_config


def test_sortformer_shapes_and_limit(app_config, tmp_path):
    _sortformer_config(app_config)
    from conftest import make_wav
    from src.diarization.sortformer_diarizer import SortformerDiarizer
    wav = make_wav(tmp_path / "s.wav", seconds=1.0)
    raw = [
        {"start": 0.0, "end": 1.0, "speaker": "speaker_0"},
        [1.0, 2.0, "speaker_1"],
        SimpleNamespace(start=2.0, end=3.0, label="speaker_2"),
        {"start": 5.0, "end": 4.0, "speaker": "x"},  # nevalidní -> pryč
        "nesmysl",  # uninterpretable -> pryč
    ]
    captured = {}
    with patch.dict(sys.modules, _nemo_stub(captured, raw)):
        out = SortformerDiarizer(app_config).process(str(wav))
    assert captured["model_id"] == "nvidia/diar_streaming_sortformer_4spk-v2.1"
    assert [(s["speaker"], s["start"], s["end"]) for s in out] == [
        ("SPEAKER_00", 0.0, 1.0), ("SPEAKER_01", 1.0, 2.0), ("SPEAKER_02", 2.0, 3.0)]


def test_sortformer_missing_nemo(app_config):
    _sortformer_config(app_config)
    from src.diarization.sortformer_diarizer import SortformerDiarizer
    assert "nemo" not in sys.modules
    with pytest.raises(RuntimeError, match="NeMo"):
        SortformerDiarizer(app_config).initialize()


# ---------- nemotron-3 ----------

def _transformers_stub(captured, segments):
    mod = types.ModuleType("transformers")

    class FakeBatch(dict):
        @property
        def attention_mask(self):
            import torch
            return torch.ones(1, 10)

        def to(self, *a, **k):
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            captured["model_id"] = model_id
            return cls()

        def __call__(self, audio, sampling_rate):
            captured["sr"] = sampling_rate
            captured["audio_len"] = len(audio)
            return FakeBatch({"input_values": [1]})

        def extract_speaker_dict(self, logits, mask):
            return [segments]

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

        def eval(self):
            return self

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

        def to(self, *a, **k):
            return self

        def __call__(self, **kw):
            return SimpleNamespace(logits=torch.zeros(1, 10, 8))

    mod.AutoProcessor = FakeProcessor
    mod.AutoModelForAudioFrameClassification = FakeModel
    return {"transformers": mod}


def test_nemotron_offline_segments(app_config, tmp_path):
    app_config.diarization.enabled = True
    app_config.diarization.module = "nemotron-3"
    from conftest import make_wav
    from src.diarization.nemotron_diarizer import NemotronDiarizer
    wav = make_wav(tmp_path / "n.wav", seconds=1.0)
    captured = {}
    segs = [{"Speaker": 0, "Start": 0.0, "End": 0.5},
            {"Speaker": 3, "Start": 0.5, "End": 1.0}]
    with patch.dict(sys.modules, _transformers_stub(captured, segs)):
        out = NemotronDiarizer(app_config).process(str(wav))
    assert captured["model_id"] == "nvidia/Nemotron-3-Diarization"
    assert [(s["speaker"], s["start"], s["end"]) for s in out] == [
        ("SPEAKER_00", 0.0, 0.5), ("SPEAKER_03", 0.5, 1.0)]


def test_nemotron_missing_transformers(app_config):
    app_config.diarization.module = "nemotron-3"
    from src.diarization.nemotron_diarizer import NemotronDiarizer
    assert "transformers" not in sys.modules
    with pytest.raises(RuntimeError, match="transformers"):
        NemotronDiarizer(app_config).initialize()


# ---------- factory + config + release ----------

def test_factory_branches(app_config):
    from src.diarization.diarizer_factory import DiarizationFactory
    from src.diarization.sortformer_diarizer import SortformerDiarizer
    from src.diarization.nemotron_diarizer import NemotronDiarizer
    from src.diarization.pyannote_diarizer import PyAnnoteDiarizer
    app_config.diarization.enabled = True
    app_config.diarization.module = "sortformer"
    assert isinstance(DiarizationFactory.create_diarizer(app_config), SortformerDiarizer)
    app_config.diarization.module = "nemotron-3"
    assert isinstance(DiarizationFactory.create_diarizer(app_config), NemotronDiarizer)
    app_config.diarization.pyannote.enabled = True
    app_config.diarization.module = "pyannote"
    assert isinstance(DiarizationFactory.create_diarizer(app_config), PyAnnoteDiarizer)
    app_config.diarization.module = "neexistuje"
    with pytest.raises(ValueError, match="Unknown diarization module"):
        DiarizationFactory.create_diarizer(app_config)
    app_config.diarization.enabled = False
    assert DiarizationFactory.create_diarizer(app_config) is None


def test_config_backward_compat(tmp_path):
    """Staré YAML bez sortformer/nemotron sekcí -> None (FÁZE 4 je rozšíří)."""
    import yaml
    from src.utils.config import AppConfig
    repo_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config", "config.yaml")
    with open(repo_cfg, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # nové sekce z aktuálního configu fungují...
    assert data["diarization"]["sortformer"]["model"].startswith("nvidia/")
    assert data["diarization"]["nemotron"]["model"].startswith("nvidia/")
    cfg = AppConfig.from_yaml(repo_cfg)
    assert cfg.diarization.sortformer.model.startswith("nvidia/")
    # ...a staré bez nich dají None
    del data["diarization"]["sortformer"]
    del data["diarization"]["nemotron"]
    old = tmp_path / "old.yaml"
    old.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg_old = AppConfig.from_yaml(str(old))
    assert cfg_old.diarization.sortformer is None
    assert cfg_old.diarization.nemotron is None


def test_release_frees_sortformer_model():
    from types import SimpleNamespace
    from src.core.pipeline import Pipeline
    inner = SimpleNamespace(pipeline=None, model=object())
    pipe = SimpleNamespace(transcriber=SimpleNamespace(model=None),
                           diarizer=SimpleNamespace(diarizer=inner))
    Pipeline.release_model_resources(pipe)
    assert inner.model is None
    assert pipe.diarizer.diarizer is None
