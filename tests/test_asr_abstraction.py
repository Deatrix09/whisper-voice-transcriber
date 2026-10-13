# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""P0 0.4 ASR abstrakce: ASRProvider, factory, lazy pipeline, unload."""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


def test_provider_interface():
    """Oba transcribery implementují stejné rozhraní beze změny signatur."""
    from src.transcription.base_transcriber import ASRProvider
    from src.transcription.transcriber import Transcriber
    from src.transcription.parakeet_transcriber import ParakeetTranscriber
    import inspect
    assert issubclass(Transcriber, ASRProvider)
    assert issubclass(ParakeetTranscriber, ASRProvider)
    for cls in (Transcriber, ParakeetTranscriber):
        assert list(inspect.signature(cls.__init__).parameters) == ["self", "config"]
        assert list(inspect.signature(cls.transcribe).parameters) == ["self", "audio_path"]
        assert callable(getattr(cls, "unload"))
        assert callable(getattr(cls, "initialize_model"))


def test_unload_default_idempotent():
    from src.transcription.base_transcriber import ASRProvider

    class Dummy(ASRProvider):
        def __init__(self):
            self.model = object()

        def transcribe(self, audio_path):
            return {}

    d = Dummy()
    d.unload()
    assert d.model is None
    d.unload()  # podruhé bez pádu
    SimpleNamespace().model = None  # kouřový dotyk
    ASRProvider.unload(SimpleNamespace(model=None))


def test_factory_light_import():
    """asr_factory jde importovat bez torch/nemo/whisper (lazy engine)."""
    code = ("import sys; "
            "[sys.modules.__setitem__(m, None) for m in "
            "('torch', 'faster_whisper', 'nemo', 'nemo.collections', 'nemo.collections.asr')]; "
            "import src.transcription.asr_factory; print('factory-ok')")
    import subprocess
    import pathlib
    root = str(pathlib.Path(__file__).resolve().parents[1])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=root, timeout=120)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "factory-ok" in r.stdout


def _fw_transcriber_stub(monkeypatch, calls):
    import types as _t
    mod = _t.ModuleType("faster_whisper")

    class FakeModel:
        def __init__(self, model_size, device="auto", compute_type="default", **kw):
            calls.append(model_size)

    mod.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    import src.transcription.transcriber as T
    monkeypatch.setattr(T, "WhisperModel", FakeModel)
    monkeypatch.setattr(T.Transcriber, "check_ffmpeg", lambda self: True)


def test_factory_returns_initialized_provider(app_config, monkeypatch):
    from src.transcription.asr_factory import create_asr_provider
    from src.transcription.transcriber import Transcriber
    calls = []
    _fw_transcriber_stub(monkeypatch, calls)
    app_config.asr.engine = "faster-whisper"
    t = create_asr_provider(app_config)
    assert isinstance(t, Transcriber)
    assert calls == ["large-v3-turbo"]  # eager init hned ve factory
    assert t.model is not None


def test_pipeline_lazy_components(app_config):
    """Pipeline() bez těžkých modelů; vytvoří se až v process_file."""
    from src.core.pipeline import Pipeline
    app_config.diarization.enabled = False
    pipe = Pipeline(app_config)
    assert pipe.transcriber is None and pipe.diarizer is None
    assert pipe.audio_processor is not None


def test_pipeline_lazy_transcribe_and_unload(app_config, tmp_path, monkeypatch):
    from conftest import make_wav
    from src.core.pipeline import Pipeline
    wav = make_wav(tmp_path / "l.wav", seconds=0.5)
    app_config.diarization.enabled = False
    pipe = Pipeline(app_config)
    fake = MagicMock()
    fake.transcribe.return_value = {
        "text": "hi", "segments": [{"start": 0.0, "end": 0.5, "text": "hi"}]}
    with patch("src.transcription.asr_factory.create_asr_provider",
               return_value=fake) as factory:
        out = pipe.process_file(str(wav))
    factory.assert_called_once_with(app_config)
    assert pipe.transcriber is fake  # cachováno pro další soubor
    assert out[0]["text"] == "hi"
    fake.unload.assert_called_once()  # finally unload i při úspěchu


def test_pipeline_unload_on_error(app_config, tmp_path, monkeypatch):
    from conftest import make_wav
    from src.core.pipeline import Pipeline
    wav = make_wav(tmp_path / "e.wav", seconds=0.5)
    app_config.diarization.enabled = False
    pipe = Pipeline(app_config)
    fake = MagicMock()
    fake.transcribe.side_effect = RuntimeError("boom")
    with patch("src.transcription.asr_factory.create_asr_provider", return_value=fake):
        with pytest.raises(RuntimeError):
            pipe.process_file(str(wav))
    fake.unload.assert_called_once()  # finally i při pádu


def test_release_delegates_unload():
    from unittest.mock import MagicMock
    from src.core.pipeline import Pipeline
    fake = MagicMock()
    fake.model = object()
    pipe = SimpleNamespace(transcriber=fake, diarizer=SimpleNamespace(diarizer=None))
    Pipeline.release_model_resources(pipe)
    fake.unload.assert_called_once()
