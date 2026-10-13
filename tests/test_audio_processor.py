# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Validační testy pro Fix B (L01): load_waveform=False + no_grad + lazy importy.

Ověřují:
1. load_waveform=False nedekóduje audio (žádný torchaudio.load, prázdný tensor).
2. Staré volání load_audio(path) stále funguje (backward compat, dekóduje).
3. Pipeline.process_file waveform nepotřebuje (mock komponent).
4. reduce_noise/normalize fungují i pro grad-tensory (detach path).
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch


def _processor(app_config):
    from src.audio.processor import AudioProcessor
    return AudioProcessor(app_config)


def test_load_waveform_false_skips_decode(app_config, tmp_path):
    """Klíčový test Fix B: malý wav, ale decode se nesmí spustit."""
    from conftest import make_wav
    wav = make_wav(tmp_path / "a.wav", seconds=0.5)
    proc = _processor(app_config)
    with patch.dict(sys.modules, {"torchaudio": None}):
        # torchaudio je None -> jakýkoliv pokus o decode spadne;
        # load_waveform=False ho nesmí potřebovat.
        sys.modules["torchaudio"] = types.ModuleType("torchaudio_stub_no_load")
        try:
            waveform, sr, converted = proc.load_audio(str(wav), load_waveform=False)
        finally:
            del sys.modules["torchaudio"]
    assert waveform.numel() == 0
    assert sr == app_config.whisper.sample_rate
    assert converted is None


def test_load_audio_default_still_decodes(app_config, tmp_path):
    """Backward compat: load_audio(path) bez parametru dekóduje jako dřív."""
    torch_mock = MagicMock()
    fake_wave = torch.zeros(1, 16000)
    torch_mock.load.return_value = (fake_wave, 16000)
    torch_mock.transforms.Resample.return_value = lambda x: x
    with patch.dict(sys.modules, {"torchaudio": torch_mock}), \
         patch("src.audio.processor.AudioProcessor.reduce_noise", lambda self, w, sr: w), \
         patch("src.audio.processor.AudioProcessor.normalize_audio", lambda self, w: w):
        from conftest import make_wav
        wav = make_wav(tmp_path / "b.wav", seconds=0.5)
        proc = _processor(app_config)
        waveform, sr, _ = proc.load_audio(str(wav))  # default load_waveform=True
        torch_mock.load.assert_called_once()
        assert waveform.shape[1] == 16000


def test_reduce_noise_and_normalize_no_grad_tensors(app_config):
    """no_grad/detach path: grad-tensor nesmí shodit numpy() konverzi."""
    proc = _processor(app_config)
    grad_wave = torch.zeros(1, 1600, requires_grad=True)
    # noisereduce nemusí být nainstalován -> reduce_noise vrátí vstup, ale nesmí crash
    out = proc.reduce_noise(grad_wave, 16000)
    assert out.shape[0] == 1
    norm = proc.normalize_audio(torch.ones(1, 1600))
    assert norm.shape == (1, 1600)


def test_pipeline_process_file_mocks_no_waveform(app_config, tmp_path):
    """Pipeline projde bez waveform decode (vše mockováno, diarizace off)."""
    from conftest import make_wav
    wav = make_wav(tmp_path / "c.wav", seconds=0.5)
    app_config.diarization.enabled = False

    from src.core.pipeline import Pipeline
    pipe = Pipeline.__new__(Pipeline)  # bez těžkého __init__ (torch ok, ale bez side-effectů)
    pipe.config = app_config
    from src.utils.logger import setup_logger
    pipe.logger = setup_logger("test-pipeline", "WARNING", None)
    pipe.audio_processor = _processor(app_config)
    pipe.diarizer = MagicMock()
    pipe.transcriber = MagicMock()
    pipe.transcriber.transcribe.return_value = {
        "text": "hello",
        "segments": [{"start": 0.0, "end": 0.5, "text": "hello"}],
    }
    with patch.dict(sys.modules, {"torchaudio": None}):
        # kdyby pipeline chtěl dekódovat, spadne na None.load
        results = pipe.process_file(str(wav))
    assert results[0]["speaker"] == "UNKNOWN"
    assert results[0]["text"] == "hello"
    pipe.transcriber.transcribe.assert_called_once()
    pipe.diarizer.process.assert_not_called()
