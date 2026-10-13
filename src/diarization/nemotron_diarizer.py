# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""NVIDIA Nemotron-3-Diarization přes 🤗 transformers (ne NeMo!).

Model: nvidia/Nemotron-3-Diarization (open-weight, openmdw-1.1).
- streaming + offline inference, až 8 speakerů, arrival-order labely.
- API (dle HF model card): AutoProcessor + AutoModelForAudioFrameClassification,
  logits (1, frames, 8) po 10 ms -> processor.extract_speaker_dict().
- Bez device_map (nevyžaduje accelerate): ruční .to(device) + bf16 na CUDA.
- transformers se importuje lazy až v initialize(); bez něj jasná hláška.
"""
from typing import Dict, List
import numpy as np
import torch
from src.diarization.base_diarizer import BaseDiarizer


class NemotronDiarizer(BaseDiarizer):
    DEFAULT_MODEL = "nvidia/Nemotron-3-Diarization"
    MAX_SPEAKERS = 8

    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.processor = None

    def _model_id(self) -> str:
        section = getattr(self.config.diarization, "nemotron", None)
        configured = getattr(section, "model", None) if section is not None else None
        return configured or self.DEFAULT_MODEL

    def initialize(self):
        """Load model + processor 1x (lazy singleton na instanci)."""
        if self.model is not None:
            return self.model
        try:
            from transformers import (
                AutoModelForAudioFrameClassification, AutoProcessor)
        except ImportError as e:
            raise RuntimeError(
                "transformers není nainstalováno (potřeba pro nemotron-3 větev). "
                "Spusť: pip install -U transformers (verze s podporou "
                "nvidia/Nemotron-3-Diarization)."
            ) from e
        try:
            model_id = self._model_id()
            self.logger.info(f"Loading Nemotron-3-Diarization: {model_id} (max 8 speakers)")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForAudioFrameClassification.from_pretrained(model_id)
            self.model.eval()
            from src.utils.config import resolve_device
            try:
                device = resolve_device(self.config.performance.device)
            except ValueError:
                device = "cpu"
            try:
                self.model = self.model.to(torch.device(device))
                if device == "cuda":
                    try:
                        self.model = self.model.to(dtype=torch.bfloat16)
                    except Exception as e:
                        self.logger.warning(f"bf16 nepodporováno ({e}), jedu fp32")
                self.logger.info(f"Nemotron model on {device}")
            except Exception as e:
                self.logger.warning(f"Device přesun selhal ({e}), jedu na CPU")
            self.logger.info("Successfully initialized Nemotron diarizer")
            return self.model
        except Exception as e:
            self.logger.error(f"Failed to initialize Nemotron diarizer: {str(e)}")
            raise

    def _load_mono(self, audio_path: str):
        """numpy float32 mono + sr (processor si případně resampluje sám)."""
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        mono = np.mean(audio, axis=1) if audio.shape[1] > 1 else audio[:, 0]
        return mono.astype(np.float32), int(sr)

    def process(self, audio_path: str) -> List[Dict[str, any]]:
        try:
            if self.model is None:
                self.initialize()
            self.logger.info(f"Starting Nemotron-3 diarization for: {audio_path}")
            audio, sr = self._load_mono(audio_path)
            inputs = self.processor(audio, sampling_rate=sr)
            try:
                target_device = next(self.model.parameters()).device
            except StopIteration:
                target_device = torch.device("cpu")
            try:
                model_dtype = next(self.model.parameters()).dtype
            except StopIteration:
                model_dtype = torch.float32
            inputs = inputs.to(target_device, dtype=model_dtype)
            with torch.inference_mode():
                logits = self.model(**inputs).logits  # (1, frames, 8)
            raw = self.processor.extract_speaker_dict(
                logits, inputs.attention_mask)[0]
            segments: List[Dict[str, any]] = []
            for item in raw or []:
                try:
                    start = float(item["Start"])
                    end = float(item["End"])
                    label = item.get("Speaker", item.get("speaker"))
                except (KeyError, TypeError, ValueError):
                    self.logger.debug(f"Skipping uninterpretable item: {item!r}")
                    continue
                if end <= start:
                    continue
                segments.append({
                    "speaker": self.normalize_speaker_label(f"speaker_{label}"),
                    "start": start,
                    "end": end,
                    "duration": end - start,
                })
            n_speakers = len({s["speaker"] for s in segments})
            if n_speakers > self.MAX_SPEAKERS:
                self.logger.warning(
                    f"Detected {n_speakers} speakers > model limit {self.MAX_SPEAKERS}.")
            segments = self.resolve_overlaps(segments)
            self.logger.info(
                f"Nemotron done: {len(segments)} segments, {n_speakers} speakers")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return segments
        except Exception as e:
            self.logger.error(f"Nemotron diarization failed: {str(e)}")
            raise
