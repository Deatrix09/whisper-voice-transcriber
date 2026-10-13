# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""P0 0.4: společné rozhraní ASR enginů.

Nový engine (parakeet, ...) přidáš novým souborem + jednou větví
v asr_factory.create_asr_provider - BEZ zásahu do pipeline.py.
"""
from abc import ABC, abstractmethod
from typing import Dict


class ASRProvider(ABC):
    """Kontrakt: transcribe() + unload(). Atributy config/logger/model/device."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> Dict:
        """Vrať {"text": str, "segments": [{"start","end","text"}], "language": str}."""
        raise NotImplementedError

    def unload(self) -> None:
        """Uvolni model z RAM/VRAM. Idempotentní (volá pipeline finally)."""
        import gc
        try:
            if getattr(self, "model", None) is not None:
                del self.model
                self.model = None
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
