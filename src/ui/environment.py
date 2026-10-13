# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Detekce CUDA a ffmpeg. Import stránky nesmí spadnout, když chybí."""

import shutil


def probe_environment() -> dict:
    ffmpeg = shutil.which("ffmpeg") is not None
    cuda = False
    gpu_name = None
    torch_ok = False
    try:
        import torch
        torch_ok = True
        cuda = bool(torch.cuda.is_available())
        if cuda:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = None
    except Exception:
        torch_ok = False
        cuda = False
        gpu_name = None
    return {"ffmpeg": ffmpeg, "cuda": cuda, "gpu_name": gpu_name, "torch": torch_ok}


def can_start(env: dict, audio_path: str) -> bool:
    return bool(env.get("ffmpeg") and env.get("torch") and audio_path)
