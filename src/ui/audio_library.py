# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bezpečné uložení uploadu a výpis audia v data/input."""

import os
from pathlib import Path

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".mp4"}


def safe_audio_basename(name: str) -> str:
    """Jen název souboru, bez cesty. Nepovolené znaky se nahradí."""
    raw = str(name or "").replace("\\", "/").split("/")[-1]
    raw = raw.replace("\x00", "").strip()
    cleaned = []
    for char in raw:
        if char.isalnum() or char in "._- ":
            cleaned.append(char)
        else:
            cleaned.append("_")
    base = "".join(cleaned).strip().strip(".")
    if not base or base in {".", ".."}:
        raise ValueError("Neplatný název souboru.")
    suffix = Path(base).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise ValueError("Nepodporovaný typ souboru. Povolené: mp3, wav, m4a, flac, ogg, aac, mp4.")
    return base


def input_dir() -> Path:
    directory = Path(os.getcwd()) / "data" / "input"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_upload(filename: str, data: bytes) -> str:
    base = safe_audio_basename(filename)
    path = input_dir() / base
    path.write_bytes(data)
    return str(path.resolve())


def list_input_audio() -> list:
    directory = Path(os.getcwd()) / "data" / "input"
    if not directory.is_dir():
        return []
    files = []
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(str(path.resolve()))
    return sorted(files, key=lambda item: Path(item).name.lower())
