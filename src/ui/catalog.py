# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Read-only výpisy výstupů a složek konfigurace. YAML se nečte do UI."""

import os
from pathlib import Path


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def config_root() -> Path:
    return (Path(os.getcwd()) / "config").resolve()


def list_config_folders() -> list:
    root = config_root()
    if not root.is_dir():
        return []
    folders = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        has_yaml = any(path.glob("*.yaml")) or any(path.glob("*.yml"))
        if has_yaml:
            folders.append(str(path.resolve()))
    return sorted(folders, key=lambda item: Path(item).name.lower())


def list_yaml_names(folder: str) -> list:
    """Jen názvy souborů, seřazené. Obsah YAML se nečte."""
    root = config_root()
    candidate = Path(folder).resolve()
    if candidate != root and not _under(candidate, root):
        raise ValueError("Složka musí být uvnitř config/.")
    if not candidate.is_dir():
        return []
    names = []
    for path in candidate.iterdir():
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            names.append(path.name)
    return sorted(names, key=str.lower)


def yaml_paths(folder: str) -> list:
    root = config_root()
    candidate = Path(folder).resolve()
    if candidate != root and not _under(candidate, root):
        raise ValueError("Složka musí být uvnitř config/.")
    paths = []
    for name in list_yaml_names(candidate):
        paths.append(str((candidate / name).resolve()))
    return paths


def list_output_files() -> list:
    directory = Path(os.getcwd()) / "data" / "output"
    if not directory.is_dir():
        return []
    files = [str(path.resolve()) for path in directory.iterdir() if path.is_file()]
    return sorted(files, key=lambda item: os.path.getmtime(item), reverse=True)


def list_comparison_runs() -> list:
    root = Path(os.getcwd()) / "data" / "algorithm_output"
    if not root.is_dir():
        return []
    runs = []
    for audio_dir in root.iterdir():
        if not audio_dir.is_dir():
            continue
        for run_dir in audio_dir.iterdir():
            if run_dir.is_dir():
                runs.append({
                    "audio": audio_dir.name,
                    "stamp": run_dir.name,
                    "path": str(run_dir.resolve()),
                })
    runs.sort(key=lambda item: (item["stamp"], item["audio"]), reverse=True)
    return runs


def list_run_files(run_dir: str) -> list:
    root = (Path(os.getcwd()) / "data" / "algorithm_output").resolve()
    candidate = Path(run_dir).resolve()
    if not _under(candidate, root) or not candidate.is_dir():
        return []
    files = [str(path.resolve()) for path in candidate.iterdir() if path.is_file()]
    return sorted(files, key=lambda item: Path(item).name.lower())


def readable_data_file(path: str) -> Path:
    """Soubor smí být jen v data/output nebo data/algorithm_output."""
    base = Path(os.getcwd()).resolve()
    allowed = [base / "data" / "output", base / "data" / "algorithm_output"]
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(str(candidate))
    if not any(_under(candidate, root) for root in allowed):
        raise ValueError("Náhled je jen pro uložené výstupy.")
    return candidate
