# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""P0 0.2 Závislosti: čitelné soubory, split base/cuda, pyproject+lock."""
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _req_lines(name):
    with open(os.path.join(REPO, name), encoding="utf-8") as f:  # padne pokud není UTF-8
        return [l.split("#", 1)[0].strip() for l in f.read().splitlines()]


def _parsed(reqs):
    from packaging.requirements import Requirement
    out = []
    for r in reqs:
        if not r or r.startswith("-r") or r.startswith("--"):
            continue
        out.append(Requirement(r))
    return out


def test_requirements_utf8_and_valid():
    reqs = _req_lines("requirements.txt")
    parsed = _parsed(reqs)
    assert len(parsed) > 100
    from packaging.utils import canonicalize_name
    names = {canonicalize_name(r.name) for r in parsed}
    assert "torch" not in names and "torchaudio" not in names  # jen v cuda/cpu
    assert "python-dotenv" in names  # P0 0.1 .env flow
    assert "faster-whisper" in names and "pyannote-audio" in names


def test_cuda_cpu_split():
    for f, suffix in (("requirements-cuda.txt", "cu126"), ("requirements-cpu.txt", "cpu")):
        reqs = _req_lines(f)
        assert any(r.startswith("-r requirements.txt") for r in reqs)
        assert any("extra-index-url" in r and ("/cu126" in r or "/cpu" in r) for r in reqs)
        names = {r.name.lower() for r in _parsed(reqs)}
        assert {"torch", "torchaudio"} <= names
        torch_pin = next(r for r in _parsed(reqs) if r.name.lower() == "torch")
        assert suffix in str(torch_pin.specifier), torch_pin


def test_pyproject_and_lock():
    import tomllib
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as f:
        data = tomllib.load(f)
    assert data["project"]["requires-python"].startswith(">=3.11")
    deps = " ".join(data["project"]["dependencies"]).lower()
    for must in ("torch==", "faster-whisper", "pyannote.audio", "pyyaml", "python-dotenv"):
        assert must in deps, must
    lock = os.path.join(REPO, "uv.lock")
    assert os.path.getsize(lock) > 100_000  # reálně zamčeno (196 balíčků)
    with open(lock, encoding="utf-8") as f:
        head = f.read(400)
    assert "pyannote-audio" in head or "name = " in head


def test_help_without_third_party(tmp_path):
    """--help funguje i s prázdným sys.path třetích stran (lazy importy)."""
    import subprocess
    import sys
    code = ("import sys; "
            "sys.path = [p for p in sys.path if 'site-packages' not in p]; "
            "sys.argv=['main.py', '--help']; "
            "import runpy; runpy.run_path('main.py', run_name='__main__')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=REPO, timeout=60)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "usage" in r.stdout.lower()


def _dep_names(req_strings):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    return {canonicalize_name(Requirement(r).name) for r in req_strings}


def test_pyproject_requirements_in_sync():
    """Pojistka proti driftu: každá pyproject závislost musí existovat
    v requirements.txt (+cuda/cpu trio)."""
    import tomllib
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    py_names = _dep_names(deps)
    have = set()
    for fname in ("requirements.txt", "requirements-cuda.txt", "requirements-cpu.txt"):
        with open(os.path.join(REPO, fname), encoding="utf-8") as f:
            reqs = [l.split("#", 1)[0].strip() for l in f.read().splitlines()]
        reqs = [r for r in reqs if r and not r.startswith(("-r", "--"))]
        have |= _dep_names(reqs)
    missing = py_names - have
    assert not missing, f"v requirements chybí: {missing}"


def _lock_versions():
    import tomllib
    from packaging.utils import canonicalize_name
    with open(os.path.join(REPO, "uv.lock"), "rb") as f:
        lock = tomllib.load(f)
    return {canonicalize_name(p["name"]): p.get("version") for p in lock["package"]}


def test_requirements_match_lock():
    """requirements*.txt se generují z uv.lock - stejné verze, žádný ruční drift."""
    from packaging.utils import canonicalize_name
    locked = _lock_versions()
    for fname in ("requirements.txt", "requirements-cuda.txt", "requirements-cpu.txt"):
        for r in _parsed(_req_lines(fname)):
            pins = [sp.version for sp in r.specifier if sp.operator == "=="]
            assert pins, f"{fname}: {r} není pinnutý"
            want = locked.get(canonicalize_name(r.name))
            assert want, f"{fname}: {r.name} není v uv.lock"
            got = pins[0].split("+")[0]  # torch==2.14.1+cu126 -> 2.14.1
            assert got == want.split("+")[0], f"{fname}: {r.name} {got} != lock {want}"
