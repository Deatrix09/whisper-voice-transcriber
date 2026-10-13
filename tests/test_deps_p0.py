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
    names = {r.name.lower() for r in parsed}
    assert "torch" not in names and "torchaudio" not in names  # jen v cuda/cpu
    assert "python-dotenv" in names  # P0 0.1 .env flow
    assert "faster-whisper" in names and "pyannote.audio" in names  # packaging .name nedělá PEP503 (tečky zůstávají)


def test_cuda_cpu_split():
    for f, suffix in (("requirements-cuda.txt", "cu118"), ("requirements-cpu.txt", "cpu")):
        reqs = _req_lines(f)
        assert any(r.startswith("-r requirements.txt") for r in reqs)
        assert any("extra-index-url" in r and ("cu118" in r or "/cpu" in r) for r in reqs)
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
