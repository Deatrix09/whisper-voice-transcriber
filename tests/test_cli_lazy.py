"""Validační testy pro Fix F (L09): --help a importy bez těžkých závislostí.

Zablokujeme torch/torchaudio/whisper/pyannote/nemo (sys.modules=None -> ImportError)
a ověříme, že lehké cesty stále fungují.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCKED = ",".join([
    "torch", "torch.serialization", "torch.torch_version",
    "torchaudio", "whisper", "pyannote.audio", "pyannote.audio.core.task",
    "nemo", "nemo.collections", "nemo.collections.asr",
])

PRELUDE = (
    "import sys; "
    f"blocked = '{BLOCKED}'.split(','); "
    "[sys.modules.__setitem__(m, None) for m in blocked]; "
)


def _run(code, **kw):
    return subprocess.run([sys.executable, "-c", PRELUDE + code],
                          capture_output=True, text=True, cwd=str(ROOT), **kw)


def test_help_without_heavy_deps():
    """python main.py --help nesmí potřebovat torch (F)."""
    r = subprocess.run(
        [sys.executable, "-c",
         PRELUDE + "import sys; sys.argv=['main.py', '--help']; "
         "import runpy; runpy.run_path('main.py', run_name='__main__')"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "usage" in r.stdout.lower()


def test_pipeline_import_without_torch():
    r = _run("import src.core.pipeline; print('pipeline-ok')")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "pipeline-ok" in r.stdout


def test_release_without_torch():
    r = _run("from types import SimpleNamespace; "
             "from src.core.pipeline import Pipeline; "
             "Pipeline.release_model_resources(SimpleNamespace()); print('release-ok')")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "release-ok" in r.stdout


def test_parse_arguments_unaffected():
    r = _run("from main import parse_arguments; print('parse-ok')")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "parse-ok" in r.stdout
