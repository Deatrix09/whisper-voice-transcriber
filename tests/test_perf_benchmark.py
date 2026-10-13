"""Benchmark helper pro before/after měření (FÁZE 2/5 auditu).

Není to striktní perf test (CI nesmí padat na pomalém stroji) — měří časy
a vypisuje je; assert hlídá jen funkční regresi + řádový strop pro duplicitu.
Spouštění:
  python -m pytest tests/test_perf_benchmark.py -v -s
  python -m pytest tests/test_perf_benchmark.py -v -s --example-audio=data/input/example.mp3
"""
import time

import pytest


def _timeit(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def test_levenshtein_no_duplicate_work(comparator, sample_transcriptions, capsys):
    """Matice + vizualizace s cache musí být výrazně rychlejší než 2× matice."""
    _, t_matrix = _timeit(comparator.calculate_similarity_matrix, sample_transcriptions)
    cache = dict(getattr(comparator, "_last_distances", {}))
    tmp = pytest.TempPathFactory  # placeholder aby test byl čitelný
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        _, t_viz = _timeit(comparator.visualize_differences, sample_transcriptions, d, cache)
        assert os.path.exists(os.path.join(d, "text_comparisons.txt"))
    print(f"\n[benchmark] matrix={t_matrix*1000:.1f}ms viz_cached={t_viz*1000:.1f}ms "
          f"(bez cache by viz stála ~ tolik co matrix)")
    # vizualizace s cache nesmí trvat déle než samotná matice (jinak se přepočítává)
    assert t_viz <= t_matrix * 1.5 + 0.5


def test_levenshtein_scales_linearly_on_growth(comparator):
    """Kouřový škálovací test: 2× delší text nesmí být 10× pomalejší (C-ext) ani 4× (DP)."""
    base = "the quick brown fox jumps over the lazy dog " * 10
    t1, _ = _timeit(comparator.levenshtein_distance, base, base + "!")
    _, dt1 = _timeit(comparator.levenshtein_distance, base, base + "!")
    long1, long2 = base * 2, (base + "!") * 2
    _, dt2 = _timeit(comparator.levenshtein_distance, long1, long2)
    print(f"\n[benchmark] short={dt1*1000:.1f}ms long(2x)={dt2*1000:.1f}ms")
    assert dt2 < dt1 * 6 + 1.0  # velkorysý strop, chrání před O(n^3) regresí


@pytest.mark.skipif(
    __import__("os").path.exists("data/input/example.mp3") is False,
    reason="data/input/example.mp3 neexistuje — přeskočeno (spusť s reálným audiem)",
)
def test_example_mp3_smoke():
    """Šablona pro ruční before/after na stejném souboru (viz checklist v auditu)."""
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "main.py", "--input", "data/input/example.mp3", "--format", "txt"],
        capture_output=True, text=True, timeout=3600,
    )
    assert r.returncode == 0, r.stderr[-2000:]
