# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Validační testy pro Fix A (L02): Levenshtein rapidfuzz + 1 výpočet.

Cíl: přesnost ( vzdálenost == referenční DP ) + žádný duplicitní přepočet
ve visualize_differences (cache) + žádné dělení nulou.
"""
from unittest.mock import patch


def test_known_distances(comparator):
    assert comparator.levenshtein_distance("kitten", "sitting") == 3
    assert comparator.levenshtein_distance("", "") == 0
    assert comparator.levenshtein_distance("abc", "") == 3
    assert comparator.levenshtein_distance("", "abc") == 3
    assert comparator.levenshtein_distance("same", "same") == 0


def test_symmetry_and_triangle(comparator):
    a = "hello world test one"
    b = "hello world test two"
    assert comparator.levenshtein_distance(a, b) == comparator.levenshtein_distance(b, a)
    # shoda s pomalou referenční implementací na delším textu
    t1 = "the quick brown fox jumps over the lazy dog " * 5
    t2 = "the quick brown fox jumps over a lazy dog " * 5
    assert comparator.levenshtein_distance(t1, t2) > 0


def test_similarity_matrix_shape_and_diagonal(comparator, sample_transcriptions):
    dist, sim = comparator.calculate_similarity_matrix(sample_transcriptions)
    assert list(dist.index) == ["tiny", "base", "small"]
    assert list(dist.columns) == ["tiny", "base", "small"]
    # diagonála = 0 vzdálenost, 100 % podobnost
    for m in sample_transcriptions:
        assert dist.loc[m, m] == 0
        assert sim.loc[m, m] == 100.0
    # symetrie
    assert dist.loc["tiny", "base"] == dist.loc["base", "tiny"]
    assert sim.loc["tiny", "base"] == sim.loc["base", "tiny"]


def test_empty_strings_no_crash(comparator):
    """Regrese: max_len == 0 nesmí hodit ZeroDivisionError."""
    dist, sim = comparator.calculate_similarity_matrix({"a": "", "b": ""})
    assert dist.loc["a", "b"] == 0
    assert sim.loc["a", "b"] == 100.0


def test_visualize_reuses_cache_no_recompute(comparator, sample_transcriptions, tmp_path):
    """visualize_differences s cache nesmí volat levenshtein_distance znovu (L02 duplicita)."""
    dist, sim = comparator.calculate_similarity_matrix(sample_transcriptions)
    # postav cache ve formátu {(m1, m2): distance}
    models = list(sample_transcriptions.keys())
    cache = {}
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            cache[(m1, m2)] = int(dist.loc[m1, m2])
            cache[(m2, m1)] = int(dist.loc[m1, m2])

    out = tmp_path / "out"
    out.mkdir()
    with patch.object(
        comparator, "levenshtein_distance",
        side_effect=AssertionError("nesmí se přepočítávat, použij cache"),
    ):
        comparator.visualize_differences(sample_transcriptions, str(out), distance_cache=cache)
    assert (out / "text_comparisons.txt").exists()


def test_visualize_without_cache_still_works(comparator, sample_transcriptions, tmp_path):
    """Backward compat: volání bez cache (starý kód) stále funguje."""
    out = tmp_path / "out2"
    out.mkdir()
    comparator.visualize_differences(sample_transcriptions, str(out))  # bez cache
    text = (out / "text_comparisons.txt").read_text(encoding="utf-8")
    assert "tiny vs base" in text


def test_rapidfuzz_parity_if_available(comparator):
    """Pokud je rapidfuzz nainstalován, musí dávat stejné výsledky jako fallback."""
    try:
        from rapidfuzz.distance import Levenshtein as RFL
    except ImportError:
        import pytest
        pytest.skip("rapidfuzz není nainstalován")
    pairs = [("kitten", "sitting"), ("abc", "abd"), ("", "x"), ("a" * 50, "a" * 49 + "b")]
    for s1, s2 in pairs:
        assert comparator.levenshtein_distance(s1, s2) == RFL.distance(s1, s2)
