"""Validační testy pro FÁZI 6: WER/CER + cpWER + RTFx + --models.

jiwer je reálně nainstalován (4.0.0); pipeline se mockuje (žádné váhy).
"""
from unittest.mock import patch

import pytest


def test_wer_cer_basics(comparator):
    import algorithm.levenshtein_comparator as M
    assert M._HAS_JIWER, "jiwer musí být nainstalován"
    df = comparator.calculate_wer_cer({"a": "hello world", "b": "hello world"})
    assert (df["wer"] == 0.0).all() and (df["cer"] == 0.0).all()
    df2 = comparator.calculate_wer_cer({"ref": "hello world", "hyp": "hello there world"},
                                       reference="ref")
    assert df2.set_index("model").loc["hyp", "wer"] == pytest.approx(0.5)
    assert df2["reference"].unique().tolist() == ["ref"]


def test_reference_pick(comparator):
    assert comparator._pick_reference({"tiny": "a", "large-v3": "b"}) == "large-v3"
    assert comparator._pick_reference({"tiny": "a", "base": "b"}) == "base"
    assert comparator._pick_reference({"tiny": "a", "turbo": "b"}) == "turbo"
    assert comparator._pick_reference({"tiny": "a"}) == "tiny"


def test_cpwer_permutation_invariance(comparator):
    """cpWER: prohození speaker labelů = 0 (klíčová vlastnost)."""
    ref = [{"start": 0.0, "end": 1.0, "speaker": "A", "text": "hello world"},
           {"start": 1.0, "end": 2.0, "speaker": "B", "text": "foo bar"}]
    swapped = [{"start": 0.0, "end": 1.0, "speaker": "B", "text": "hello world"},
               {"start": 1.0, "end": 2.0, "speaker": "A", "text": "foo bar"}]
    out = comparator.calculate_cpwer({"ref": ref, "hyp": swapped}, reference="ref")
    assert out == {"ref": 0.0, "hyp": 0.0}
    other = [{"start": 0.0, "end": 1.0, "speaker": "A", "text": "completely different words here"}]
    out2 = comparator.calculate_cpwer({"ref": ref, "hyp": other}, reference="ref")
    assert out2["hyp"] > 0.0


def test_set_models_validation(comparator):
    assert comparator.set_models(["tiny", "turbo", "parakeet"]) == ["tiny", "turbo", "parakeet"]
    with pytest.raises(ValueError, match="Neznámé modely"):
        comparator.set_models(["tiny", "xxl"])
    with pytest.raises(ValueError):
        comparator.set_models([])
    assert comparator._resolve_entry("parakeet") == {"engine": "parakeet", "model_name": "parakeet"}
    assert comparator._resolve_entry("tiny") == {"engine": "faster-whisper", "model_name": "tiny"}


def _fake_pipeline_factory(config):
    from unittest.mock import MagicMock
    m = MagicMock()
    name = config.model.name
    m.process_file.return_value = [
        {"start": 0.0, "end": 1.0, "text": f"shared text {name}",
         "speaker": "SPEAKER_00"}]
    return m


def test_process_audio_timing_restore_release(comparator, tmp_path, monkeypatch):
    """Mock pipeline: timing+RTFx, WER vs reference, config restore, release 2x."""
    from conftest import make_wav
    import src.core.pipeline as pipe_mod
    wav = make_wav(tmp_path / "e.wav", seconds=0.5)
    comparator.set_models(["tiny", "base"])
    orig_name, orig_engine, orig_asr = (comparator.config.model.name,
                                        comparator.config.asr.engine,
                                        comparator.config.asr.model)
    calls = []
    with patch("src.core.pipeline.Pipeline", side_effect=_fake_pipeline_factory) as Mock, \
         patch.object(pipe_mod.Pipeline, "release_model_resources",
                      side_effect=lambda p: calls.append(p)), \
         patch("src.audio.processor.AudioProcessor.cleanup", return_value=None):
        out = comparator.process_audio(str(wav))
    assert set(out["transcriptions"]) == {"tiny", "base"}
    assert list(out["timing"]["model"]) == ["tiny", "base"]
    assert (out["timing"]["rtfx"] >= 0).all()
    assert "wer_cer" in out and "cpwer" in out["wer_cer"].columns
    assert out["wer_cer"]["reference"].unique().tolist() == ["base"]
    assert len(calls) == 2  # release po každém modelu
    # config obnoven (FÁZE 4 asr.* + legacy model.name)
    assert comparator.config.model.name == orig_name
    assert comparator.config.asr.engine == orig_engine
    assert comparator.config.asr.model == orig_asr


def test_process_audio_without_jiwer(comparator, tmp_path, monkeypatch):
    import algorithm.levenshtein_comparator as M
    from conftest import make_wav
    wav = make_wav(tmp_path / "j.wav", seconds=0.5)
    comparator.set_models(["tiny"])
    monkeypatch.setattr(M, "_HAS_JIWER", False)
    with patch("src.core.pipeline.Pipeline", side_effect=_fake_pipeline_factory), \
         patch("src.audio.processor.AudioProcessor.cleanup", return_value=None):
        out = comparator.process_audio(str(wav))
    assert "wer_cer" not in out  # přeskočeno s warningem
    assert "timing" in out and len(out["timing"]) == 1


def test_save_results_report_sections(comparator, tmp_path, monkeypatch):
    """timing.csv + wer_cer.csv + RTFx/WER sekce v summary_report.txt."""
    monkeypatch.chdir(tmp_path)
    trans = {"tiny": "hello world", "base": "hello world"}
    dist, sim = comparator.calculate_similarity_matrix(trans)
    wer_cer = comparator.calculate_wer_cer(trans)
    import pandas as pd
    timing = pd.DataFrame([{"model": "tiny", "seconds": 8.1, "audio_seconds": 100.0, "rtfx": 12.35},
                           {"model": "base", "seconds": 20.0, "audio_seconds": 100.0, "rtfx": 5.0}])
    results = {"levenshtein_distances": dist, "similarity_scores": sim,
               "timing": timing, "wer_cer": wer_cer}
    comparator.save_results(results, "audio_x.mp3", dict(trans))
    outdir = next((tmp_path / "data" / "algorithm_output" / "audio_x").iterdir())
    assert (outdir / "timing.csv").exists()
    assert (outdir / "wer_cer.csv").exists()
    assert (outdir / "levenshtein_distances.csv").exists()  # původní výstup zachován
    summary = (outdir / "summary_report.txt").read_text(encoding="utf-8")
    assert "RTFx" in summary and "tiny: 12.35x" in summary
    assert "WER" in summary and "CER" in summary


def test_run_comparison_models_arg(monkeypatch):
    import sys
    from run_comparison import parse_arguments
    monkeypatch.setattr(sys, "argv", ["run_comparison.py", "--input", "x.mp3",
                                      "--models", "tiny,turbo,parakeet"])
    args = parse_arguments()
    assert args.models == "tiny,turbo,parakeet"
