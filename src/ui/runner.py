# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Synchronní běh pipeline na volajícím vlákně. CUDA a Whisper se nespouští ve vlákně."""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from src.ui.settings import phase_from_log, redact, secret_values

COMPARISON_FILES = (
    "levenshtein_distances.csv",
    "similarity_scores.csv",
    "summary_report.txt",
    "text_comparisons.txt",
)


def register_torch_safe_globals() -> None:
    """Stejná registrace jako v main.py, dřív než se načte pyannote."""
    try:
        from torch.serialization import add_safe_globals
        from torch.torch_version import TorchVersion
        try:
            from pyannote.audio.core.task import Specifications, Problem, Resolution
            add_safe_globals([TorchVersion, Specifications, Problem, Resolution])
        except Exception:
            pass
    except Exception:
        pass


class _PhaseHandler(logging.Handler):
    def __init__(self, on_phase: Optional[Callable[[str], None]]):
        super().__init__(level=logging.INFO)
        self._on_phase = on_phase
        self._last = None
        # Diarizace může logovat z vlákna pipeline. Streamlit se smí volat jen z vlákna skriptu.
        self._thread = threading.get_ident()

    def emit(self, record: logging.LogRecord) -> None:
        if self._on_phase is None or threading.get_ident() != self._thread:
            return
        try:
            phase = phase_from_log(record.getMessage())
            if phase and phase != self._last:
                self._last = phase
                self._on_phase(phase)
        except Exception:
            return


class _LineHandler(logging.Handler):
    def __init__(self, on_line: Optional[Callable[[str], None]]):
        super().__init__(level=logging.INFO)
        self._on_line = on_line
        self._thread = threading.get_ident()

    def emit(self, record: logging.LogRecord) -> None:
        if self._on_line is None or threading.get_ident() != self._thread:
            return
        try:
            self._on_line(record.getMessage())
        except Exception:
            return


def read_log_text(path: Optional[str], secrets: list, limit: int = 150_000) -> str:
    if not path or not os.path.isfile(path):
        return ""
    try:
        data = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(data) > limit:
        data = "…\n" + data[-limit:]
    return redact(data, secrets)


def _attach(handler: logging.Handler):
    root = logging.getLogger()
    root.addHandler(handler)
    return root


def run_transcription(
    config,
    audio_path: str,
    output_format: str,
    on_phase: Optional[Callable[[str], None]] = None,
) -> dict:
    register_torch_safe_globals()
    from src.core.pipeline import Pipeline
    from src.utils.formatters import OutputFormatter
    from src.utils.logger import get_current_log_file
    from src.utils.path_utils import PathUtils

    if on_phase:
        on_phase("prepare")
    input_path, output_path = PathUtils.validate_and_prepare_paths(
        audio_path, output_format=output_format
    )
    handler = _PhaseHandler(on_phase)
    root = _attach(handler)
    secrets = secret_values(config)
    try:
        warnings = [redact(item, secrets) for item in config.validate()]
        pipeline = Pipeline(config)
        segments = pipeline.process_file(input_path)
        if on_phase:
            on_phase("save")
        OutputFormatter().save_output(segments, output_path, output_format)
        return {
            "segments": segments,
            "output_path": output_path,
            "output_format": output_format,
            "input_path": input_path,
            "log_text": read_log_text(get_current_log_file(), secrets),
            "warnings": warnings,
        }
    finally:
        root.removeHandler(handler)


def latest_comparison_dir(audio_path: str, not_before: float) -> str:
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    base = os.path.join(os.getcwd(), "data", "algorithm_output", audio_name)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Složka porovnání neexistuje: {base}")
    candidates = []
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if os.path.isdir(path) and os.path.getmtime(path) >= not_before - 2:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"Porovnání neuložilo výstup do {base}")
    return max(candidates, key=os.path.getmtime)


def run_comparison(config, audio_path: str, on_line: Optional[Callable[[str], None]] = None) -> dict:
    """LevenshteinComparator dostane vlastní kopii konfigurace a model si přepíše sám."""
    register_torch_safe_globals()
    from algorithm.levenshtein_comparator import LevenshteinComparator
    from src.utils.path_utils import PathUtils

    input_path, _ = PathUtils.validate_and_prepare_paths(audio_path, output_format="txt")
    started = time.time()
    comparator = LevenshteinComparator(config)
    handler = _LineHandler(on_line)
    root = _attach(handler)
    try:
        results = comparator.process_audio(input_path)
        transcriptions = results.pop("transcriptions", {})
        similarity = results["similarity_scores"].copy(deep=True)
        distances = results["levenshtein_distances"].copy(deep=True)
        comparator.save_results(results, input_path, transcriptions)
        folder = latest_comparison_dir(input_path, started)
        summary_path = os.path.join(folder, "summary_report.txt")
        summary = ""
        if os.path.isfile(summary_path):
            summary = Path(summary_path).read_text(encoding="utf-8", errors="replace")
        return {
            "folder": folder,
            "similarity": similarity,
            "distances": distances,
            "summary": summary,
            "transcriptions": transcriptions,
        }
    finally:
        root.removeHandler(handler)


def batch_output_paths(audio_path: str, config_path: str, output_format: str) -> tuple:
    """Cesta přes PathUtils. Název nese i konfiguraci, ať se běhy ve stejné sekundě nepřepíšou."""
    from src.utils.path_utils import PathUtils

    config_stem = Path(config_path).stem
    default_path = PathUtils.get_default_output_path(audio_path, output_format)
    root, ext = os.path.splitext(default_path)
    output_path = f"{root}_{config_stem}{ext}"
    return PathUtils.validate_and_prepare_paths(
        audio_path, output_path=output_path, output_format=output_format
    )


def run_batch(
    audio_path: str,
    config_paths: list,
    output_format: Optional[str],
    on_update: Optional[Callable[[list], None]] = None,
) -> dict:
    """Každá konfigurace má vlastní Pipeline. Chyba jedné nezastaví další."""
    from src.core.pipeline import Pipeline
    from src.utils.config import AppConfig
    from src.utils.formatters import OutputFormatter

    rows = []
    for config_path in config_paths:
        rows.append({
            "Konfigurace": os.path.basename(config_path),
            "Stav": "čeká",
            "Doba (s)": "",
            "Výstup": "",
            "Chyba": "",
        })
    if on_update:
        on_update(rows)

    started = time.perf_counter()
    successes = 0
    failures = 0
    outputs = []

    for index, config_path in enumerate(config_paths):
        rows[index]["Stav"] = "běží"
        if on_update:
            on_update(rows)
        item_started = time.perf_counter()
        config = None
        try:
            register_torch_safe_globals()
            config = AppConfig.from_yaml(config_path)
            chosen = output_format or config.output.format
            if output_format:
                config.output.format = output_format
            config.validate()
            input_path, output_path = batch_output_paths(audio_path, config_path, chosen)
            pipeline = Pipeline(config)
            try:
                segments = pipeline.process_file(input_path)
                OutputFormatter().save_output(segments, output_path, chosen)
            finally:
                Pipeline.release_model_resources(pipeline)
            rows[index]["Stav"] = "hotovo"
            rows[index]["Výstup"] = output_path
            successes += 1
            outputs.append(output_path)
        except Exception as exc:
            from src.ui.settings import public_error_message
            rows[index]["Stav"] = "chyba"
            rows[index]["Chyba"] = public_error_message(exc, secret_values(config))
            failures += 1
        finally:
            rows[index]["Doba (s)"] = f"{time.perf_counter() - item_started:.1f}"
            if on_update:
                on_update(rows)

    return {
        "rows": rows,
        "successes": successes,
        "failures": failures,
        "seconds": time.perf_counter() - started,
        "outputs": outputs,
    }
