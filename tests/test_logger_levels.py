# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Validační testy pro Fix E (L10): logger respektuje file/console level."""
import logging
import uuid

import pytest


@pytest.fixture
def isolated_logger(monkeypatch, tmp_path):
    """Izolace globálního stavu logger modulu + cwd -> tmp."""
    monkeypatch.chdir(tmp_path)
    from src.utils import logger as L
    saved_loggers = dict(L._initialized_loggers)
    saved_current = L._current_log_file
    L._initialized_loggers.clear()
    L._current_log_file = None
    yield L
    for lg in list(L._initialized_loggers.values()):
        for h in lg.handlers:
            try:
                h.close()
            except Exception:
                pass
    L._initialized_loggers.clear()
    L._initialized_loggers.update(saved_loggers)
    L._current_log_file = saved_current


def _fresh_name():
    return f"test-logger-{uuid.uuid4().hex[:8]}"


def test_defaults_unchanged(isolated_logger):
    """Bez parametrů: konzole INFO, soubor DEBUG (původní chování)."""
    L = isolated_logger
    lg = L.setup_logger(_fresh_name(), "INFO", "f")
    console = next(h for h in lg.handlers if type(h) is logging.StreamHandler)
    fh = next(h for h in lg.handlers if isinstance(h, logging.FileHandler))
    assert console.level == logging.INFO
    assert fh.level == logging.DEBUG


def test_file_level_respected(isolated_logger):
    """file_level=WARNING: debug/info do souboru nesmí, warning ano."""
    L = isolated_logger
    lg = L.setup_logger(_fresh_name(), "DEBUG", "g",
                        file_level="WARNING", console_level="CRITICAL")
    lg.debug("dbg-msg-xyz")
    lg.info("info-msg-xyz")
    lg.warning("warn-msg-xyz")
    for h in lg.handlers:
        h.flush()
    with open(L._current_log_file, encoding="utf-8") as f:
        content = f.read()
    assert "dbg-msg-xyz" not in content
    assert "info-msg-xyz" not in content
    assert "warn-msg-xyz" in content


def test_invalid_level_falls_back(isolated_logger):
    L = isolated_logger
    lg = L.setup_logger(_fresh_name(), "INFO", "h",
                        file_level="NESMYSL", console_level=None)
    fh = next(h for h in lg.handlers if isinstance(h, logging.FileHandler))
    console = next(h for h in lg.handlers if type(h) is logging.StreamHandler)
    assert fh.level == logging.DEBUG
    assert console.level == logging.INFO
