# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""P0 0.1 Secrets HOTFIX: žádný hardcoded token, env flow, fail-fast."""
import os
import re

import pytest

TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{10,}")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_hardcoded_tokens_in_config():
    """grep důkaz: žádný hf_* token v config/ (revoke stejně proveď na HF!)."""
    hits = []
    for dirpath, _, files in os.walk(os.path.join(REPO, "config")):
        for f in files:
            if f.endswith((".yaml", ".yml")):
                p = os.path.join(dirpath, f)
                with open(p, encoding="utf-8") as fh:
                    for i, line in enumerate(fh, 1):
                        if TOKEN_RE.search(line):
                            hits.append(f"{p}:{i}")
    assert not hits, f"hardcoded tokeny: {hits}"


def test_placeholder_present():
    with open(os.path.join(REPO, "config", "config.yaml"), encoding="utf-8") as f:
        assert "${HF_TOKEN}" in f.read()


def test_expand_env(monkeypatch):
    from src.utils.config import expand_env
    monkeypatch.setenv("HF_TOKEN", "sekret123")
    assert expand_env("${HF_TOKEN}") == "sekret123"
    monkeypatch.delenv("HF_TOKEN")
    assert expand_env("${HF_TOKEN}") == ""
    assert expand_env("plain-value") == "plain-value"
    assert expand_env("") == ""


def test_from_yaml_expands_token(tmp_path, monkeypatch):
    import yaml
    from src.utils.config import AppConfig
    with open(os.path.join(REPO, "config", "config.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "env-tok-1")
    assert AppConfig.from_yaml(str(p)).diarization.pyannote.auth_token == "env-tok-1"
    monkeypatch.delenv("HF_TOKEN")
    assert AppConfig.from_yaml(str(p)).diarization.pyannote.auth_token == ""


def _diarizer(app_config):
    from src.diarization.pyannote_diarizer import PyAnnoteDiarizer
    return PyAnnoteDiarizer(app_config)


def test_resolve_token_chain(app_config, monkeypatch):
    app_config.diarization.pyannote.auth_token = "cfg-tok"
    assert _diarizer(app_config)._resolve_token() == "cfg-tok"
    app_config.diarization.pyannote.auth_token = "${HF_TOKEN}"  # placeholder
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(ValueError, match="FAIL FAST|chybí HuggingFace token"):
        _diarizer(app_config)._resolve_token()
    monkeypatch.setenv("HUGGINGFACE_ACCESS_TOKEN", "second-tok")
    assert _diarizer(app_config)._resolve_token() == "second-tok"
    monkeypatch.setenv("HF_TOKEN", "first-tok")  # HF_TOKEN má přednost
    assert _diarizer(app_config)._resolve_token() == "first-tok"


def _pipe_uninit(app_config, name):
    from src.core.pipeline import Pipeline
    from src.utils.logger import setup_logger
    pipe = Pipeline.__new__(Pipeline)
    pipe.config = app_config
    pipe.logger = setup_logger(name, "WARNING", None)
    return pipe


def test_pipeline_fail_fast_no_token(app_config, monkeypatch):
    """Bez tokenu pád v konstrukci, ne po minutách transkripce."""
    for var in ("HF_TOKEN", "HUGGINGFACE_ACCESS_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    app_config.diarization.enabled = True
    app_config.diarization.module = "pyannote"
    app_config.diarization.pyannote.enabled = True
    app_config.diarization.pyannote.auth_token = ""
    pipe = _pipe_uninit(app_config, "test-p0-fail")
    with pytest.raises(ValueError, match="HF_TOKEN"):
        pipe._validate_diarization_config()


def test_pipeline_validate_ok_with_token(app_config, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    app_config.diarization.enabled = True
    app_config.diarization.module = "pyannote"
    app_config.diarization.pyannote.auth_token = ""
    _pipe_uninit(app_config, "test-p0-ok")._validate_diarization_config()  # bez raise
    app_config.diarization.enabled = False  # vypnutá diarizace nikdy nevadí
    _pipe_uninit(app_config, "test-p0-off")._validate_diarization_config()


def test_env_example_and_gitignore():
    example = os.path.join(REPO, ".env.example")
    assert os.path.exists(example)
    with open(example, encoding="utf-8") as f:
        assert "HF_TOKEN=" in f.read()
    with open(os.path.join(REPO, ".gitignore"), encoding="utf-8") as f:
        lines = [l.strip() for l in f]
    assert ".env" in lines
    assert ".env.example" not in lines  # example se commitovat MÁ
