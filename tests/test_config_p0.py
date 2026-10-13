# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""P0 0.3 Config validace: kontextové chyby, defaulty, device auto, --validate."""
import os

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _base_cfg():
    with open(os.path.join(REPO, "config", "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write(tmp_path, data, name="c.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(p)


def test_missing_section_reports_name(tmp_path):
    from src.utils.config import AppConfig, ConfigError
    data = _base_cfg()
    del data["audio"]
    with pytest.raises(ConfigError, match="audio"):
        AppConfig.from_yaml(_write(tmp_path, data))


def test_syntax_error_reports_line(tmp_path):
    from src.utils.config import AppConfig, ConfigError
    p = tmp_path / "bad.yaml"
    p.write_text("model:\n  name: [neuzavreno\n", encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        AppConfig.from_yaml(str(p))
    assert "ádek" in str(e.value) or "radek" in str(e.value).lower() or ":" in str(e.value)


def test_bad_keys_report_section(tmp_path):
    from src.utils.config import AppConfig, ConfigError
    data = _base_cfg()
    data["model"] = {"neexistujici_klic": 1}
    with pytest.raises(ConfigError, match="klíče"):
        AppConfig.from_yaml(_write(tmp_path, data))


def test_unknown_model_is_warning_not_fatal(tmp_path):
    from src.utils.config import AppConfig
    data = _base_cfg()
    data["model"]["name"] = "future-model-xl"
    cfg = AppConfig.from_yaml(_write(tmp_path, data))
    warnings = cfg.validate()
    assert any("future-model-xl" in w for w in warnings)


def test_device_validation(tmp_path):
    from src.utils.config import AppConfig, ConfigError, resolve_device
    data = _base_cfg()
    data["performance"]["device"] = "quantum"
    with pytest.raises(ConfigError, match="device"):
        AppConfig.from_yaml(_write(tmp_path, data)).validate()
    data["performance"]["device"] = "auto"
    assert AppConfig.from_yaml(_write(tmp_path, data)).validate() is not None
    assert resolve_device("auto") == "cpu"  # tu není CUDA
    assert resolve_device("cuda") == "cpu"  # fallback bez pádu
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        resolve_device("quantum")


def test_chunk_rules(tmp_path):
    from src.utils.config import AppConfig, ConfigError
    data = _base_cfg()
    data["audio"]["chunk_size"] = 10
    with pytest.raises(ConfigError, match="chunk_size"):
        AppConfig.from_yaml(_write(tmp_path, data)).validate()
    data["audio"]["chunk_size"] = 30
    data["audio"]["chunk_overlap_sec"] = 30
    with pytest.raises(ConfigError, match="chunk_overlap"):
        AppConfig.from_yaml(_write(tmp_path, data)).validate()


def test_module_format_engine_rules(tmp_path):
    from src.utils.config import AppConfig, ConfigError
    data = _base_cfg()
    data["diarization"]["module"] = "whisperx"
    with pytest.raises(ConfigError, match="module"):
        AppConfig.from_yaml(_write(tmp_path, data)).validate()
    data["diarization"]["module"] = "pyannote"
    data["diarization"]["num_speakers"] = 0
    with pytest.raises(ConfigError, match="num_speakers"):
        AppConfig.from_yaml(_write(tmp_path, data)).validate()
    data["diarization"]["num_speakers"] = None
    data["output"]["format"] = "docx"
    with pytest.raises(ConfigError, match="format"):
        AppConfig.from_yaml(_write(tmp_path, data)).validate()
    data["output"]["format"] = "txt"
    data["asr"]["engine"] = "conformer"
    with pytest.raises(ConfigError, match="engine"):
        AppConfig.from_yaml(_write(tmp_path, data)).validate()


def test_old_yaml_defaults(tmp_path):
    """Staré YAML: chybějící chunk_overlap_sec -> 0, num_speakers -> None."""
    from src.utils.config import AppConfig
    data = _base_cfg()
    del data["audio"]["chunk_overlap_sec"]
    del data["diarization"]["num_speakers"]
    del data["asr"]
    cfg = AppConfig.from_yaml(_write(tmp_path, data))
    assert cfg.audio.chunk_overlap_sec == 0
    assert cfg.diarization.num_speakers is None
    assert cfg.asr.engine == "faster-whisper"  # defaulty
    cfg.validate()  # bez raise (device cuda -> jen warning)


def test_repo_config_validates():
    from src.utils.config import AppConfig
    warnings = AppConfig.from_yaml(os.path.join(REPO, "config", "config.yaml")).validate()
    assert isinstance(warnings, list)  # cuda warning na CPU boxu je OK


def test_validate_flag_subprocess(tmp_path):
    import subprocess
    import sys
    data = _base_cfg()
    good = _write(tmp_path, data, "good.yaml")
    r = subprocess.run([sys.executable, "main.py", "--validate", "--config", good],
                       capture_output=True, text=True, cwd=REPO, timeout=120)
    assert r.returncode == 0, r.stderr[-1500:] + r.stdout[-1500:]
    assert "OK" in r.stdout
    data["output"]["format"] = "docx"
    bad = _write(tmp_path, data, "bad.yaml")
    r2 = subprocess.run([sys.executable, "main.py", "--validate", "--config", bad],
                        capture_output=True, text=True, cwd=REPO, timeout=120)
    assert r2.returncode == 2
    assert "NEPLATNÝ" in r2.stdout


def test_input_still_required():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "main.py", "--config", "config/config.yaml"],
                       capture_output=True, text=True, cwd=REPO, timeout=60)
    assert r.returncode == 2
    assert "--input" in (r.stdout + r.stderr)
