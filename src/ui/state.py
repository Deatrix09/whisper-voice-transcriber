# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Sdílená nastavení Přepisu a Porovnání. V session jsou jen řetězce a čísla."""

from src.ui.environment import probe_environment
from src.ui.settings import defaults_from_config

_FIELDS = (
    ("ui_model", "model"),
    ("ui_language", "language"),
    ("ui_task", "task"),
    ("ui_output_format", "output_format"),
    ("ui_device", "device"),
    ("ui_diarization_enabled", "diarization_enabled"),
    ("ui_diarization_module", "diarization_module"),
    ("ui_num_speakers", "num_speakers"),
    ("ui_temperature", "temperature"),
    ("ui_chunk_size", "chunk_size"),
    ("ui_chunk_threshold_mb", "chunk_threshold_mb"),
    ("ui_chunk_overlap_sec", "chunk_overlap_sec"),
    ("ui_num_workers", "num_workers"),
    ("ui_fp16", "fp16"),
)


def ensure_settings_state() -> dict:
    import streamlit as st
    from src.utils.config import AppConfig

    env = probe_environment()
    if "form_settings" not in st.session_state:
        config = AppConfig.from_yaml("config/config.yaml")
        st.session_state.form_settings = defaults_from_config(
            config, cuda_available=env["cuda"]
        )
    saved = st.session_state.form_settings
    if not env["cuda"] and saved.get("device") == "cuda":
        saved["device"] = "cpu"
    # Klíče widgetů Streamlit smaže, když formulář na stránce není.
    # Trvalé hodnoty jsou ve form_settings / saved_audio_path.
    st.session_state.setdefault("saved_audio_path", "")
    st.session_state.setdefault("last_transcription", None)
    st.session_state.setdefault("transcription_error", None)
    st.session_state.setdefault("last_comparison", None)
    st.session_state.setdefault("comparison_error", None)
    st.session_state.setdefault("last_batch", None)
    st.session_state.setdefault("batch_error", None)
    return env


def restore_form_widgets(env: dict) -> None:
    """Doplň klíče widgetů těsně před jejich vykreslením."""
    import streamlit as st

    saved = st.session_state.form_settings
    if not env.get("cuda") and saved.get("device") == "cuda":
        saved["device"] = "cpu"
    for state_key, field in _FIELDS:
        if state_key not in st.session_state:
            st.session_state[state_key] = saved[field]
    if not env.get("cuda") and st.session_state.get("ui_device") == "cuda":
        st.session_state.ui_device = "cpu"
        saved["device"] = "cpu"


def remember_form_widgets() -> None:
    """Ulož hodnoty widgetů, ať přežijí přechod na jinou stránku."""
    import streamlit as st

    saved = st.session_state.form_settings
    for state_key, field in _FIELDS:
        if state_key in st.session_state:
            saved[field] = st.session_state[state_key]


def read_settings() -> dict:
    import streamlit as st

    remember_form_widgets()
    return dict(st.session_state.form_settings)
