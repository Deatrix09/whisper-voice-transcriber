# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Společné widgety. Samy o sobě nespouští pipeline."""

import os
from pathlib import Path

import streamlit as st

from src.ui.audio_library import list_input_audio, save_upload
from src.ui.settings import (
    OUTPUT_FORMATS,
    TASKS,
    WHISPER_MODELS,
    load_language_choices,
    module_label,
    module_options,
)
from src.ui.state import remember_form_widgets, restore_form_widgets


def render_banner(env: dict) -> None:
    cuda = "ano" if env.get("cuda") else "ne"
    gpu = f" ({env['gpu_name']})" if env.get("gpu_name") else ""
    ffmpeg = "na PATH" if env.get("ffmpeg") else "chybí"
    st.markdown(f"**Zařízení:** CUDA {cuda}{gpu} · **ffmpeg:** {ffmpeg}")
    if not env.get("ffmpeg"):
        st.warning("ffmpeg musí být nainstalovaný a na PATH.")
    if not env.get("torch"):
        st.warning("Balík torch není nainstalovaný. Přepis nejde spustit.")


def render_audio_selector() -> str:
    if "audio_path" not in st.session_state:
        st.session_state.audio_path = st.session_state.get("saved_audio_path") or ""
    uploaded = st.file_uploader(
        "Nahrát audio",
        type=["mp3", "wav", "m4a", "flac", "ogg", "aac", "mp4"],
        key="audio_uploader",
    )
    if uploaded is not None:
        token = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
        if st.session_state.get("upload_token") != token:
            try:
                saved = save_upload(uploaded.name, uploaded.getvalue())
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state.upload_token = token
                st.session_state.audio_path = saved

    files = list_input_audio()
    options = [""] + files
    current = st.session_state.get("audio_path") or ""
    if current not in options:
        st.session_state.audio_path = ""
    st.selectbox(
        "Soubor v data/input",
        options,
        format_func=lambda path: "(nevybráno)" if not path else Path(path).name,
        key="audio_path",
    )
    selected = st.session_state.get("audio_path") or ""
    st.session_state.saved_audio_path = selected
    if selected and os.path.isfile(selected):
        st.audio(selected)
    return selected


def render_settings_form(env: dict) -> None:
    restore_form_widgets(env)
    languages = load_language_choices()
    labels = {code: label for label, code in languages}
    codes = [code for _label, code in languages]
    current_language = st.session_state.get("ui_language") or ""
    if current_language not in labels:
        codes.append(current_language)
        labels[current_language] = current_language or "Automaticky"

    task_labels = {code: label for label, code in TASKS}
    task_codes = [code for _label, code in TASKS]
    current_task = st.session_state.get("ui_task") or "translate"
    if current_task not in task_codes:
        task_codes.append(current_task)
        task_labels[current_task] = current_task

    modules = module_options(st.session_state.get("ui_diarization_module") or "pyannote")

    left, right = st.columns(2)
    with left:
        st.selectbox("Model", list(WHISPER_MODELS), key="ui_model")
        st.selectbox(
            "Jazyk",
            codes,
            format_func=lambda code: labels.get(code, code),
            key="ui_language",
        )
        st.selectbox(
            "Úkol",
            task_codes,
            format_func=lambda code: task_labels.get(code, code),
            key="ui_task",
        )
        st.selectbox("Výstupní formát", list(OUTPUT_FORMATS), key="ui_output_format")
        if env.get("cuda"):
            st.selectbox("Zařízení", ["cuda", "cpu"], key="ui_device")
        else:
            st.selectbox("Zařízení", ["cpu"], key="ui_device")
            st.caption("CUDA není k dispozici. Volba cuda je vypnutá, vybrané zařízení je CPU.")
    with right:
        st.checkbox("Diarizace", key="ui_diarization_enabled")
        st.selectbox(
            "Modul diarizace",
            modules,
            format_func=module_label,
            key="ui_diarization_module",
        )
        st.text_input(
            "Počet mluvčích",
            key="ui_num_speakers",
            help="Prázdné pole znamená automatickou detekci.",
        )

    with st.expander("Pokročilé"):
        adv_left, adv_right = st.columns(2)
        with adv_left:
            st.number_input(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                key="ui_temperature",
            )
            st.number_input(
                "chunk_size (s)",
                min_value=11,
                step=1,
                key="ui_chunk_size",
            )
            st.number_input(
                "chunk_threshold_mb",
                min_value=1,
                step=1,
                key="ui_chunk_threshold_mb",
            )
        with adv_right:
            st.number_input(
                "chunk_overlap_sec",
                min_value=0,
                step=1,
                key="ui_chunk_overlap_sec",
            )
            st.number_input("num_workers", min_value=0, step=1, key="ui_num_workers")
            st.checkbox("fp16", key="ui_fp16")
    remember_form_widgets()
