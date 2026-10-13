# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Porovnání šesti Whisper modelů přes LevenshteinComparator."""

import os
from pathlib import Path

import streamlit as st

from src.ui.components import render_audio_selector, render_banner
from src.ui.environment import can_start, probe_environment
from src.ui.runner import COMPARISON_FILES, run_comparison
from src.ui.settings import build_config, module_label, public_error_message, secret_values
from src.ui.state import read_settings


def _settings_caption(settings: dict) -> str:
    language = settings["language"] or "automaticky"
    task = "překlad do angličtiny" if settings["task"] == "translate" else "přepis"
    diarization = "zapnuto" if settings["diarization_enabled"] else "vypnuto"
    module = module_label(settings["diarization_module"])
    speakers = (settings.get("num_speakers") or "").strip() or "automaticky"
    return (
        f"Z stránky Přepis: jazyk {language}, úkol {task}, zařízení {settings['device']}, "
        f"diarizace {diarization} ({module}), mluvčí {speakers}. "
        "Model si porovnání přepíše samo."
    )


def _render_result(payload: dict) -> None:
    folder = payload.get("folder") or ""
    st.caption(f"Složka výsledků: {folder}")
    columns = st.columns(4)
    for column, name in zip(columns, COMPARISON_FILES):
        path = os.path.join(folder, name)
        with column:
            if os.path.isfile(path):
                data = Path(path).read_bytes()
                st.download_button(
                    f"Stáhnout {name}",
                    data=data,
                    file_name=name,
                    key=f"cmp_dl_{name}",
                )
            else:
                st.caption(f"Chybí {name}")

    similarity = payload.get("similarity")
    st.subheader("Podobnost (%)")
    if similarity is not None:
        try:
            styled = similarity.style.background_gradient(cmap="Blues", axis=None).format("{:.1f}")
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(similarity, use_container_width=True)
    distances = payload.get("distances")
    st.subheader("Levenshteinova vzdálenost")
    if distances is not None:
        st.dataframe(distances, use_container_width=True)

    st.subheader("Souhrn")
    st.text(payload.get("summary") or "(prázdný souhrn)")

    transcriptions = payload.get("transcriptions") or {}
    if transcriptions:
        st.subheader("Texty modelů")
        tabs = st.tabs(list(transcriptions.keys()))
        for tab, (model, text) in zip(tabs, transcriptions.items()):
            with tab:
                st.text_area(
                    model,
                    text,
                    height=240,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"cmp_text_{model}",
                )


def render() -> None:
    env = probe_environment()
    render_banner(env)
    st.title("Porovnání modelů")
    st.caption("Stejné audio postupně přepíšou vybrané modely. Výsledek je tabulka podobnosti.")

    audio_path = render_audio_selector()
    settings = read_settings()
    st.caption(_settings_caption(settings))
    st.warning(
        "Audio se zpracuje šesti modely za sebou "
        "(tiny, base, small, medium, large-v2, large-v3). "
        "Běh trvá násobně déle než jeden přepis."
    )

    can_run = can_start(env, audio_path)
    if not audio_path:
        st.caption("Vyberte nebo nahrajte audio.")
    clicked = st.button("Spustit porovnání", type="primary", disabled=not can_run)

    if clicked:
        config = None
        try:
            config = build_config(settings, include_model=False)
            config.validate()
            with st.status("Porovnání modelů běží", expanded=True) as status:
                st.caption("Průběh uvnitř modelu se neodečítá v procentech.")

                def on_line(message: str) -> None:
                    marker = "Processing with model:"
                    if marker in message:
                        model = message.split(marker, 1)[1].strip().split()[0]
                        status.update(label=f"Model {model}")

                payload = run_comparison(config, audio_path, on_line=on_line)
                status.update(label="Hotovo", state="complete")
            st.session_state.last_comparison = payload
            st.session_state.comparison_error = None
        except Exception as exc:
            secrets = secret_values(config) if config is not None else []
            st.session_state.comparison_error = public_error_message(exc, secrets)

    if st.session_state.get("comparison_error"):
        st.error(st.session_state.comparison_error)
    if st.session_state.get("last_comparison"):
        _render_result(st.session_state.last_comparison)
