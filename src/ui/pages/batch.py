# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dávka konfigurací. YAML se nevypisuje a needituje."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.ui.catalog import list_config_folders, list_yaml_names, yaml_paths
from src.ui.components import render_audio_selector, render_banner
from src.ui.environment import can_start, probe_environment
from src.ui.runner import run_batch
from src.ui.settings import public_error_message


def _render_summary(payload: dict) -> None:
    seconds = float(payload.get("seconds") or 0)
    cols = st.columns(3)
    cols[0].metric("Úspěchy", payload.get("successes") or 0)
    cols[1].metric("Chyby", payload.get("failures") or 0)
    cols[2].metric("Celková doba", f"{seconds:.1f} s ({seconds / 60:.1f} min)")
    outputs = payload.get("outputs") or []
    st.subheader("Výstupní soubory")
    if outputs:
        st.text("\n".join(outputs))
    else:
        st.caption("Žádný soubor se neuložil.")


def render() -> None:
    env = probe_environment()
    render_banner(env)
    st.title("Dávka konfigurací")
    st.caption("Jeden zvukový soubor se zpracuje každou konfigurací ve vybrané složce.")

    audio_path = render_audio_selector()
    folders = list_config_folders()
    if not folders:
        st.warning("V config/ není složka s YAML soubory.")
        return

    labels = {path: Path(path).name for path in folders}
    default_path = next((path for path in folders if Path(path).name == "fine_tuning"), folders[0])
    if "batch_folder" not in st.session_state:
        saved_folder = st.session_state.get("saved_batch_folder") or ""
        st.session_state.batch_folder = saved_folder if saved_folder in folders else default_path
    elif st.session_state.batch_folder not in folders:
        st.session_state.batch_folder = default_path
    folder = st.selectbox(
        "Složka v config/",
        folders,
        format_func=lambda path: labels.get(path, path),
        key="batch_folder",
    )
    st.session_state.saved_batch_folder = folder
    names = list_yaml_names(folder)
    st.markdown("**Nalezené konfigurace**")
    if names:
        st.text("\n".join(names))
    else:
        st.caption("Složka neobsahuje YAML.")

    if "batch_output_format" not in st.session_state:
        st.session_state.batch_output_format = st.session_state.get("saved_batch_format") or ""
    st.selectbox(
        "Výstupní formát",
        ["", "txt", "json", "subtitle"],
        format_func=lambda value: "z daného YAML" if value == "" else value,
        key="batch_output_format",
    )
    st.session_state.saved_batch_format = st.session_state.get("batch_output_format") or ""

    can_run = can_start(env, audio_path) and bool(names)
    if not audio_path:
        st.caption("Vyberte nebo nahrajte audio.")
    clicked = st.button("Spustit dávku", type="primary", disabled=not can_run)

    table = st.empty()
    if clicked:
        try:
            paths = yaml_paths(folder)
            chosen = st.session_state.get("batch_output_format") or None

            def on_update(rows: list) -> None:
                table.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            payload = run_batch(audio_path, paths, chosen, on_update=on_update)
            st.session_state.last_batch = payload
            st.session_state.batch_error = None
        except Exception as exc:
            st.session_state.batch_error = public_error_message(exc)
    elif st.session_state.get("last_batch"):
        table.dataframe(
            pd.DataFrame(st.session_state.last_batch.get("rows") or []),
            use_container_width=True,
            hide_index=True,
        )

    if st.session_state.get("batch_error"):
        st.error(st.session_state.batch_error)
    if st.session_state.get("last_batch"):
        _render_summary(st.session_state.last_batch)
