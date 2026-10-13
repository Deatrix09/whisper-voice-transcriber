# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Read-only historie výstupů. Nic se nemaže."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.ui.catalog import (
    list_comparison_runs,
    list_output_files,
    list_run_files,
    readable_data_file,
)
from src.ui.components import render_banner
from src.ui.environment import probe_environment
from src.ui.settings import env_secrets, redact

_PREVIEW_LIMIT = 100_000
_TEXT_SUFFIXES = {".txt", ".subtitle", ".text", ".log", ".md"}


def _show_file(path: str, key_prefix: str) -> None:
    try:
        file_path = readable_data_file(path)
    except Exception as exc:
        st.error(str(exc))
        return
    data = file_path.read_bytes()
    st.download_button(
        f"Stáhnout {file_path.name}",
        data=data,
        file_name=file_path.name,
        key=f"{key_prefix}_dl",
    )
    suffix = file_path.suffix.lower()
    secrets = env_secrets()
    if suffix == ".csv":
        try:
            frame = pd.read_csv(file_path, index_col=0)
            st.dataframe(frame, use_container_width=True)
            return
        except Exception:
            pass
    if suffix in _TEXT_SUFFIXES or suffix in {".json", ".csv"} or suffix == "":
        text = data.decode("utf-8", errors="replace")
        clipped = len(text) > _PREVIEW_LIMIT
        shown = redact(text[:_PREVIEW_LIMIT], secrets)
        if clipped:
            st.caption("Náhled je zkrácený.")
        st.text_area(
            file_path.name,
            shown,
            height=320,
            disabled=True,
            label_visibility="collapsed",
            key=f"{key_prefix}_view",
        )
        return
    st.caption("Pro tento typ souboru je k dispozici jen stažení.")


def render() -> None:
    env = probe_environment()
    render_banner(env)
    st.title("Historie")
    st.caption("Uložené přepisy a běhy porovnání. Soubory se odsud nemažou.")

    st.subheader("data/output")
    outputs = list_output_files()
    if not outputs:
        st.caption("Zatím tu nejsou žádné soubory.")
    else:
        selected = st.selectbox(
            "Výstup přepisu",
            outputs,
            format_func=lambda path: Path(path).name,
            key="history_output",
        )
        _show_file(selected, "history_output")

    st.subheader("data/algorithm_output")
    runs = list_comparison_runs()
    if not runs:
        st.caption("Zatím tu nejsou žádné běhy porovnání.")
        return
    labels = {item["path"]: f"{item['audio']} / {item['stamp']}" for item in runs}
    paths = [item["path"] for item in runs]
    run_path = st.selectbox(
        "Běh porovnání",
        paths,
        format_func=lambda path: labels.get(path, path),
        key="history_run",
    )
    files = list_run_files(run_path)
    if not files:
        st.caption("Složka běhu je prázdná.")
        return
    chosen = st.selectbox(
        "Soubor běhu",
        files,
        format_func=lambda path: Path(path).name,
        key="history_run_file",
    )
    _show_file(chosen, "history_run_file")
