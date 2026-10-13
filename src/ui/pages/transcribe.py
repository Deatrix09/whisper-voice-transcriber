# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Stránka přepisu jednoho souboru."""

from pathlib import Path

import streamlit as st

from src.ui.components import render_audio_selector, render_banner, render_settings_form
from src.ui.environment import can_start, probe_environment
from src.ui.results import encode_segments, segment_metrics, table_rows, text_by_speaker
from src.ui.runner import run_transcription
from src.ui.settings import (
    OUTPUT_FORMATS,
    PHASE_LABELS,
    PHASE_ORDER,
    build_config,
    public_error_message,
    secret_values,
)
from src.ui.state import read_settings
from src.utils.formatters import OutputFormatter


def _paint_phases(box, reached: list, current: str, diarization_enabled: bool) -> None:
    lines = []
    for key in PHASE_ORDER:
        label = PHASE_LABELS[key]
        if key == current:
            lines.append(f"▶ {label}")
        elif key in reached:
            lines.append(f"✓ {label}")
        elif key == "diarize" and not diarization_enabled:
            lines.append(f"· {label} (vypnuto)")
        else:
            lines.append(f"· {label}")
    box.markdown("  \n".join(lines))


def _render_result(payload: dict) -> None:
    segments = payload.get("segments") or []
    metrics = segment_metrics(segments)
    length = OutputFormatter().format_timestamp(metrics["audio_end"])
    metric_cols = st.columns(4)
    metric_cols[0].metric("Segmenty", metrics["segments"])
    metric_cols[1].metric("Mluvčí", metrics["speakers"])
    metric_cols[2].metric("Délka audia", length)
    metric_cols[3].metric("Slova", metrics["words"])

    for warning in payload.get("warnings") or []:
        st.warning(warning)

    saved = payload.get("output_path") or ""
    if saved:
        st.caption(f"Uloženo ({payload.get('output_format')}): {saved}")

    if not segments:
        st.info("Přepis nevrátil žádné segmenty.")
        return

    names = metrics["speaker_names"]
    options = ["Všichni"] + names
    if "transcript_speaker" not in st.session_state:
        saved_speaker = st.session_state.get("saved_transcript_speaker") or "Všichni"
        st.session_state.transcript_speaker = saved_speaker if saved_speaker in options else "Všichni"
    elif st.session_state.transcript_speaker not in options:
        st.session_state.transcript_speaker = "Všichni"
    st.selectbox("Filtr mluvčího", options, key="transcript_speaker")
    chosen = st.session_state.transcript_speaker
    st.session_state.saved_transcript_speaker = chosen
    speaker = None if chosen == "Všichni" else chosen

    st.dataframe(table_rows(segments, speaker), use_container_width=True, hide_index=True)
    st.text_area(
        "Souvislý text",
        text_by_speaker(segments, speaker) or "(prázdný text)",
        height=240,
        disabled=True,
    )

    stem = Path(saved).stem if saved else "prepis"
    download_cols = st.columns(3)
    for column, fmt in zip(download_cols, OUTPUT_FORMATS):
        with column:
            st.download_button(
                f"Stáhnout {fmt}",
                data=encode_segments(segments, fmt),
                file_name=f"{stem}.{fmt}",
                mime="application/json" if fmt == "json" else "text/plain",
                key=f"tr_dl_{fmt}",
            )

    log_text = payload.get("log_text") or ""
    with st.expander("Protokol běhu"):
        if log_text:
            st.text(log_text)
        else:
            st.caption("Protokol zatím není.")


def render() -> None:
    env = probe_environment()
    render_banner(env)
    st.title("Přepis")
    st.caption("Nahrání audia, nastavení a přepis s volitelnou diarizací mluvčích.")

    audio_path = render_audio_selector()
    render_settings_form(env)

    can_run = can_start(env, audio_path)
    if not audio_path:
        st.caption("Vyberte nebo nahrajte audio.")
    clicked = st.button("Spustit přepis", type="primary", disabled=not can_run)

    if clicked:
        settings = read_settings()
        config = None
        try:
            config = build_config(settings, include_model=True)
            config.validate()
            reached = []
            with st.status("Přepis běží", expanded=True) as status:
                box = st.empty()
                _paint_phases(box, reached, "prepare", settings["diarization_enabled"])

                def on_phase(phase: str) -> None:
                    if phase not in reached:
                        reached.append(phase)
                    status.update(label=PHASE_LABELS.get(phase, "Přepis běží"))
                    _paint_phases(box, reached, phase, settings["diarization_enabled"])

                payload = run_transcription(
                    config,
                    audio_path,
                    settings["output_format"],
                    on_phase=on_phase,
                )
                if "save" not in reached:
                    reached.append("save")
                status.update(label="Hotovo", state="complete")
                _paint_phases(box, reached, "", settings["diarization_enabled"])
            st.session_state.last_transcription = payload
            st.session_state.transcription_error = None
        except Exception as exc:
            secrets = secret_values(config) if config is not None else []
            st.session_state.transcription_error = public_error_message(exc, secrets)

    if st.session_state.get("transcription_error"):
        st.error(st.session_state.transcription_error)
    if st.session_state.get("last_transcription"):
        _render_result(st.session_state.last_transcription)
