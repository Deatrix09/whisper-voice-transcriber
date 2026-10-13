# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Grafické rozhraní. Spouští se z kořene projektu: streamlit run app.py"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

from src.ui.settings import public_error_message


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def main() -> None:
    st.set_page_config(page_title="Přepis hlasu", layout="wide")
    from src.ui.pages.batch import render as render_batch
    from src.ui.pages.compare import render as render_compare
    from src.ui.pages.history import render as render_history
    from src.ui.pages.transcribe import render as render_transcribe
    from src.ui.state import ensure_settings_state

    try:
        ensure_settings_state()
    except Exception as exc:
        st.error(public_error_message(exc))
        return

    st.sidebar.title("Přepis hlasu")
    page = st.sidebar.radio(
        "Navigace",
        ["Přepis", "Porovnání modelů", "Dávka konfigurací", "Historie"],
    )
    pages = {
        "Přepis": render_transcribe,
        "Porovnání modelů": render_compare,
        "Dávka konfigurací": render_batch,
        "Historie": render_history,
    }
    try:
        pages[page]()
    except Exception as exc:
        st.error(public_error_message(exc))


if __name__ == "__main__":
    if _in_streamlit():
        main()
    else:
        raise SystemExit("Spusťte z kořene projektu: streamlit run app.py")
