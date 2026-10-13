# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Sestavení AppConfig z hodnot formuláře. Na disk se nezapisuje."""

import copy
import os
import re
from typing import Optional

from src.utils.config import AppConfig

WHISPER_MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "turbo",
)

# faster-whisper bere "turbo" i "large-v3-turbo" jako stejný model.
WHISPER_MODEL_ALIASES = {
    "large-v3-turbo": "turbo",
    "turbo": "turbo",
}

OUTPUT_FORMATS = ("txt", "json", "subtitle")
TASKS = (
    ("Přepis", "transcribe"),
    ("Překlad do angličtiny", "translate"),
)
DIARIZATION_MODULES = ("pyannote", "nemo")
MODULE_LABELS = {
    "pyannote": "Pyannote",
    "nemo": "NeMo",
    "nemotron-3": "Nemotron-3",
    "sortformer": "Sortformer",
}

# Název z whisper_language_codes.json -> český popisek. Neznámý název zůstane.
CZECH_LANGUAGE_NAMES = {
    "english": "Angličtina",
    "chinese": "Čínština",
    "german": "Němčina",
    "spanish": "Španělština",
    "russian": "Ruština",
    "korean": "Korejština",
    "french": "Francouzština",
    "japanese": "Japonština",
    "portuguese": "Portugalština",
    "turkish": "Turečtina",
    "polish": "Polština",
    "catalan": "Katalánština",
    "dutch": "Nizozemština",
    "arabic": "Arabština",
    "swedish": "Švédština",
    "italian": "Italština",
    "indonesian": "Indonéština",
    "hindi": "Hindština",
    "finnish": "Finština",
    "vietnamese": "Vietnamština",
    "hebrew": "Hebrejština",
    "ukrainian": "Ukrajinština",
    "greek": "Řečtina",
    "malay": "Malajština",
    "czech": "Čeština",
    "romanian": "Rumunština",
    "danish": "Dánština",
    "hungarian": "Maďarština",
    "tamil": "Tamilština",
    "norwegian": "Norština",
    "thai": "Thajština",
    "urdu": "Urdština",
    "croatian": "Chorvatština",
    "bulgarian": "Bulharština",
    "lithuanian": "Litevština",
    "latin": "Latina",
    "maori": "Maorština",
    "malayalam": "Malajálamština",
    "welsh": "Velština",
    "slovak": "Slovenština",
    "telugu": "Telugština",
    "persian": "Perština",
    "latvian": "Lotyština",
    "bengali": "Bengálština",
    "serbian": "Srbština",
    "azerbaijani": "Ázerbájdžánština",
    "slovenian": "Slovinština",
    "kannada": "Kannadština",
    "estonian": "Estonština",
    "macedonian": "Makedonština",
    "breton": "Bretonština",
    "basque": "Baskičtina",
    "icelandic": "Islandština",
    "armenian": "Arménština",
    "nepali": "Nepálština",
    "mongolian": "Mongolština",
    "bosnian": "Bosenština",
    "kazakh": "Kazaština",
    "albanian": "Albánština",
    "swahili": "Svahilština",
    "galician": "Galicijština",
    "marathi": "Maráthština",
    "punjabi": "Paňdžábština",
    "sinhala": "Sinhálština",
    "khmer": "Khmerština",
    "shona": "Šonština",
    "yoruba": "Jorubština",
    "somali": "Somálština",
    "afrikaans": "Afrikánština",
    "occitan": "Okcitánština",
    "georgian": "Gruzínština",
    "belarusian": "Běloruština",
    "tajik": "Tádžičtina",
    "sindhi": "Sindhština",
    "gujarati": "Gudžarátština",
    "amharic": "Amharština",
    "yiddish": "Jidiš",
    "lao": "Laoština",
    "uzbek": "Uzbečtina",
    "faroese": "Faerština",
    "haitian creole": "Haitská kreolština",
    "pashto": "Paštština",
    "turkmen": "Turkmenština",
    "nynorsk": "Nynorsk",
    "maltese": "Maltština",
    "sanskrit": "Sanskrit",
    "luxembourgish": "Lucemburština",
    "myanmar": "Barmština",
    "tibetan": "Tibetština",
    "tagalog": "Tagalog",
    "malagasy": "Malgaština",
    "assamese": "Ásámština",
    "tatar": "Tatarština",
    "hawaiian": "Havajština",
    "lingala": "Lingalština",
    "hausa": "Hauština",
    "bashkir": "Baškirština",
    "javanese": "Javánština",
    "sundanese": "Sundština",
}

PHASE_ORDER = ("prepare", "diarize", "transcribe", "merge", "save")
PHASE_LABELS = {
    "prepare": "Příprava souboru",
    "diarize": "Diarizace",
    "transcribe": "Přepis",
    "merge": "Sloučení",
    "save": "Uložení",
}

_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{8,}")
_AUTH_TOKEN_RE = re.compile(r"(auth_token\s*[:=]\s*)(\S+)", re.IGNORECASE)
_ENV_SECRET_VARS = (
    "HF_TOKEN",
    "HUGGINGFACE_ACCESS_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def display_model_name(name: str) -> str:
    text = (name or "").strip()
    mapped = WHISPER_MODEL_ALIASES.get(text, text)
    if mapped in WHISPER_MODELS:
        return mapped
    return "turbo"


def module_options(selected: str) -> list:
    """Pyannote a NeMo jsou vždy v nabídce. Jiný modul z YAML zůstane výchozí."""
    options = list(DIARIZATION_MODULES)
    current = (selected or "").strip()
    if current and current not in options:
        options.insert(0, current)
    return options


def module_label(name: str) -> str:
    return MODULE_LABELS.get(name, name)


def language_choices(mapping: dict) -> list:
    """Seznam (popisek, kód). První položka je automatická detekce."""
    items = []
    for name, code in mapping.items():
        label = CZECH_LANGUAGE_NAMES.get(str(name), str(name)[:1].upper() + str(name)[1:])
        items.append((f"{label} ({code})", str(code)))
    items.sort(key=lambda item: item[0].lower())
    return [("Automaticky", "")] + items


def load_language_choices(path: str = "config/whisper_language_codes.json") -> list:
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("whisper_language_codes.json musí být mapa název -> kód")
    return language_choices(payload)


def defaults_from_config(config: AppConfig, *, cuda_available: bool) -> dict:
    """Hodnoty widgetů. Bez tokenu a bez polí, která formulář nemění."""
    asr = getattr(config, "asr", None)
    effective = ""
    if asr is not None:
        effective = (getattr(asr, "model", "") or "").strip()
    if not effective:
        effective = str(config.model.name).strip()
    device = str(getattr(config.performance, "device", "cpu") or "cpu").strip().lower()
    if device == "auto":
        device = "cuda" if cuda_available else "cpu"
    if device == "cuda" and not cuda_available:
        device = "cpu"
    if device not in {"cuda", "cpu"}:
        device = "cpu"
    output_format = config.output.format if config.output.format in OUTPUT_FORMATS else "txt"
    task = config.transcription.task if config.transcription.task in {"transcribe", "translate"} else "translate"
    speakers = getattr(config.diarization, "num_speakers", None)
    return {
        "model": display_model_name(effective),
        "language": config.model.language or "",
        "task": task,
        "output_format": output_format,
        "device": device,
        "diarization_enabled": bool(config.diarization.enabled),
        "diarization_module": str(config.diarization.module),
        "num_speakers": "" if speakers is None else str(speakers),
        "temperature": float(config.transcription.temperature),
        "chunk_size": int(config.audio.chunk_size),
        "chunk_threshold_mb": int(config.audio.chunk_threshold_mb),
        "chunk_overlap_sec": int(config.audio.chunk_overlap_sec),
        "num_workers": int(config.performance.num_workers),
        "fp16": bool(config.transcription.fp16),
    }


def parse_num_speakers(raw) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError("Počet mluvčích musí být celé číslo ≥ 1, nebo prázdné pole.")
    if isinstance(raw, int):
        if raw < 1:
            raise ValueError("Počet mluvčích musí být celé číslo ≥ 1, nebo prázdné pole.")
        return raw
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("Počet mluvčích musí být celé číslo ≥ 1, nebo prázdné pole.")
    value = int(text)
    if value < 1:
        raise ValueError("Počet mluvčích musí být celé číslo ≥ 1, nebo prázdné pole.")
    return value


def _apply_model(config: AppConfig, selected: str) -> None:
    """Když výběr odpovídá YAML (včetně aliasu turbo), nech původní řetězce."""
    if selected not in WHISPER_MODELS:
        raise ValueError(f"Neznámý model: {selected}")
    asr = getattr(config, "asr", None)
    current = ""
    if asr is not None:
        current = (getattr(asr, "model", "") or "").strip()
    if not current:
        current = str(config.model.name).strip()
    if display_model_name(current) == selected:
        return
    config.model.name = selected
    if asr is not None:
        asr.engine = "faster-whisper"
        asr.model = selected


def apply_ui_settings(config: AppConfig, settings: dict, *, include_model: bool) -> AppConfig:
    """Přepiš jen pole z formuláře. Token a tiché prahy Whisperu zůstanou."""
    if include_model:
        _apply_model(config, settings["model"])
    language = settings.get("language") or ""
    config.model.language = "" if language == "" else str(language)
    task = settings["task"]
    if task not in {"transcribe", "translate"}:
        raise ValueError("Úkol musí být transcribe nebo translate.")
    config.transcription.task = task
    output_format = settings["output_format"]
    if output_format not in OUTPUT_FORMATS:
        raise ValueError("Výstupní formát musí být txt, json nebo subtitle.")
    config.output.format = output_format
    device = settings["device"]
    if device not in {"cuda", "cpu"}:
        raise ValueError("Zařízení musí být cuda nebo cpu.")
    config.performance.device = device
    config.performance.num_workers = int(settings["num_workers"])
    config.audio.chunk_size = int(settings["chunk_size"])
    config.audio.chunk_threshold_mb = int(settings["chunk_threshold_mb"])
    config.audio.chunk_overlap_sec = int(settings["chunk_overlap_sec"])
    config.transcription.temperature = float(settings["temperature"])
    config.transcription.fp16 = bool(settings["fp16"])
    config.diarization.enabled = bool(settings["diarization_enabled"])
    config.diarization.module = str(settings["diarization_module"])
    config.diarization.num_speakers = parse_num_speakers(settings.get("num_speakers"))
    return config


def build_config(
    settings: dict,
    *,
    include_model: bool,
    config_path: str = "config/config.yaml",
) -> AppConfig:
    """Čerstvá kopie YAML. Objekt v session state se tím nemění."""
    config = copy.deepcopy(AppConfig.from_yaml(config_path))
    return apply_ui_settings(config, settings, include_model=include_model)


def env_secrets() -> list:
    found = []
    for name in _ENV_SECRET_VARS:
        raw = (os.environ.get(name) or "").strip()
        if len(raw) >= 8:
            found.append(raw)
    return found


def secret_values(config: Optional[AppConfig] = None) -> list:
    found = env_secrets()
    if config is None:
        return found
    pyannote = getattr(getattr(config, "diarization", None), "pyannote", None)
    token = (getattr(pyannote, "auth_token", "") or "").strip()
    if token and not token.startswith("${") and len(token) >= 8 and token not in found:
        found.append(token)
    return found


def redact(text: str, secrets: Optional[list] = None) -> str:
    if not text:
        return ""
    redacted = str(text)
    for secret in secrets or []:
        if secret:
            redacted = redacted.replace(secret, "[skryto]")
    redacted = _HF_TOKEN_RE.sub("[skryto]", redacted)
    return _AUTH_TOKEN_RE.sub(r"\1[skryto]", redacted)


def public_error_message(exc: BaseException, secrets: Optional[list] = None) -> str:
    """Zpráva pro UI. Bez tracebacku a bez konfigurace."""
    raw = str(exc).strip() or exc.__class__.__name__
    text = redact(raw, secrets)
    nemo_failed = isinstance(exc, ImportError) and "nemo" in raw.lower()
    if nemo_failed or "setup_nemo.py" in raw or "Failed to import NeMo" in raw:
        hint = (
            "NeMo není nainstalované. "
            "Nainstalujte ho příkazem: python scripts/setup_nemo.py"
        )
        if "setup_nemo.py" not in text:
            text = hint + "\n" + text
    return text


def phase_from_log(message: str) -> Optional[str]:
    """Fáze podle existujících hlášek pipeline. Žádný odhad procent."""
    text = (message or "").lower()
    if (
        "stage 3" in text
        or "merging diarization" in text
        or "results without diarization" in text
        or "starting to merge" in text
    ):
        return "merge"
    if (
        "transcription only" in text
        or "starting transcription" in text
        or "transcription completed" in text
    ):
        return "transcribe"
    if "diarization" in text or "diariz" in text:
        return "diarize"
    if (
        "stage 1" in text
        or "loading audio" in text
        or "starting processing" in text
        or "audio file size" in text
        or "audio ready" in text
        or "initializing pipeline" in text
    ):
        return "prepare"
    return None
