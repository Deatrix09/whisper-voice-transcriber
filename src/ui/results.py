# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Metriky a stažení z paměti. Formatter se nemění, jen se volá."""

import json
from typing import Optional

from src.utils.formatters import OutputFormatter


def segment_metrics(segments: list) -> dict:
    speakers = []
    for segment in segments:
        speaker = segment.get("speaker") or "UNKNOWN"
        if speaker not in speakers:
            speakers.append(speaker)
    audio_end = 0.0
    words = 0
    for segment in segments:
        audio_end = max(audio_end, float(segment.get("end") or 0))
        words += len(str(segment.get("text") or "").split())
    return {
        "segments": len(segments),
        "speakers": len(speakers),
        "speaker_names": speakers,
        "audio_end": audio_end,
        "words": words,
    }


def table_rows(segments: list, speaker: Optional[str] = None) -> list:
    formatter = OutputFormatter()
    rows = []
    for segment in segments:
        name = segment.get("speaker") or "UNKNOWN"
        if speaker and name != speaker:
            continue
        rows.append({
            "Od": formatter.format_timestamp(float(segment.get("start") or 0)),
            "Do": formatter.format_timestamp(float(segment.get("end") or 0)),
            "Mluvčí": name,
            "Text": str(segment.get("text") or "").strip(),
        })
    return rows


def text_by_speaker(segments: list, speaker: Optional[str] = None) -> str:
    blocks = []
    current = None
    parts = []
    for segment in segments:
        name = segment.get("speaker") or "UNKNOWN"
        if speaker and name != speaker:
            continue
        text = str(segment.get("text") or "").strip()
        if name != current:
            if parts:
                blocks.append(f"{current}: {' '.join(parts)}")
            current = name
            parts = [text] if text else []
        elif text:
            parts.append(text)
    if parts:
        blocks.append(f"{current}: {' '.join(parts)}")
    return "\n\n".join(blocks)


def encode_segments(segments: list, output_format: str) -> bytes:
    formatter = OutputFormatter()
    if output_format == "json":
        return json.dumps(segments, ensure_ascii=False, indent=2).encode("utf-8")
    if output_format == "txt":
        return formatter.to_plain_text(segments).encode("utf-8")
    if output_format == "subtitle":
        return formatter.to_subtitle_format(segments).encode("utf-8")
    raise ValueError(f"Nepodporovaný formát: {output_format}")
