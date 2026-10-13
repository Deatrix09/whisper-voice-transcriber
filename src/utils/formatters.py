# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from typing import List, Dict
from datetime import timedelta
import json

class OutputFormatter:
    def format_timestamp(self, seconds: float) -> str:
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        seconds = td.total_seconds() % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def to_subtitle_format(self, results: List[Dict]) -> str:
        output_lines = []
        
        for segment in results:
            start_time = self.format_timestamp(segment["start"])
            end_time = self.format_timestamp(segment["end"])
            
            # Format: [00:00:00.000 --> 00:00:03.760] SPEAKER_01: Text
            line = f"[{start_time} --> {end_time}] {segment['speaker']}: {segment['text'].strip()}"
            output_lines.append(line)
        
        return "\n".join(output_lines)

    def to_plain_text(self, results: List[Dict]) -> str:
        output_lines = []
        
        for segment in results:
            start_time = self.format_timestamp(segment["start"])
            end_time = self.format_timestamp(segment["end"])
            
            # Format: [00:00:00.000 - 00:00:03.760] SPEAKER_01: Text
            line = f"[{start_time} - {end_time}] {segment['speaker']}: {segment['text'].strip()}"
            output_lines.append(line)
        
        return "\n".join(output_lines)

    def save_output(self, results: List[Dict], output_path: str, output_format: str) -> None:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_format == 'json':
                    json.dump(results, f, ensure_ascii=False, indent=2)
                elif output_format == 'txt':
                    f.write(self.to_plain_text(results))
                elif output_format == 'subtitle':
                    f.write(self.to_subtitle_format(results))
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")
                    
        except Exception as e:
            raise IOError(f"Failed to save output: {str(e)}") 