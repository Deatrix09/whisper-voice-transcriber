# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
from pydub import AudioSegment
from typing import Dict, List, Tuple
from src.utils.logger import setup_logger
from src.diarization.diarizer_factory import DiarizationFactory
# Pozn.: těžké pyannote/torch importy tu byly nevyužité (mrtvý kód) -
# diarizace se vytváří přes DiarizationFactory (lazy). Odstraněno pro
# rychlý import a funkčnost bez nainstalovaného pyannote/torchaudio.

class Diarizer:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file,
                                     file_level=getattr(config.logging, "file_level", None),
                                     console_level=getattr(config.logging, "console_level", None))
        self.diarizer = None
    
    def initialize_pipeline(self):
        """Initialize the appropriate diarization pipeline based on configuration"""
        if self.diarizer is None:
            self.diarizer = DiarizationFactory.create_diarizer(self.config)
            
            if self.diarizer:
                self.diarizer.initialize()
                self.logger.info(f"Using diarization module: {self.config.diarization.module}")
            else:
                self.logger.warning("Diarization is disabled in configuration")

    def process(self, audio_path: str) -> List[Dict[str, any]]:
        """Process audio file and return diarization results"""
        try:
            if not self.config.diarization.enabled:
                self.logger.info("Diarization is disabled, returning empty result")
                return []
                
            if self.diarizer is None:
                self.initialize_pipeline()
            
            self.logger.info(f"Starting diarization for: {audio_path}")
            results = self.diarizer.process(audio_path)
            self.logger.info(f"Diarization completed with {len(results)} segments")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Diarization failed: {str(e)}")
            raise

    def _ensure_wav_format(self, audio_path: str) -> str:
        if not audio_path.lower().endswith('.wav'):
            self.logger.info(f"Converting audio file to WAV format for consistent processing: {audio_path}")
            temp_wav_path = os.path.splitext(audio_path)[0] + '_temp.wav'
            
            # Load audio with pydub (handles various formats)
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono with consistent sample rate
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)  # 16kHz is standard for many speech processing tasks
            
            # Export to WAV with PCM encoding
            audio.export(temp_wav_path, format="wav", parameters=[
                "-ac", "1",         # mono
                "-ar", "16000",     # 16kHz sample rate
                "-acodec", "pcm_s16le"  # 16-bit PCM encoding
            ])
            
            self.logger.info(f"Created temporary WAV file: {temp_wav_path}")
            return temp_wav_path
        else:
            # For WAV files, check if they need conversion for consistency
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                
                # If WAV file doesn't meet our requirements, convert it
                if info.channels > 1 or info.samplerate != 16000:
                    self.logger.info(f"Converting WAV file to ensure consistent properties: {audio_path}")
                    temp_wav_path = os.path.splitext(audio_path)[0] + '_temp.wav'
                    
                    audio = AudioSegment.from_file(audio_path)
                    audio = audio.set_channels(1)
                    audio = audio.set_frame_rate(16000)
                    
                    audio.export(temp_wav_path, format="wav", parameters=[
                        "-ac", "1", 
                        "-ar", "16000",
                        "-acodec", "pcm_s16le"
                    ])
                    
                    self.logger.info(f"Created temporary WAV file: {temp_wav_path}")
                    return temp_wav_path
            except Exception as e:
                self.logger.warning(f"Could not check WAV file properties, using as-is: {str(e)}")
                
            return audio_path
    
    def _cleanup_temp_file(self, temp_file_path: str):
        import os
        try:
            if os.path.exists(temp_file_path) and '_temp.wav' in temp_file_path:
                os.remove(temp_file_path)
                self.logger.info(f"Cleaned up temporary file: {temp_file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temporary file: {str(e)}")