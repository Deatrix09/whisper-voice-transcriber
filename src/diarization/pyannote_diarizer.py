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
import torch
from src.utils.logger import setup_logger
from src.diarization.base_diarizer import BaseDiarizer
# Pozn.: pyannote importy (Pipeline, safe_globals, task) jsou lazy v initialize(),
# aby šel modul importovat bez pyannote/torchaudio a cold start byl rychlejší.

class PyAnnoteDiarizer(BaseDiarizer):
    # pyannote.audio 4.x + community-1 (CC-BY-4.0, free; NE precision-2 paid).
    # Staré 3.1 ID ("...speaker-diarization-3.1") jedou legacy větví dál.
    MODERN_MARKERS = ("community", "precision")

    def __init__(self, config):
        super().__init__(config)
        self.pipeline = None
        self._is_modern = False

    def _resolve_token(self) -> str:
        # P0 0.1: config nese "${HF_TOKEN}" placeholder (expandovaný už ve
        # from_yaml) nebo prázdné -> bereme z prostředí. Pořadí: HF_TOKEN,
        # HUGGINGFACE_ACCESS_TOKEN, HUGGING_FACE_HUB_TOKEN.
        raw = (getattr(self.config.diarization.pyannote, "auth_token", "") or "").strip()
        if raw.startswith("${"):
            raw = ""  # nevyexpandovaný placeholder (chybí .env) -> jako prázdné
        token = raw
        if not token:
            for var in ("HF_TOKEN", "HUGGINGFACE_ACCESS_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                token = (os.environ.get(var, "") or "").strip()
                if token:
                    break
        if not token:
            raise ValueError(
                "pyannote: chybí HuggingFace token (gated model) - FAIL FAST před diarizací. "
                "Nastav HF_TOKEN v .env (viz .env.example) nebo v prostředí; "
                "token vytvoříš na https://hf.co/settings/tokens + přijmi podmínky modelu."
            )
        return token

    def _is_modern_model(self, model_id: str) -> bool:
        return any(m in model_id.lower() for m in self.MODERN_MARKERS)

    def _resolve_batch_size(self) -> int:
        """Batch pro segmentaci/embeddingy z asr.batch_size (FÁZE 4).

        faster-whisper file-batch API nemá (num_workers je pro concurrent
        transcribe volání, ne pro jeden soubor) - batch_size tak řídí
        pyannote batche + později parakeet (FÁZE 5).
        """
        asr = getattr(self.config, "asr", None)
        try:
            return max(1, int(getattr(asr, "batch_size", 1) or 1))
        except (TypeError, ValueError):
            return 1

    def initialize(self):
        try:
            self.logger.info("Initializing PyAnnote diarization pipeline...")
            token = self._resolve_token()

            try:
                from pyannote.audio import Pipeline
                from torch.serialization import add_safe_globals
                from torch.torch_version import TorchVersion
                from pyannote.audio.core.task import Specifications, Problem, Resolution
            except ImportError as e:
                raise RuntimeError(
                    "pyannote.audio není nainstalováno. Spusť: "
                    "pip install \"pyannote.audio>=4.0\" (pro community-1)."
                ) from e
            except Exception as e:
                # Např. nekompatibilní torchaudio (chybějící AudioMetaData):
                # instalace je rozbitá, ne jen chybějící.
                raise RuntimeError(
                    f"pyannote.audio se nepodařilo importovat ({e}). Zkontroluj "
                    f"kompatibilní verzi: pip install \"pyannote.audio>=4.0\" "
                    f"+ odpovídající torchaudio."
                ) from e

            # Register safe globals for PyTorch 2.6+ compatibility
            add_safe_globals([TorchVersion, Specifications, Problem, Resolution])
            self.logger.info("Registered safe globals for PyTorch compatibility")

            model_id = self.config.diarization.pyannote.model
            self._is_modern = self._is_modern_model(model_id)

            if self._is_modern:
                # pyannote.audio 4.x API: token= (use_auth_token je pryč).
                self.pipeline = Pipeline.from_pretrained(model_id, token=token)
            else:
                self.pipeline = Pipeline.from_pretrained(
                    model_id,
                    use_auth_token=token
                )

            self.logger.info("Setting pipeline parameters:")
            self.logger.info(f"- Min duration off: {self.config.diarization.pyannote.min_duration_off}")
            if self._is_modern:
                # 4.x clustering má jiné klíče (threshold/Fa/Fb, default VBx):
                # přepisujeme jen segmentaci, zbytek = defaulty pipeline.
                self.pipeline.instantiate({
                    "segmentation": {
                        "min_duration_off": self.config.diarization.pyannote.min_duration_off
                    },
                })
            else:
                self.logger.info(f"- Min cluster size: {self.config.diarization.pyannote.min_cluster_size}")
                self.logger.info(f"- Clustering method: {self.config.diarization.pyannote.method}")
                # Set pipeline parameters
                self.pipeline.instantiate({
                    "segmentation": {
                        "min_duration_off": self.config.diarization.pyannote.min_duration_off
                    },
                    "clustering": {
                        "method": self.config.diarization.pyannote.method,
                        "min_cluster_size": self.config.diarization.pyannote.min_cluster_size
                    }
                })

            batch_size = self._resolve_batch_size()
            if batch_size > 1:
                for attr in ("segmentation_batch_size", "embedding_batch_size"):
                    try:
                        setattr(self.pipeline, attr, batch_size)
                        self.logger.info(f"Pyannote {attr} = {batch_size}")
                    except Exception as e:
                        self.logger.warning(f"Nelze nastavit {attr}: {e}")

            if torch.cuda.is_available() and self._effective_device() == "cuda":
                try:
                    self.pipeline = self.pipeline.to(torch.device("cuda"))
                    self.logger.info("Diarization pipeline moved to CUDA")
                except Exception as e:
                    self.logger.warning(f"Pipeline .to(cuda) selhalo ({e}), jedu na CPU")
            else:
                self.logger.info("Diarization pipeline running on CPU")

            self.logger.info("Successfully initialized PyAnnote diarization pipeline")

        except Exception as e:
            self.logger.error(f"Failed to initialize PyAnnote diarization pipeline: {str(e)}")
            raise

    def _effective_device(self) -> str:
        """P0 0.3: auto|cuda|cpu -> cuda|cpu (neznámé = cpu + warning)."""
        from src.utils.config import resolve_device
        try:
            return resolve_device(getattr(self.config.performance, "device", "cpu"))
        except ValueError as e:
            self.logger.warning(f"{e}, fallback cpu")
            return "cpu"

    def _speaker_kwargs(self) -> dict:
        """num_speakers z configu: 4.x jako min/max hint, 3.1 jako exact."""
        ns = getattr(self.config.diarization, "num_speakers", None)
        if ns is None:
            self.logger.info("Using automatic speaker detection")
            return {}
        self.logger.info(f"Using specified number of speakers: {ns}")
        if self._is_modern:
            return {"min_speakers": ns, "max_speakers": ns}
        return {"num_speakers": ns}

    @staticmethod
    def _select_annotation(diarization):
        """Preferuj exclusive (bez překryvů, 4.x), fallback regular/legacy."""
        exclusive = getattr(diarization, "exclusive_speaker_diarization", None)
        if exclusive is not None:
            return exclusive
        regular = getattr(diarization, "speaker_diarization", None)
        if regular is not None:
            return regular
        return diarization  # legacy Annotation z 3.1

    _DECODER_ERROR_HINTS = ("audiodecoder", "torchcodec", "ffmpeg")

    def _run_pipeline(self, wav_path: str):
        """Pipeline nad cestou k souboru (pyannote 4 dekóduje přes torchcodec, šetří RAM).

        Bez torchcodec / sdílených FFmpeg knihoven (typicky Windows) fallback:
        WAV (16 kHz mono z _ensure_wav_format) načtený přes soundfile do paměti.
        """
        kwargs = self._speaker_kwargs()
        try:
            return self.pipeline(wav_path, **kwargs)
        except Exception as e:
            text = f"{type(e).__name__}: {e}".lower()
            if not any(h in text for h in self._DECODER_ERROR_HINTS):
                raise
            self.logger.warning(
                f"pyannote nemůže dekódovat soubor ({e}); fallback na waveform v paměti"
            )
            import soundfile as sf
            data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T.copy())  # (channel, time)
            return self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate}, **kwargs
            )

    def process(self, audio_path: str) -> List[Dict[str, any]]:
        try:
            if self.pipeline is None:
                self.initialize()
                
            self.logger.info(f"Starting PyAnnote diarization for: {audio_path}")
            
            # Ensure audio file is in WAV format to prevent tensor size mismatch errors
            wav_audio_path = self._ensure_wav_format(audio_path)
            
            # Run diarization
            self.logger.info("Running PyAnnote diarization pipeline...")
            try:
                diarization = self._run_pipeline(wav_audio_path)
            finally:
                # Clean up temporary file if needed
                if wav_audio_path != audio_path:
                    self._cleanup_temp_file(wav_audio_path)

            annotation = self._select_annotation(diarization)
            results = []
            speaker_segments = {}

            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segment = {
                    "speaker": self.normalize_speaker_label(speaker),
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.end - turn.start
                }
                results.append(segment)
                
                # Track speaker statistics
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append(segment)
            
            self.logger.info("Diarization Results Summary:")
            self.logger.info(f"Total segments: {len(results)}")
            self.logger.info(f"Unique speakers detected: {len(speaker_segments)}")
            
            for speaker, segments in speaker_segments.items():
                total_duration = sum(seg["duration"] for seg in segments)
                avg_duration = total_duration / len(segments)
                self.logger.info(f"Speaker {speaker}:")
                self.logger.info(f"  - Segments: {len(segments)}")
                self.logger.info(f"  - Total speaking time: {total_duration:.2f} seconds")
                self.logger.info(f"  - Average segment duration: {avg_duration:.2f} seconds")
            
            # Log potential issues
            short_segments = [s for s in results if s["duration"] < 1.0]
            if short_segments:
                self.logger.warning(f"Found {len(short_segments)} segments shorter than 1 second")
            
            overlapping = 0
            for i in range(len(results)-1):
                if results[i]["end"] > results[i+1]["start"]:
                    overlapping += 1
            if overlapping:
                self.logger.warning(f"Found {overlapping} overlapping segments")

            # Handling overlapu: exclusive anotace (4.x) už je bez překryvů,
            # pro ostatní cesty ořež pozdější segment (sjednocení časů drží).
            results = self.resolve_overlaps(results)

            self.logger.info(f"Successfully completed PyAnnote diarization for {audio_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"PyAnnote diarization failed: {str(e)}")
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
