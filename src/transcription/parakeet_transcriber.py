# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""NeMo Parakeet-TDT dual-engine (FÁZE 5). faster-whisper soubor NESAHÁ.

Model: nvidia/parakeet-tdt-0.6b-v3 (600M, CC-BY-4.0, 25 jazyků vč. cs).
- Load: nemo.collections.asr.models.ASRModel.from_pretrained (NeMo 2.x).
- Greedy decode: TDT dekodér v transcribe() cestě beam nemá (greedy per design).
- bf16 na CUDA (guarded fallback fp32), batch přes transcribe(files, batch_size).
- Nativní word+segment timestamps (timestamps=True), interpunkce z modelu.
- Stejný chunk/merge kontrakt jako Transcriber (30s chunky z processor.py).
- Stejné atributy (config/logger/model/device + initialize_model/transcribe/
  check_ffmpeg) -> Pipeline.release_model_resources funguje beze změny.
"""
import time
from typing import Dict, List, Optional
from src.utils.logger import setup_logger
from src.utils.config import resolve_device
from src.transcription.base_transcriber import ASRProvider
import os
import subprocess
import shutil

import torch

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

# 25 jazyků v3 (kódy dle HF model card; cs_cz WER 11.01 referenčně).
PARAKEET_LANGS = frozenset({
    "en", "es", "fr", "de", "ru", "it", "pl", "uk", "ro", "nl", "hu",
    "el", "sv", "cs", "bg", "pt", "sk", "hr", "da", "fi", "lt", "sl",
    "lv", "et", "mt",
})


def supports_language(lang: Optional[str]) -> bool:
    """Je jazykový kód v 25 podporovaných? ("" / None -> False)."""
    return bool(lang) and str(lang).strip().lower() in PARAKEET_LANGS


class ParakeetTranscriber(ASRProvider):
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file,
                                     file_level=getattr(config.logging, "file_level", None),
                                     console_level=getattr(config.logging, "console_level", None))
        self.model = None
        self.device = "cpu"

    def check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available in the system (chunking ho potřebuje)."""
        try:
            if shutil.which('ffmpeg') is None:
                self.logger.error("ffmpeg not found in system PATH")
                return False
            subprocess.run(['ffmpeg', '-version'],
                         capture_output=True,
                         check=True)
            self.logger.info("ffmpeg is available")
            return True
        except subprocess.SubprocessError as e:
            self.logger.error(f"Error checking ffmpeg: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error checking ffmpeg: {str(e)}")
            return False

    def _model_id(self) -> str:
        asr = getattr(self.config, "asr", None)
        configured = (getattr(asr, "parakeet_model", "") or "").strip() if asr else ""
        return configured or DEFAULT_MODEL

    def _batch_size(self) -> int:
        asr = getattr(self.config, "asr", None)
        try:
            return max(1, int(getattr(asr, "batch_size", 8) or 8))
        except (TypeError, ValueError):
            return 8

    def _vram_gb(self) -> Optional[Dict[str, float]]:
        try:
            if not torch.cuda.is_available():
                return None
            free, total = torch.cuda.mem_get_info()
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "reserved": torch.cuda.memory_reserved() / 1024**3,
                "free": free / 1024**3,
                "total": total / 1024**3,
            }
        except Exception:
            return None

    def _log_vram(self, prefix: str) -> None:
        mem = self._vram_gb()
        if mem is not None:
            self.logger.info(
                f"VRAM [{prefix}]: alloc={mem['allocated']:.2f}GB "
                f"reserved={mem['reserved']:.2f}GB free={mem['free']:.2f}GB"
            )

    def initialize_model(self):
        """Load Parakeet 1x (lazy singleton na instanci)."""
        if self.model is not None:
            return self.model
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            raise RuntimeError(
                "NeMo není nainstalováno (potřeba pro parakeet engine). "
                "Spusť: pip install \"nemo_toolkit['asr']\" (Linux + CUDA "
                "doporučeno; na Windows/CPU nemusí fungovat). Fallback: "
                "asr.engine=faster-whisper."
            ) from e
        try:
            if not self.check_ffmpeg():
                raise RuntimeError("ffmpeg is required but not found. Please install ffmpeg and add it to your PATH.")

            want = getattr(self.config.performance, "device", "cpu")
            try:
                self.device = resolve_device(want)
            except ValueError as e:
                self.logger.warning(f"{e}, fallback cpu")
                self.device = "cpu"
            model_id = self._model_id()

            self._log_vram("before-load")
            t0 = time.perf_counter()
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
            self.model.eval()
            if self.device == "cuda":
                try:
                    self.model = self.model.cuda()
                except Exception as e:
                    self.logger.warning(f"Parakeet .cuda() selhalo ({e}), jedu na CPU")
                    self.device = "cpu"
            if self.device == "cuda":
                try:
                    self.model = self.model.to(torch.bfloat16)  # ~30% VRAM dolů
                    self.logger.info("Parakeet precision: bfloat16")
                except Exception as e:
                    self.logger.warning(f"bf16 nepodporováno ({e}), jedu fp32")
            dt = time.perf_counter() - t0

            self.logger.info(f"Successfully loaded Parakeet model: {model_id}")
            self.logger.info(f"Model device: {self.device}, batch: {self._batch_size()}")
            self.logger.info(f"Model load time: {dt:.1f}s")
            self._log_vram("after-load")
            return self.model
        except Exception as e:
            self.logger.error(f"Failed to initialize Parakeet model: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def _words_of(hyp) -> List[Dict]:
        """Defenzivní mapování word timestamps (NeMo formát se může lišit)."""
        ts = getattr(hyp, "timestamp", None) or {}
        words = ts.get("word", []) if isinstance(ts, dict) else []
        out = []
        for w in words or []:
            if isinstance(w, dict):
                start = w.get("start")
                end = w.get("end")
                text = w.get("word", w.get("text", ""))
            else:
                start = getattr(w, "start", None)
                end = getattr(w, "end", None)
                text = getattr(w, "word", getattr(w, "text", ""))
            if start is None or end is None:
                continue
            out.append({"start": float(start), "end": float(end),
                        "text": str(text).strip()})
        return out

    def _segments_of(self, hyp, chunk_duration: float) -> List[Dict]:
        """Segmenty: nativní (s textem) -> jinak chunk-span fallback (offsety drží)."""
        ts = getattr(hyp, "timestamp", None) or {}
        raw = ts.get("segment", []) if isinstance(ts, dict) else []
        segments: List[Dict] = []
        for s in raw or []:
            if isinstance(s, dict):
                start, end, text = s.get("start"), s.get("end"), s.get("segment", s.get("text", ""))
            else:
                start = getattr(s, "start", None)
                end = getattr(s, "end", None)
                text = getattr(s, "segment", getattr(s, "text", ""))
            if start is None or end is None or not str(text).strip():
                continue
            segments.append({"start": float(start), "end": float(end),
                             "text": str(text).strip()})
        if not segments:
            text = (getattr(hyp, "text", "") or "").strip()
            if text:
                segments = [{"start": 0.0, "end": chunk_duration, "text": text}]
        return segments

    def _chunk_duration(self, chunk_path: str) -> float:
        try:
            import soundfile as sf
            return float(sf.info(chunk_path).duration)
        except Exception:
            return 0.0

    def transcribe(self, audio_path: str) -> Dict:
        try:
            if self.model is None:
                self.initialize_model()

            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            abs_audio_path = os.path.abspath(audio_path)
            self.logger.info(f"Transcribing file (parakeet): {abs_audio_path}")

            from src.audio.processor import AudioProcessor
            audio_processor = AudioProcessor(self.config)
            chunk_files = audio_processor.split_audio_file(abs_audio_path)
            self.logger.info(f"Parakeet: transcribing {len(chunk_files)} chunk(s) "
                             f"v 1 dávce (batch={self._batch_size()})...")

            align = bool(getattr(getattr(self.config, "asr", None),
                                 "align_words", False))
            t0 = time.perf_counter()
            with torch.inference_mode():
                # JEDEN batched call pro všechny chunky (ne loop) + nativní timestamps.
                hyps = self.model.transcribe(chunk_files,
                                             batch_size=self._batch_size(),
                                             timestamps=True)
            dt = time.perf_counter() - t0

            try:
                total_dur = sum(self._chunk_duration(c) for c in chunk_files)
                rtf = total_dur / dt if dt > 0 else 0.0
            except Exception:
                total_dur, rtf = -1.0, -1.0
            self.logger.info(
                f"Parakeet batch done: audio={total_dur:.1f}s infer={dt:.1f}s "
                f"RTFx={rtf:.2f} chunks={len(chunk_files)}")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self._log_vram("after-transcribe")

            # Převeď na pipeline formát (stejný jako faster-whisper cesta).
            language = (self.config.model.language or "").strip() or "auto"
            if len(chunk_files) == 1:
                hyps = list(hyps)
                seg_list = self._segments_of(hyps[0], self._chunk_duration(chunk_files[0]))
                words = self._words_of(hyps[0]) if align else []
                result = {"text": " ".join(s["text"] for s in seg_list),
                          "segments": seg_list, "language": language}
                if align:
                    result["words"] = words
            else:
                chunk_results = []
                for hyp, chunk_file in zip(hyps, chunk_files):
                    seg_list = self._segments_of(hyp, self._chunk_duration(chunk_file))
                    entry = {"text": " ".join(s["text"] for s in seg_list),
                             "segments": seg_list, "language": language}
                    if align:
                        entry["words"] = self._words_of(hyp)
                    chunk_results.append(entry)
                result = audio_processor.merge_transcription_results(chunk_results)

            audio_processor.cleanup()
            self.logger.info(f"Successfully transcribed audio file: {abs_audio_path}")
            return result

        except Exception as e:
            self.logger.error(f"Parakeet transcription failed: {str(e)}", exc_info=True)
            raise


def create_asr_transcriber(config):
    """P0 0.4: zpětná kompatibilita - deleguje do asr_factory."""
    from src.transcription.asr_factory import create_asr_provider
    return create_asr_provider(config)
