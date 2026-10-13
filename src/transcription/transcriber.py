import time
import torch
from typing import Dict, List, Optional
from src.utils.logger import setup_logger
import os
import subprocess
import shutil

try:
    from faster_whisper import WhisperModel
    _FasterWhisperMissing = None
except ImportError as _e:  # pragma: no cover - srozumitelná hláška místo tracebacku
    WhisperModel = None
    _FasterWhisperMissing = _e

# Výchozí VAD parametry (Silero VAD v6, ONNX bundlovaný ve faster-whisper).
# Těsnější min_silence než package default (2000 ms) -> méně halucinací v tichu.
# Vše přepisovatelné přes config.transcription (getattr fallback = staré YAML fungují).
_DEFAULT_VAD = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 200,
}

# faster-whisper akceptuje openai názvy + turbo aliasy (ověřeno v 1.2.1).
_MODEL_ALIASES = {
    "turbo": "turbo",  # alias large-v3-turbo, CTranslate2 ho zná
}


class Transcriber:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file,
                                     file_level=getattr(config.logging, "file_level", None),
                                     console_level=getattr(config.logging, "console_level", None))
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"

    def check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available in the system."""
        try:
            # Check if ffmpeg is in PATH
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

    def _resolve_device(self) -> str:
        want_cuda = getattr(self.config.performance, "device", "cpu") == "cuda"
        try:
            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False
        return "cuda" if (want_cuda and has_cuda) else "cpu"

    def _supported_compute_types(self, device: str) -> Optional[set]:
        """CT2-verifikovaná množina compute typů (4.8.3: bez bf16!)."""
        try:
            import ctranslate2
            return set(ctranslate2.get_supported_compute_types(device))
        except Exception:
            return None

    def _auto_compute_type(self, device: str) -> str:
        # FÁZE 4 verdikt: bf16 v CT2 4.8.3 NEEXISTUJE (ověřeno
        # get_supported_compute_types: cuda={int8,int8_float32,int8_float16,
        # float16,float32}). Nejlepší dostupné: int8_float16 / int8.
        return "int8_float16" if device == "cuda" else "int8"

    def _resolve_compute_type(self, device: str) -> str:
        asr = getattr(self.config, "asr", None)
        override = (getattr(asr, "compute_type", None)
                    or getattr(self.config.transcription, "compute_type", None))
        if override:
            override = str(override)
            supported = self._supported_compute_types(device)
            if supported is not None and override not in supported:
                fallback = self._auto_compute_type(device)
                self.logger.warning(
                    f"compute_type={override} není v CT2 podporováno na {device} "
                    f"({sorted(supported)}), fallback {fallback}")
                return fallback
            return override
        return self._auto_compute_type(device)

    def _resolve_model_name(self) -> str:
        # FÁZE 4: nový `asr.model` má přednost, jinak legacy `model.name`
        # (staré config.yaml beze změny fungují).
        asr = getattr(self.config, "asr", None)
        configured = (getattr(asr, "model", "") or "").strip() if asr else ""
        legacy = str(self.config.model.name).strip()
        name = configured or legacy
        if configured and configured != legacy:
            self.logger.info(f"ASR model z asr.model: {name} (model.name ignorováno)")
        return _MODEL_ALIASES.get(name, name)

    def _resolve_engine(self) -> str:
        asr = getattr(self.config, "asr", None)
        return (getattr(asr, "engine", "") or "faster-whisper").strip().lower()

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
        """Initialize the faster-whisper model (lazy singleton, 1x na instanci)."""
        if self.model is not None:
            return self.model
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper není nainstalován. "
                "Spusť: pip install \"faster-whisper==1.2.1\""
            ) from _FasterWhisperMissing
        try:
            if not self.check_ffmpeg():
                raise RuntimeError("ffmpeg is required but not found. Please install ffmpeg and add it to your PATH.")

            self.device = self._resolve_device()
            self.compute_type = self._resolve_compute_type(self.device)
            model_name = self._resolve_model_name()
            engine = self._resolve_engine()
            if engine not in ("faster-whisper",):
                # auto|parakeet router přistane ve FÁZI 5; do té doby fallback.
                self.logger.warning(
                    f"asr.engine={engine}: router až ve FÁZI 5, "
                    f"fallback na faster-whisper")

            self._log_vram("before-load")
            t0 = time.perf_counter()
            # Model se stahuje do HF cache při prvním použití (jako dřív whisper).
            # Při selhání preferovaného compute_type retry se safe defaultem
            # (např. budoucí bf16 na starém CT2) místo pádu celého běhu.
            try:
                self.model = WhisperModel(
                    model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as e:
                safe = "int8_float16" if self.device == "cuda" else "int8"
                if self.compute_type == safe:
                    raise
                self.logger.warning(
                    f"Load s compute_type={self.compute_type} selhal ({e}), "
                    f"retry {safe}")
                self.compute_type = safe
                self.model = WhisperModel(
                    model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            dt = time.perf_counter() - t0

            self.logger.info(f"Successfully loaded faster-whisper model: {model_name}")
            self.logger.info(f"Model device: {self.device}, compute_type: {self.compute_type}")
            self.logger.info(f"Model load time: {dt:.1f}s")
            self._log_vram("after-load")

            if self.config.model.language:
                self.logger.info(f"Model language set to: {self.config.model.language}")

            return self.model
        except Exception as e:
            self.logger.error(f"Failed to initialize faster-whisper model: {str(e)}", exc_info=True)
            raise

    def _vad_parameters(self) -> dict:
        tr = self.config.transcription
        return {
            "threshold": float(getattr(tr, "vad_threshold", _DEFAULT_VAD["threshold"])),
            "min_speech_duration_ms": int(getattr(tr, "vad_min_speech_ms", _DEFAULT_VAD["min_speech_duration_ms"])),
            "min_silence_duration_ms": int(getattr(tr, "vad_min_silence_ms", _DEFAULT_VAD["min_silence_duration_ms"])),
            "speech_pad_ms": int(getattr(tr, "vad_speech_pad_ms", _DEFAULT_VAD["speech_pad_ms"])),
        }

    def _transcribe_options(self) -> Dict:
        tr = self.config.transcription
        asr = getattr(self.config, "asr", None)
        # Greedy decoding: beam 1 + bez conditioning na předchozí text
        # (rychlost + méně smyček). without_timestamps=False: segmentové časy
        # MUSÍ zůstat pro merge s diarizací v pipeline.py.
        # align_words (FÁZE 4, default False): word_timestamps pro volitelný align.
        options = {
            "task": getattr(tr, "task", "transcribe"),
            "beam_size": int(getattr(tr, "beam_size", 1)),
            "condition_on_previous_text": bool(getattr(tr, "condition_on_previous_text", False)),
            "temperature": getattr(tr, "temperature", 0.0),
            "compression_ratio_threshold": getattr(tr, "compression_ratio_threshold", 2.4),
            # Pozor: faster-whisper chce log_PROB_threshold (openai-whisper měl logprob_).
            "log_prob_threshold": getattr(tr, "logprob_threshold", -1.0),
            "no_speech_threshold": getattr(tr, "no_speech_threshold", 0.6),
            "without_timestamps": False,
            "word_timestamps": bool(getattr(asr, "align_words", False)) if asr else False,
            "vad_filter": bool(getattr(tr, "vad_filter", True)),
            "vad_parameters": self._vad_parameters(),
        }

        # Add language if specified ("" / None = auto-detect).
        if self.config.model.language:
            options["language"] = self.config.model.language

        return options

    def _segments_to_dict(self, segments, info) -> Dict:
        """Převeď faster-whisper generátor na formát pipeline:
        {"text": ..., "segments": [{start, end, text}], "language": ...}."""
        seg_list = [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in segments
        ]
        language = getattr(info, "language", "") or ""
        return {
            "text": " ".join(s["text"] for s in seg_list),
            "segments": seg_list,
            "language": language,
        }

    def _transcribe_one(self, audio_path: str, options: Dict) -> Dict:
        t0 = time.perf_counter()
        # faster-whisper transcribe() je líné: inference běží až při iteraci
        # generátoru, proto list() uvnitř měření + no_grad + VRAM log.
        with torch.no_grad():
            segments_gen, info = self.model.transcribe(audio_path, **options)
            result = self._segments_to_dict(segments_gen, info)
        dt = time.perf_counter() - t0
        try:
            import soundfile as sf
            dur = sf.info(audio_path).duration
            rtf = dur / dt if dt > 0 else 0.0
        except Exception:
            dur, rtf = -1.0, -1.0
        self.logger.info(
            f"Chunk done: {os.path.basename(audio_path)} "
            f"audio={dur:.1f}s infer={dt:.1f}s RTFx={rtf:.2f} "
            f"segments={len(result['segments'])} lang={result['language']}"
        )
        return result

    def transcribe(self, audio_path: str) -> Dict:
        try:
            if self.model is None:
                self.initialize_model()

            # Verify file exists and is accessible
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            abs_audio_path = os.path.abspath(audio_path)
            self.logger.info(f"Transcribing file: {abs_audio_path}")

            options = self._transcribe_options()

            # Log transcription options
            self.logger.debug(f"Transcription options: {options}")

            # Create an instance of AudioProcessor
            from src.audio.processor import AudioProcessor
            audio_processor = AudioProcessor(self.config)

            # Split the audio file into chunks if necessary
            chunk_files = audio_processor.split_audio_file(abs_audio_path)

            if len(chunk_files) == 1:
                # Perform transcription on the single file
                self.logger.info("Starting transcription...")
                result = self._transcribe_one(chunk_files[0], options)
            else:
                # Process each chunk and merge results (model držen 1x, ne v loopu reload)
                self.logger.info(f"Processing {len(chunk_files)} audio chunks...")
                chunk_results = []

                for i, chunk_file in enumerate(chunk_files):
                    self.logger.info(f"Transcribing chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_file)}")
                    chunk_result = self._transcribe_one(chunk_file, options)
                    chunk_results.append(chunk_result)

                # Merge results from all chunks
                self.logger.info("Merging results from all chunks...")
                result = audio_processor.merge_transcription_results(chunk_results)

            # Uvolni CTranslate2/CUDA cache po dávce (FÁZE 1/4 VRAM management).
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self._log_vram("after-transcribe")

            # Clean up temporary files
            audio_processor.cleanup()

            self.logger.info(f"Successfully transcribed audio file: {abs_audio_path}")
            return result

        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}", exc_info=True)
            raise
