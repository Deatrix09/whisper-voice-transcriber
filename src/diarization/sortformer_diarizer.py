"""NeMo Sortformer diarizer (end-to-end, offline režim streaming modelu).

Model: nvidia/diar_streaming_sortformer_4spk-v2.1 (117M params).
LIMIT: max 4 speakeři (architektura: max_num_of_spks=4, fixed). Při detekci
více než 4 labelů se nic netají - zaloguje se warning (model víc neumí).
Vstup: 16kHz mono WAV. NeMo (`nemo_toolkit['asr']`) se importuje lazy až v
initialize(), bez něj jasná RuntimeError hláška.
"""
import os
from typing import Dict, List
import torch
from src.diarization.base_diarizer import BaseDiarizer


class SortformerDiarizer(BaseDiarizer):
    DEFAULT_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    MAX_SPEAKERS = 4

    def __init__(self, config):
        super().__init__(config)
        self.model = None

    def _model_id(self) -> str:
        section = getattr(self.config.diarization, "sortformer", None)
        configured = getattr(section, "model", None) if section is not None else None
        return configured or self.DEFAULT_MODEL

    def _batch_size(self) -> int:
        section = getattr(self.config.diarization, "sortformer", None)
        try:
            return int(getattr(section, "batch_size", 1) or 1)
        except (TypeError, ValueError):
            return 1

    def initialize(self):
        """Load Sortformer 1x (lazy singleton na instanci)."""
        if self.model is not None:
            return self.model
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as e:
            raise RuntimeError(
                "NeMo není nainstalováno (potřeba pro sortformer větev). "
                "Spusť: pip install \"nemo_toolkit['asr']\" (vyžaduje Linux + CUDA "
                "pro plný výkon; na Windows/CPU nemusí fungovat)."
            ) from e
        try:
            model_id = self._model_id()
            self.logger.info(f"Loading Sortformer model: {model_id} (max 4 speakers)")
            self.model = SortformerEncLabelModel.from_pretrained(model_id)
            self.model.eval()
            from src.utils.config import resolve_device
            try:
                eff_device = resolve_device(self.config.performance.device)
            except ValueError:
                eff_device = "cpu"
            if eff_device == "cuda":
                try:
                    self.model = self.model.to(torch.device("cuda"))
                    self.logger.info("Sortformer model moved to CUDA")
                except Exception as e:
                    self.logger.warning(f"Sortformer .to(cuda) selhalo ({e}), jedu na CPU")
            else:
                self.logger.info("Sortformer model running on CPU")
            self.logger.info("Successfully initialized Sortformer diarizer")
            return self.model
        except Exception as e:
            self.logger.error(f"Failed to initialize Sortformer diarizer: {str(e)}")
            raise

    def _normalize(self, raw) -> List[Dict[str, any]]:
        """Defenzivní převod výstupu diarize() na [{start, end, speaker}].

        Pokrývá: list dictů / list [start, end, label] / objekty se
        .start/.end + .speaker/.label + jednoúrovňové zanoření per-file.
        """
        if isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], (list, tuple)):
            inner = raw[0]
            if inner and isinstance(inner[0], (dict, list, tuple)):
                raw = inner
        segments: List[Dict[str, any]] = []
        for item in raw or []:
            if isinstance(item, dict):
                start = item.get("start")
                end = item.get("end")
                label = item.get("speaker", item.get("label", item.get("Speaker")))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                start, end, label = item[0], item[1], item[2]
            else:
                start = getattr(item, "start", None)
                end = getattr(item, "end", None)
                label = getattr(item, "speaker", getattr(item, "label", None))
            if start is None or end is None or label is None:
                self.logger.debug(f"Skipping uninterpretable Sortformer item: {item!r}")
                continue
            start, end = float(start), float(end)
            if end <= start:
                continue
            segments.append({
                "speaker": self.normalize_speaker_label(label),
                "start": start,
                "end": end,
                "duration": end - start,
            })
        return segments

    def _ensure_wav(self, audio_path: str) -> str:
        """Sortformer chce 16kHz mono WAV (viz NeMo/ExecuTorch docs)."""
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            if info.channels == 1 and info.samplerate == 16000:
                return audio_path
        except Exception as e:
            self.logger.warning(f"Could not read audio info: {str(e)}")
        from pydub import AudioSegment
        temp_wav_path = os.path.splitext(audio_path)[0] + "_sortformer_temp.wav"
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio.export(temp_wav_path, format="wav", parameters=[
            "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le"])
        self.logger.info(f"Created temporary WAV file: {temp_wav_path}")
        return temp_wav_path

    def process(self, audio_path: str) -> List[Dict[str, any]]:
        try:
            if self.model is None:
                self.initialize()
            self.logger.info(f"Starting Sortformer diarization for: {audio_path}")
            wav_audio_path = self._ensure_wav(audio_path)
            try:
                with torch.no_grad():
                    raw = self.model.diarize(
                        audio=[wav_audio_path], batch_size=self._batch_size())
            finally:
                if wav_audio_path != audio_path and os.path.exists(wav_audio_path) \
                        and "_sortformer_temp.wav" in wav_audio_path:
                    try:
                        os.remove(wav_audio_path)
                    except OSError as e:
                        self.logger.warning(f"Failed to clean up temp file: {str(e)}")

            segments = self._normalize(raw)

            n_speakers = len({s["speaker"] for s in segments})
            if n_speakers > self.MAX_SPEAKERS:
                self.logger.warning(
                    f"Detected {n_speakers} speakers > model limit {self.MAX_SPEAKERS} "
                    f"({self._model_id()}). Výstup může slučovat mluvčí.")
            segments = self.resolve_overlaps(segments)
            self.logger.info(
                f"Sortformer done: {len(segments)} segments, {n_speakers} speakers")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return segments
        except Exception as e:
            self.logger.error(f"Sortformer diarization failed: {str(e)}")
            raise
