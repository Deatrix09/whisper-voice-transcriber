from abc import ABC, abstractmethod
from typing import Dict, List
import re
from src.utils.logger import setup_logger

# "speaker_3" / "SPEAKER3" / "SPEAKER_03" -> SPEAKER_03 ( formát pyannote classes() )
_SPEAKER_RE = re.compile(r"^speaker_?(\d+)$", re.IGNORECASE)


class BaseDiarizer(ABC):
    """Base class for diarization implementations"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file,
                                     file_level=getattr(config.logging, "file_level", None),
                                     console_level=getattr(config.logging, "console_level", None))
    
    @abstractmethod
    def initialize(self):
        """Initialize the diarization pipeline"""
        pass
    
    @abstractmethod
    def process(self, audio_path: str) -> List[Dict[str, any]]:
        """Process audio file and return diarization results
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of diarization segments with start, end, and speaker information
        """
        pass
    
    @staticmethod
    def normalize_speaker_label(label) -> str:
        """Sjednoť label napříč enginy na SPEAKER_XX (pyannote formát).

        "speaker_0" -> "SPEAKER_00", "SPEAKER_1" -> "SPEAKER_01",
        "SPEAKER_00" beze změny, ostatní -> upper().
        """
        text = str(label).strip()
        m = _SPEAKER_RE.match(text)
        if m:
            return f"SPEAKER_{int(m.group(1)):02d}"
        return text.upper()

    @staticmethod
    def resolve_overlaps(segments: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Odstraň překryvy: pozdější segment se ořízne o předchozí.

        Zachovává sjednocení časů i původní pořadí (setříděno podle startu,
        stabilně). Prázdné/nevalidní segmenty po ořezu vypadnou.
        Idempotentní: bez překryvů vrátí stejný obsah.
        """
        ordered = sorted(segments, key=lambda s: (s["start"], s["end"]))
        resolved: List[Dict[str, any]] = []
        for seg in ordered:
            seg = dict(seg)
            if resolved and seg["start"] < resolved[-1]["end"]:
                seg["start"] = resolved[-1]["end"]
            if seg["end"] <= seg["start"]:
                continue
            seg["duration"] = seg["end"] - seg["start"]
            resolved.append(seg)
        return resolved

    @staticmethod
    def _ensure_wav_format(audio_path: str) -> str:
        """Ensure audio is in WAV format for processing
        
        Args:
            audio_path: Path to the input audio file
            
        Returns:
            Path to the WAV format audio file
        """
        pass
