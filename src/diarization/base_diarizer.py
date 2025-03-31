from abc import ABC, abstractmethod
from typing import Dict, List
from src.utils.logger import setup_logger

class BaseDiarizer(ABC):
    """Base class for diarization implementations"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file)
    
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
    def _ensure_wav_format(audio_path: str) -> str:
        """Ensure audio is in WAV format for processing
        
        Args:
            audio_path: Path to the input audio file
            
        Returns:
            Path to the WAV format audio file
        """
        pass
