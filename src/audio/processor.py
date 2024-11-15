import torch
import torchaudio
from typing import Tuple
import numpy as np
from src.utils.logger import setup_logger

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file)
        
    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio file."""
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.config.whisper.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, 
                    self.config.whisper.sample_rate
                )
                waveform = resampler(waveform)
            
            self.logger.info(f"Successfully loaded audio file: {file_path}")
            return waveform, self.config.whisper.sample_rate
            
        except Exception as e:
            self.logger.error(f"Error loading audio file: {str(e)}")
            raise 