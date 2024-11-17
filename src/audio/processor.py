import torch
import torchaudio
from typing import Tuple, Optional
import numpy as np
from src.utils.logger import setup_logger
import librosa
import noisereduce as nr
from pydub import AudioSegment
import os

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file)
        self.converted_file_path: Optional[str] = None
        
    def reduce_noise(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply noise reduction to the audio signal."""
        try:
            self.logger.info("Applying noise reduction...")
            
            # Convert torch tensor to numpy array for noise reduction
            audio_np = waveform.numpy().squeeze()
            
            # Apply noise reduction
            reduced_noise = nr.reduce_noise(
                y=audio_np,
                sr=sample_rate,
                stationary=True,
                prop_decrease=0.75
            )
            
            # Convert back to torch tensor
            return torch.from_numpy(reduced_noise).unsqueeze(0)
            
        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {str(e)}")
            return waveform

    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio to a consistent volume level."""
        try:
            self.logger.info("Normalizing audio levels...")
            return torch.nn.functional.normalize(waveform, dim=1)
        except Exception as e:
            self.logger.warning(f"Audio normalization failed: {str(e)}")
            return waveform

    def convert_m4a_to_wav(self, input_path: str) -> str:
        try:
            self.logger.info(f"Converting M4A to WAV: {input_path}")
            output_path = os.path.splitext(input_path)[0] + '.wav'
            
            audio = AudioSegment.from_file(input_path, format="m4a")
            
            # Convert to mono and set sample rate
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(self.config.whisper.sample_rate)
            
            audio.export(output_path, format="wav", parameters=[
                "-ac", "1",
                "-ar", str(self.config.whisper.sample_rate),
                "-acodec", "pcm_s16le"
            ])
            
            self.converted_file_path = output_path
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to convert M4A to WAV: {str(e)}")
            raise

    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio file with enhanced processing."""
        try:
            if file_path.lower().endswith('.m4a'):
                file_path = self.convert_m4a_to_wav(file_path)
            
            # Load audio file
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            waveform = self.reduce_noise(waveform, sample_rate)
            
            waveform = self.normalize_audio(waveform)
            
            # Resample if necessary
            if sample_rate != self.config.whisper.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, 
                    self.config.whisper.sample_rate
                )
                waveform = resampler(waveform)
            
            self.logger.info(f"Successfully loaded and preprocessed audio file: {file_path}")
            return waveform, self.config.whisper.sample_rate
            
        except Exception as e:
            self.logger.error(f"Error loading audio file: {str(e)}")
            raise
            
    def cleanup(self):
        """Clean up any temporary converted files."""
        if self.converted_file_path and os.path.exists(self.converted_file_path):
            try:
                os.remove(self.converted_file_path)
                self.logger.info(f"Cleaned up temporary file: {self.converted_file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary file: {str(e)}")