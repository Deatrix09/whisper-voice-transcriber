import whisper
import torch
from typing import Dict, List
from src.utils.logger import setup_logger
import os
import subprocess
import shutil

class Transcriber:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file)
        self.model = None
        
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

    def initialize_model(self):
        """Initialize the Whisper model."""
        try:
            # Check ffmpeg first
            if not self.check_ffmpeg():
                raise RuntimeError("ffmpeg is required but not found. Please install ffmpeg and add it to your PATH.")
                
            # Set device
            device = "cuda" if torch.cuda.is_available() and self.config.performance.device == "cuda" else "cpu"
            
            # Load model
            self.model = whisper.load_model(
                self.config.model.name,
                device=device
            )
            
            self.logger.info(f"Successfully loaded Whisper model: {self.config.model.name}")
            self.logger.info(f"Model device: {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper model: {str(e)}")
            raise

    def transcribe(self, audio_path: str) -> Dict:
        """Transcribe audio file using Whisper."""
        try:
            if self.model is None:
                self.initialize_model()
            
            # Verify file exists and is accessible
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            abs_audio_path = os.path.abspath(audio_path)
            self.logger.info(f"Transcribing file: {abs_audio_path}")
                
            # Prepare transcription options
            options = {
                "task": self.config.transcription.task,
                "temperature": self.config.transcription.temperature,
                "compression_ratio_threshold": self.config.transcription.compression_ratio_threshold,
                "logprob_threshold": self.config.transcription.logprob_threshold,
                "no_speech_threshold": self.config.transcription.no_speech_threshold,
                "fp16": self.config.transcription.fp16 if torch.cuda.is_available() else False,
                "verbose": self.config.transcription.verbose,
            }
            
            # Add language if specified
            if self.config.model.language:
                options["language"] = self.config.model.language
                
            # Log transcription options
            self.logger.debug(f"Transcription options: {options}")
                
            # Perform transcription
            self.logger.info("Starting transcription...")
            result = self.model.transcribe(
                abs_audio_path,
                **options
            )
            
            self.logger.info(f"Successfully transcribed audio file: {abs_audio_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}", exc_info=True)
            raise 