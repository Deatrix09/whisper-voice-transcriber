from pyannote.audio import Pipeline
import os
from pydub import AudioSegment
from typing import Dict, List, Tuple
import torch
from src.utils.logger import setup_logger
from torch.serialization import add_safe_globals
from torch.torch_version import TorchVersion
from pyannote.audio.core.task import Specifications, Problem, Resolution

class Diarizer:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file)
        self.pipeline = None
        
    def initialize_pipeline(self):
        try:
            self.logger.info("Initializing diarization pipeline...")
            
            # Register safe globals for PyTorch 2.6+ compatibility
            add_safe_globals([TorchVersion, Specifications, Problem, Resolution])
            self.logger.info("Registered safe globals for PyTorch compatibility")
            
            self.pipeline = Pipeline.from_pretrained(
                self.config.diarization.model,
                use_auth_token=self.config.diarization.auth_token
            )
            
            self.logger.info("Setting pipeline parameters:")
            self.logger.info(f"- Min duration off: {self.config.diarization.min_duration_off}")
            self.logger.info(f"- Min cluster size: {self.config.diarization.min_cluster_size}")
            self.logger.info(f"- Clustering method: {self.config.diarization.method}")
            
            # Set pipeline parameters
            self.pipeline.instantiate({
                "segmentation": {
                    "min_duration_off": self.config.diarization.min_duration_off
                },
                "clustering": {
                    "method": self.config.diarization.method,
                    "min_cluster_size": self.config.diarization.min_cluster_size
                }
            })
            
            if torch.cuda.is_available() and self.config.performance.device == "cuda":
                self.pipeline = self.pipeline.to(torch.device("cuda"))
                self.logger.info("Diarization pipeline moved to CUDA")
            else:
                self.logger.info("Diarization pipeline running on CPU")
                
            self.logger.info("Successfully initialized diarization pipeline")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize diarization pipeline: {str(e)}")
            raise

    def process(self, audio_path: str) -> List[Dict[str, any]]:
        """Process audio file and return diarization results."""
        try:
            if self.pipeline is None:
                self.initialize_pipeline()
                
            self.logger.info(f"Starting diarization for: {audio_path}")
            
            # Ensure audio file is in WAV format to prevent tensor size mismatch errors
            wav_audio_path = self._ensure_wav_format(audio_path)
            
            # Run diarization
            self.logger.info("Running diarization pipeline...")
            
            # Check if num_speakers is specified in config
            if hasattr(self.config.diarization, 'num_speakers') and self.config.diarization.num_speakers is not None:
                self.logger.info(f"Using specified number of speakers: {self.config.diarization.num_speakers}")
                diarization = self.pipeline(wav_audio_path, num_speakers=self.config.diarization.num_speakers)
            else:
                self.logger.info("Using automatic speaker detection")
                diarization = self.pipeline(wav_audio_path)
            
            # Clean up temporary file if needed
            if wav_audio_path != audio_path:
                self._cleanup_temp_file(wav_audio_path)
            
            results = []
            speaker_segments = {}
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = {
                    "speaker": speaker,
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
                
            self.logger.info(f"Successfully completed diarization for {audio_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Diarization failed: {str(e)}")
            raise 

    def _ensure_wav_format(self, audio_path: str) -> str:
        """
        Ensure the audio file is in WAV format with consistent properties.
        This prevents tensor size mismatch errors in the diarization pipeline.
        
        Args:
            audio_path: Path to the input audio file
            
        Returns:
            Path to the WAV format audio file (either the original or a converted temporary file)
        """

        
        # If already a WAV file, we still convert it for consistency
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
        """Clean up temporary file if it exists"""
        import os
        try:
            if os.path.exists(temp_file_path) and '_temp.wav' in temp_file_path:
                os.remove(temp_file_path)
                self.logger.info(f"Cleaned up temporary file: {temp_file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temporary file: {str(e)}")