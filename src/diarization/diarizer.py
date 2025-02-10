from pyannote.audio import Pipeline
from typing import Dict, List, Tuple
import torch
from src.utils.logger import setup_logger

class Diarizer:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file)
        self.pipeline = None
        
    def initialize_pipeline(self):
        """Initialize the pyannote pipeline with authentication."""
        try:
            self.logger.info("Initializing diarization pipeline...")
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
            
            # Run diarization
            self.logger.info("Running diarization pipeline...")
            
            # Check if num_speakers is specified in config
            if hasattr(self.config.diarization, 'num_speakers') and self.config.diarization.num_speakers is not None:
                self.logger.info(f"Using specified number of speakers: {self.config.diarization.num_speakers}")
                diarization = self.pipeline(audio_path, num_speakers=self.config.diarization.num_speakers)
            else:
                self.logger.info("Using automatic speaker detection")
                diarization = self.pipeline(audio_path)
            
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