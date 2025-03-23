from typing import Dict, List
import os
from src.audio.processor import AudioProcessor
from src.diarization.diarizer import Diarizer
from src.transcription.transcriber import Transcriber
from src.utils.logger import setup_logger
import torch

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file)
        
        self.logger.info("Initializing pipeline components...")
        self.audio_processor = AudioProcessor(config)
        self.diarizer = Diarizer(config)
        self.transcriber = Transcriber(config)
        
    def merge_diarization_with_transcription(
        self, 
        diarization_results: List[Dict], 
        transcription_result: Dict
    ) -> List[Dict]:
        """Merge diarization and transcription results."""
        try:
            self.logger.info("Starting to merge diarization and transcription results...")
            self.logger.info(f"Diarization segments: {len(diarization_results)}")
            self.logger.info(f"Transcription segments: {len(transcription_result['segments'])}")
            
            merged_results = []
            unmatched_segments = 0
            multiple_speakers = 0
            
            # Process segments in sequential order
            prev_segment_text = None
            
            for segment in transcription_result["segments"]:
                segment_start = segment["start"]
                segment_end = segment["end"]
                segment_text = segment["text"].strip()
                
                # Skip near-duplicate segments
                if prev_segment_text and self._is_similar_text(segment_text, prev_segment_text):
                    self.logger.debug(f"Skipping duplicate segment text: '{segment_text}'")
                    continue
                
                prev_segment_text = segment_text
                
                # Find matching speakers
                matching_speakers = [
                    d["speaker"] for d in diarization_results
                    if max(segment_start, d["start"]) < min(segment_end, d["end"])
                ]
                
                if not matching_speakers:
                    unmatched_segments += 1
                    speaker = "UNKNOWN"
                elif len(matching_speakers) > 1:
                    multiple_speakers += 1
                    # Use the speaker with the longest overlap
                    speaker = max(
                        matching_speakers,
                        key=lambda s: sum(
                            min(segment_end, d["end"]) - max(segment_start, d["start"])
                            for d in diarization_results
                            if d["speaker"] == s
                        )
                    )
                else:
                    speaker = matching_speakers[0]
                
                # Skip empty or very short segments (less than 0.5 seconds with no useful content)
                if not segment_text or (segment_end - segment_start < 0.5 and len(segment_text.split()) <= 1):
                    continue
                    
                # Skip segments with repetitive content
                if self._is_repetitive_content(segment_text):
                    self.logger.debug(f"Skipping repetitive segment: '{segment_text}'")
                    continue
                    
                merged_results.append({
                    "start": segment_start,
                    "end": segment_end,
                    "speaker": speaker,
                    "text": segment_text,
                    "duration": segment_end - segment_start
                })
            
            # Log merge statistics
            self.logger.info("Merge Results Summary:")
            self.logger.info(f"Total merged segments: {len(merged_results)}")
            self.logger.info(f"Unmatched segments: {unmatched_segments}")
            self.logger.info(f"Multiple speaker segments: {multiple_speakers}")
            
            # Log speaker statistics
            speaker_stats = {}
            for segment in merged_results:
                speaker = segment["speaker"]
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {
                        "count": 0,
                        "total_duration": 0,
                        "total_words": 0
                    }
                speaker_stats[speaker]["count"] += 1
                speaker_stats[speaker]["total_duration"] += segment["duration"]
                speaker_stats[speaker]["total_words"] += len(segment["text"].split())
            
            for speaker, stats in speaker_stats.items():
                self.logger.info(f"\nSpeaker {speaker} statistics:")
                self.logger.info(f"  - Segments: {stats['count']}")
                self.logger.info(f"  - Total duration: {stats['total_duration']:.2f} seconds")
                self.logger.info(f"  - Total words: {stats['total_words']}")
                self.logger.info(f"  - Average words per segment: {stats['total_words']/stats['count']:.2f}")
            
            return merged_results
            
        except Exception as e:
            self.logger.error(f"Error merging results: {str(e)}")
            raise
            
    def _is_similar_text(self, text1: str, text2: str) -> bool:
        """Check if two text segments are similar enough to be considered duplicates."""
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # If either text is very short, use direct comparison
        if len(text1) < 10 or len(text2) < 10:
            return text1 == text2
            
        # If texts are identical, they're obviously duplicates
        if text1 == text2:
            return True
            
        # Check for significant overlap (more than 70% of shorter text)
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        overlap = words1.intersection(words2)
        shorter_len = min(len(words1), len(words2))
        
        if shorter_len == 0:
            return False
        
        overlap_ratio = len(overlap) / shorter_len
        return overlap_ratio > 0.7

    def _is_repetitive_content(self, text: str) -> bool:
        """Check if a segment contains repetitive content that should be filtered out."""
        text = text.lower().strip()
        
        # If text is very short, don't consider it repetitive
        if len(text) < 10:
            return False
            
        # Check for repeated phrases
        words = text.split()
        if len(words) <= 1:
            return False
            
        # Check if the same phrase is repeated multiple times
        if len(set(words)) == 1:
            return True
            
        # Check for repeating patterns
        # Example: "I am a native, I am a native, I am a native"
        for phrase_length in range(2, min(5, len(words) // 2 + 1)):
            for start in range(0, len(words) - phrase_length * 2 + 1):
                phrase1 = " ".join(words[start:start + phrase_length])
                phrase2 = " ".join(words[start + phrase_length:start + phrase_length * 2])
                if phrase1 == phrase2:
                    return True
                    
        return False

    def process_file(self, audio_path: str) -> List[Dict]:
        try:
            self.logger.info(f"Starting processing of {audio_path}")
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if not os.path.isfile(audio_path):
                raise ValueError(f"Path is not a file: {audio_path}")
            
            # Check file size
            file_size = os.path.getsize(audio_path)
            self.logger.info(f"Audio file size: {file_size / (1024*1024):.2f} MB")
            
            # Load and preprocess audio
            self.logger.info("Loading and preprocessing audio...")
            waveform, sample_rate, converted_path = self.audio_processor.load_audio(audio_path)
            
            # Use converted path if available
            if converted_path:
                self.logger.info(f"Using converted WAV file: {converted_path}")
                audio_path = converted_path
            
            # Log audio properties
            self.logger.info(f"Audio loaded: {waveform.shape}, Sample rate: {sample_rate}")
            
            if torch.cuda.is_available() and self.config.performance.device == "cuda":
                self.logger.info("CUDA is available and will be used")
            else:
                self.logger.warning("CUDA is not available, using CPU")
            
            # Perform diarization if enabled
            diarization_results = []
            if self.config.diarization.enabled:
                self.logger.info("Starting diarization...")
                diarization_results = self.diarizer.process(audio_path)
                self.logger.info(f"Diarization completed: {len(diarization_results)} segments")
            
            # Perform transcription
            self.logger.info("Starting transcription...")
            transcription_result = self.transcriber.transcribe(audio_path)
            self.logger.info("Transcription completed")
            
            # Merge results if diarization is enabled
            if self.config.diarization.enabled:
                self.logger.info("Merging diarization and transcription results...")
                final_results = self.merge_diarization_with_transcription(
                    diarization_results,
                    transcription_result
                )
            else:
                self.logger.info("Creating results without diarization...")
                final_results = [{
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": "UNKNOWN",
                    "text": segment["text"].strip()
                } for segment in transcription_result["segments"]]
            
            self.logger.info(f"Successfully processed {audio_path}")
            self.logger.info(f"Generated {len(final_results)} segments")
            
            return final_results
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found error: {str(e)}")
            raise
        except ValueError as e:
            self.logger.error(f"Value error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}", exc_info=True)
            raise