import torch
import torchaudio
from typing import Tuple, Optional, List, Dict
import numpy as np
from src.utils.logger import setup_logger
import librosa
import noisereduce as nr
from pydub import AudioSegment
import os
import tempfile

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file)
        self.converted_file_path: Optional[str] = None
        self.temp_files: List[str] = []
        
    def reduce_noise(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
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

    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int, Optional[str]]:
        try:
            converted_path = None
            if file_path.lower().endswith('.m4a'):
                converted_path = self.convert_m4a_to_wav(file_path)
                file_path = converted_path
            
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
            return waveform, self.config.whisper.sample_rate, converted_path
            
        except Exception as e:
            self.logger.error(f"Error loading audio file: {str(e)}")
            raise
    
    def split_audio_file(self, file_path: str) -> List[str]:
        """
        Split a large audio file into smaller chunks for better processing.
        Returns a list of paths to the temporary chunked files.
        """
        try:
            self.logger.info(f"Splitting audio file: {file_path}")
            
            # Get file size in MB
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # Get thresholds from config or use defaults
            threshold_mb = self.config.audio.chunk_threshold_mb
            self.logger.info(f"Using chunk threshold: {threshold_mb} MB")
            
            # If file is smaller than threshold, don't split
            if file_size_mb < threshold_mb:
                self.logger.info(f"Audio file is small enough ({file_size_mb:.2f} MB), skipping split")
                return [file_path]
            
            # Load the audio file
            if file_path.lower().endswith('.m4a'):
                file_path = self.convert_m4a_to_wav(file_path)
                
            audio = AudioSegment.from_file(file_path)
            
            # Get duration in seconds
            duration = len(audio) / 1000
            self.logger.info(f"Audio duration: {duration:.2f} seconds")
            
            # Get chunk size from config (in seconds)
            chunk_size = self.config.audio.chunk_size
            chunk_overlap = self.config.audio.chunk_overlap_sec
            self.logger.info(f"Using chunk size: {chunk_size} seconds, overlap: {chunk_overlap} seconds")
            
            # Calculate number of chunks
            num_chunks = int(np.ceil(duration / chunk_size))
            
            # If only one chunk is needed, return original file
            if num_chunks <= 1:
                self.logger.info("Audio file doesn't need splitting")
                return [file_path]
                
            self.logger.info(f"Splitting audio into {num_chunks} chunks of approximately {chunk_size/60:.1f} minutes")
            
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            
            # Split audio into chunks
            chunk_files = []
            
            for i in range(num_chunks):
                start_ms = i * chunk_size * 1000
                end_ms = min((i + 1) * chunk_size * 1000, len(audio))
                
                # Add overlap for better continuity
                if i > 0:
                    start_ms -= chunk_overlap * 1000  # Convert overlap seconds to milliseconds
                
                chunk = audio[start_ms:end_ms]
                
                # Save chunk to temp file
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                chunk.export(chunk_path, format="wav", parameters=[
                    "-ac", "1",
                    "-ar", str(self.config.whisper.sample_rate),
                    "-acodec", "pcm_s16le"
                ])
                
                chunk_files.append(chunk_path)
                self.temp_files.append(chunk_path)
                
            self.logger.info(f"Successfully split audio into {len(chunk_files)} chunks")
            return chunk_files
            
        except Exception as e:
            self.logger.error(f"Error splitting audio file: {str(e)}")
            raise
    
    def merge_transcription_results(self, results_list: List[Dict]) -> Dict:
        """
        Merge transcription results from multiple chunks into a single result.
        Handles overlapping segments and adjusts timestamps.
        """
        try:
            self.logger.info(f"Merging transcription results from {len(results_list)} chunks")
            
            if len(results_list) == 1:
                return results_list[0]
                
            # Initialize result structure
            merged_result = {
                "text": "",
                "segments": [],
                "language": results_list[0].get("language", "")
            }
            
            # Offset for each chunk (in seconds)
            chunk_size = self.config.audio.chunk_size
            chunk_overlap = getattr(self.config.audio, 'chunk_overlap_sec', 2)
            
            # Keep track of last segment's text from previous chunk for duplicate detection
            last_segments_texts = []
            
            # Process each chunk's results
            for i, chunk_result in enumerate(results_list):
                # Calculate time offset for this chunk
                time_offset = i * chunk_size
                
                # Subtract overlap for all chunks except the first
                if i > 0:
                    time_offset -= chunk_overlap
                
                # Get segments from this chunk
                chunk_segments = chunk_result.get("segments", [])
                
                # Process segments
                for segment in chunk_segments:
                    # Adjust timestamps
                    adjusted_segment = segment.copy()
                    adjusted_segment["start"] += time_offset
                    adjusted_segment["end"] += time_offset
                    
                    # Skip segments that are completely in the overlap region (only for chunks after the first)
                    if i > 0 and adjusted_segment["end"] <= time_offset + chunk_overlap:
                        continue
                    
                    # Check for duplicate text content (from overlap regions)
                    segment_text = adjusted_segment["text"].strip()
                    is_duplicate = False
                    
                    # Check against recent segments to detect duplicates
                    for prev_text in last_segments_texts:
                        if self._is_similar_text(segment_text, prev_text):
                            is_duplicate = True
                            self.logger.debug(f"Skipping duplicate segment: '{segment_text}'")
                            break
                    
                    # If not a duplicate, add to merged result
                    if not is_duplicate:
                        merged_result["segments"].append(adjusted_segment)
                        # Keep track of this segment's text for duplicate detection
                        last_segments_texts.append(segment_text)
                        # Only keep last N segments for comparison to avoid performance issues
                        if len(last_segments_texts) > 5:
                            last_segments_texts.pop(0)
            
            # Sort segments by start time
            merged_result["segments"].sort(key=lambda x: x["start"])
            
            # Rebuild the complete transcript
            merged_result["text"] = " ".join(s["text"].strip() for s in merged_result["segments"])
            
            self.logger.info(f"Successfully merged results: {len(merged_result['segments'])} segments")
            return merged_result
            
        except Exception as e:
            self.logger.error(f"Error merging transcription results: {str(e)}")
            raise

    def _is_similar_text(self, text1, text2):
        """
        Check if two text segments are similar enough to be considered duplicates.
        Uses simple overlap-based detection.
        """
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

    def cleanup(self):
        # Clean up converted file
        if self.converted_file_path and os.path.exists(self.converted_file_path):
            try:
                os.remove(self.converted_file_path)
                self.logger.info(f"Cleaned up temporary file: {self.converted_file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary file: {str(e)}")
        
        # Clean up temp chunk files
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp chunk file: {str(e)}")
        
        # Clean up temp directory if it exists
        if self.temp_files and os.path.exists(os.path.dirname(self.temp_files[0])):
            try:
                os.rmdir(os.path.dirname(self.temp_files[0]))
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp directory: {str(e)}")
        
        self.temp_files = []