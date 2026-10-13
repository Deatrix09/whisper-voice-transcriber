import torch
from typing import Tuple, Optional, List, Dict
import numpy as np
from src.utils.logger import setup_logger
import os
import tempfile
import time

# Těžké audio závislosti lazy (torchaudio/librosa/noisereduce/pydub),
# aby import modulu fungoval i bez nich a cold start byl rychlejší.
# Viz audit L09.

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file,
                                     file_level=getattr(config.logging, "file_level", None),
                                     console_level=getattr(config.logging, "console_level", None))
        self.converted_file_path: Optional[str] = None
        self.temp_files: List[str] = []
        
    def reduce_noise(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        try:
            import noisereduce as nr
            self.logger.info("Applying noise reduction...")

            # Convert torch tensor to numpy array for noise reduction
            # .detach().cpu() aby to fungovalo i pro CUDA / grad tensory.
            with torch.no_grad():
                audio_np = waveform.detach().cpu().numpy().squeeze()
            
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
            with torch.no_grad():
                return torch.nn.functional.normalize(waveform.detach(), dim=1)
        except Exception as e:
            self.logger.warning(f"Audio normalization failed: {str(e)}")
            return waveform

    def convert_m4a_to_wav(self, input_path: str) -> str:
        try:
            from pydub import AudioSegment
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

    def load_audio(
        self, file_path: str, load_waveform: bool = True
    ) -> Tuple[torch.Tensor, int, Optional[str]]:
        """Načti audio. load_waveform=False jen zajistí konverzi m4a->wav
        a nealokuje ~460MB tensor (pipeline waveform nepotřebuje, L01)."""
        try:
            converted_path = None
            if file_path.lower().endswith('.m4a'):
                converted_path = self.convert_m4a_to_wav(file_path)
                file_path = converted_path

            if not load_waveform:
                self.logger.info(
                    f"Skipping waveform decode (load_waveform=False): {file_path}"
                )
                return torch.empty(0), self.config.whisper.sample_rate, converted_path

            import torchaudio
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
        Streamuje přes PyAV (16kHz mono) místo načtení celého souboru do RAM,
        každý 30s chunk projde Silero VAD prefiltrem (tiché chunky = placeholder).
        Vrací List[str] temp wav chunků; časy řeší merge_transcription_results.
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

            chunk_size = int(getattr(self.config.audio, "chunk_size", 30))
            chunk_overlap = int(getattr(self.config.audio, "chunk_overlap_sec", 2))
            self.logger.info(f"Using chunk size: {chunk_size} seconds, overlap: {chunk_overlap} seconds")

            # ZÁKAZ krátkých chunků: 10s chunky prokazatelně zvedají WER (+3.5%).
            if chunk_size <= 10:
                raise ValueError(
                    f"chunk_size={chunk_size}s je zakázán: chunky <=10s zvedají WER "
                    f"(+3.5% u 10s). Použij >=30s."
                )
            if chunk_size < 30:
                self.logger.warning(
                    f"chunk_size={chunk_size}s < 30s: kratší kontext může zhoršit přesnost"
                )
            if chunk_overlap < 0 or chunk_overlap >= chunk_size:
                raise ValueError(
                    f"chunk_overlap_sec={chunk_overlap} musí být v [0, chunk_size)"
                )

            try:
                import av  # noqa: F401 - PyAV je tranzitivní závislost faster-whisper
                return self._split_streaming(file_path, chunk_size, chunk_overlap)
            except ImportError as e:
                self.logger.warning(
                    f"PyAV/Silero nedostupné ({e}), fallback na pydub (full-RAM)"
                )
                return self._split_via_pydub(file_path, chunk_size, chunk_overlap)

        except Exception as e:
            self.logger.error(f"Error splitting audio file: {str(e)}")
            raise

    def _vad_speech_ratio(self, audio_f32: np.ndarray, sample_rate: int) -> Tuple[float, float]:
        """Silero VAD v6 (bundlovaný ve faster-whisper) -> (speech_ratio, speech_sec).

        Při chybějícím faster-whisper vrací (1.0, délka) = pass-through, chunk se
        normálně přepošle Whisperu (ten má vlastní vad_filter).
        """
        duration = len(audio_f32) / sample_rate if sample_rate else 0.0
        try:
            from faster_whisper.vad import get_speech_timestamps, VadOptions
        except ImportError as e:
            self.logger.warning(f"Silero VAD nedostupné ({e}), VAD prefiltr přeskočen")
            return 1.0, duration
        opts = VadOptions(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        )
        stamps = get_speech_timestamps(audio_f32, vad_options=opts, sampling_rate=sample_rate)
        speech_sec = sum(s["end"] - s["start"] for s in stamps) / sample_rate
        ratio = (speech_sec / duration) if duration > 0 else 0.0
        return ratio, speech_sec

    def _write_chunk(
        self, window_i16: np.ndarray, temp_dir: str, idx: int, sample_rate: int,
        min_speech_sec: float = 0.5,
    ) -> str:
        """Zapiš 1 chunk + VAD prefiltr + per-chunk timing log.

        Tichý chunk (<min_speech_sec řeči) -> 0.5s silence placeholder:
        indexová matematika merge (offset = idx*step) zůstává, Whisper jím
        proletí a vad_filter vrátí prázdné segmenty.
        """
        import soundfile as sf
        t0 = time.perf_counter()
        chunk_path = os.path.join(temp_dir, f"chunk_{idx}.wav")

        audio_f32 = window_i16.astype(np.float32) / 32768.0
        ratio, speech_sec = self._vad_speech_ratio(audio_f32, sample_rate)

        if speech_sec < min_speech_sec:
            placeholder = np.zeros(sample_rate // 2, dtype=np.int16)  # 0.5s ticha
            sf.write(chunk_path, placeholder, sample_rate, subtype="PCM_16", format="WAV")
            self.temp_files.append(chunk_path)
            self.logger.info(
                f"Chunk {idx}: SILENT (speech={speech_sec:.2f}s) -> placeholder, "
                f"vad={time.perf_counter()-t0:.2f}s"
            )
            return chunk_path

        sf.write(chunk_path, window_i16, sample_rate, subtype="PCM_16", format="WAV")
        self.temp_files.append(chunk_path)
        self.logger.info(
            f"Chunk {idx}: {len(window_i16)/sample_rate:.1f}s audio, "
            f"speech={speech_sec:.1f}s ({ratio*100:.0f}%), "
            f"decode+vad+write={time.perf_counter()-t0:.2f}s"
        )
        return chunk_path

    def _split_streaming(self, file_path: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Streamuj decode přes PyAV, peak RAM ~1 chunk (ne celý soubor).

        Okna: start_idx = idx*(chunk-overlap), délka chunk; překryv se nese
        v `carry`. Výstupní časová matematika == merge_transcription_results
        (offset idx*chunk - overlap pro idx>0).
        """
        import av
        t_all = time.perf_counter()
        sample_rate = self.config.whisper.sample_rate
        chunk_samples = chunk_size * sample_rate
        overlap_samples = chunk_overlap * sample_rate
        step_samples = chunk_samples - overlap_samples

        temp_dir = tempfile.mkdtemp(prefix="wvt_chunks_")
        try:
            container = av.open(file_path)
        except Exception as e:
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass
            raise

        try:
            audio_stream = next(s for s in container.streams if s.type == "audio")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)

            buf = bytearray()  # jediný akumulátor int16 mono samplů (~1 chunk)
            chunk_files: List[str] = []
            idx = 0

            for frame in container.decode(audio_stream):
                for rframe in resampler.resample(frame):
                    arr = rframe.to_ndarray()  # (1, N) int16
                    if arr.shape[0] > 1:
                        arr = arr.mean(axis=0, keepdims=True).astype(np.int16)
                    buf += arr.tobytes()
                    while len(buf) >= chunk_samples * 2:
                        window = np.frombuffer(buf[:chunk_samples * 2], dtype=np.int16).copy()
                        chunk_files.append(self._write_chunk(window, temp_dir, idx, sample_rate))
                        idx += 1
                        del buf[:step_samples * 2]  # nech překryv pro další okno

            # Zbytek (>=0.5s) jako poslední částečný chunk (jako dřív ceil).
            if len(buf) >= sample_rate // 2 * 2:
                window = np.frombuffer(bytes(buf), dtype=np.int16).copy()
                chunk_files.append(self._write_chunk(window, temp_dir, idx, sample_rate))
                idx += 1

            if not chunk_files:
                self.logger.info("Audio file doesn't need splitting")
                return [file_path]

            self.logger.info(
                f"Successfully split audio into {len(chunk_files)} streaming chunks "
                f"({chunk_size}s/{chunk_overlap}s) in {time.perf_counter()-t_all:.1f}s"
            )
            return chunk_files
        finally:
            try:
                container.close()
            except Exception:
                pass

    def _split_via_pydub(self, file_path: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Fallback bez PyAV: původní full-RAM cesta (pomalá, žere paměť)."""
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)

        # Get duration in seconds
        duration = len(audio) / 1000
        self.logger.info(f"Audio duration: {duration:.2f} seconds")

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