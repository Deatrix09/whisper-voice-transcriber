import os
from typing import Dict, List, Optional, Union, Tuple
from pydub import AudioSegment
import torch
import numpy as np
from src.diarization.base_diarizer import BaseDiarizer
from omegaconf import OmegaConf
import soundfile as sf

class NemoDiarizer(BaseDiarizer):
    def __init__(self, config):
        super().__init__(config)
        self.vad_model = None
        self.speaker_model = None
        self.msdd_model = None
        self.temp_files = []
    
    def initialize(self):
        """Initialize NeMo models for diarization"""
        try:
            import nemo.collections.asr as nemo_asr
            
            self.logger.info("Initializing NeMo diarization models...")
            
            # We don't need to pre-load models as they will be loaded by the NeuralDiarizer
            # when diarize() is called. This is the approach used in diarize_audio.py
            self.logger.info("NeMo models will be loaded on demand during diarization")
            
            # Check if CUDA is available
            if torch.cuda.is_available() and self.config.performance.device == "cuda":
                self.logger.info("CUDA is available and will be used for diarization")
            else:
                self.logger.info("CUDA is not available, using CPU for diarization")
            
            self.logger.info("Successfully initialized NeMo diarization")
            
        except ImportError as e:
            self.logger.error(f"Failed to import NeMo: {str(e)}. Is NeMo properly installed?")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize NeMo diarization: {str(e)}")
            raise
    

    def process(self, audio_path: str) -> List[Dict[str, any]]:
        """Process audio file using NeMo diarization pipeline exactly like diarize_audio.py"""
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf
        import tempfile
        
        self.logger.info(f"Processing audio file for diarization: {audio_path}")

        # L04: temp_dir se dřív nikdy nemazal (leak v %TEMP% při každém běhu).
        # Stopa pro finally-cleanup i při výjimce před vytvořením.
        temp_dir = None
        try:
            # Create a temporary directory for manifest files
            temp_dir = tempfile.mkdtemp(prefix="nemo_diar_")
            self.logger.info(f"Created temporary directory: {temp_dir}")
            
            # Preprocess audio file - exactly like in diarize_audio.py
            preprocessed_audio_file = self._preprocess_audio(audio_path, temp_dir)
            
            # Create manifest file - exactly like in diarize_audio.py
            manifest_filepath = self._create_diarize_manifest(preprocessed_audio_file, temp_dir)
            
            # Create output directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
            output_dir = os.path.join(project_root, 'data', 'output')
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Output directory: {output_dir}")
            
            # Detect available device
            device = 'cuda' if torch.cuda.is_available() and self.config.performance.device == "cuda" else 'cpu'
            self.logger.info(f"Using device: {device}")
            
            # Configuration structure exactly like diarize_audio.py
            nemo_diar_config = {
                'device': device,
                'batch_size': 1,
                'num_workers': 0,
                'sample_rate': 16000,
                'verbose': True,
                'diarizer': {
                    'manifest_filepath': manifest_filepath,
                    'out_dir': output_dir,
                    'paths2audio_files': [preprocessed_audio_file],
                    'num_speakers': self.config.diarization.num_speakers if hasattr(self.config.diarization, 'num_speakers') else None,
                    'oracle_vad': False,
                    'collar': 0.25,
                    'ignore_overlap': False,
                    'vad': {
                        'model_path': 'vad_multilingual_marblenet',
                        'parameters': {
                            'window_length_in_sec': 0.15,
                            'shift_length_in_sec': 0.01,
                            'overlap': 0.5,
                            'smoothing': 'median',
                            'window_size_for_smoothing': 3,
                            'threshold': 0.5,
                            'onset': 0.5,
                            'offset': 0.5,
                        }
                    },
                    'msdd_model': {
                        'model_path': 'diar_msdd_telephonic',
                        'parameters': {
                            'use_speaker_model_from_ckpt': True,
                            'diar_window_length': 50,
                            'diar_shift_length': 10,
                            'speaker_overlap_threshold': 0.5,
                            'infer_batch_size': 1,
                            'seq_eval_mode': True,
                            'diar_eval_settings': [(0.25, True), (0.25, False), (0.0, False)],
                            'max_num_of_spks': 20,
                            'overlap_infer_spk_limit': 10
                        }
                    },
                    'speaker_embeddings': {
                        'model_path': 'titanet_large',
                        'parameters': {
                            'window_length_in_sec': [1.5],
                            'shift_length_in_sec': [0.75],
                            'embed_batch_size': 64,
                            'device': device,
                            'multiscale_weights': [1.0],
                            'save_embeddings': False
                        }
                    },
                    'clustering': {
                        'method': 'cosine',
                        'parameters': {
                            'max_num_speakers': 20,
                            'sparse_search_volume': 30,
                            'enhanced_count_thres': 80,
                            'pval_thres': 0.02,
                            'max_rp_threshold': 0.15,
                            'oracle_num_speakers': False
                        }
                    }
                },
                'vad': {
                    'model_path': 'vad_multilingual_marblenet',
                    'parameters': {
                        'window_length_in_sec': 0.15,
                        'shift_length_in_sec': 0.01,
                        'smoothing': 'median',
                        'window_size_for_smoothing': 31,
                        'threshold': 0.4,
                        'onset': 0.4,
                        'offset': 0.3,
                        'pad_onset': 0.1,
                        'pad_offset': 0.2
                    }
                },
                'asr': {
                    'model_path': 'stt_en_conformer_ctc_large',
                    'parameters': {
                        # L04: False = NeMo nestahuje/neloaduje Conformer-CTC (~1GB)
                        # jen pro word-timestamp VAD refinement. Náš kód čte pouze
                        # speaker RTTM, VAD dál běží přes MarbleNet (diarizer.vad).
                        # Odpovídá standardní NeMo system-VAD cestě
                        # (offline_diar_infer.py); ASR větev je opt-in
                        # (offline_diar_with_asr_infer.py).
                        'asr_based_vad': False,
                        'word_ts_anchor_offset': 0.15,
                        'ctc_decoder_params': {
                            'beam_size': 16,
                            'hotwords': []
                        }
                    }
                },
                'parallel': {
                    'vad_parallel_workers': 4,
                    'embedding_parallel_batches': 8,
                    'cluster_parallel_threads': 2
                },
                'device_settings': {
                    'vad_device': 'cpu',
                    'embedding_device': device,
                    'force_cpu': False
                },
                'preprocessing': {
                    'resample': True,
                    'target_sample_rate': 16000,
                    'audio_channel': 'mix'
                },
                'validation': {
                    'check_audio_length': True,
                    'min_audio_length': 0.5,
                    'max_audio_length': 36000,
                    'sample_rate': 16000
                }
            }
            
            # Handle number of speakers if specified
            if hasattr(self.config.diarization, 'num_speakers') and self.config.diarization.num_speakers:
                nemo_diar_config['diarizer']['clustering']['parameters']['oracle_num_speakers'] = True
                nemo_diar_config['diarizer']['clustering']['parameters']['oracle_value'] = self.config.diarization.num_speakers
            
            # Convert to OmegaConf
            diarizer_config = OmegaConf.create(nemo_diar_config)
            
            # Initialize diarizer - exactly like in diarize_audio.py
            self.logger.info("Initializing NeMo Neural Diarizer...")
            self.logger.info("ASR-based VAD is disabled (MarbleNet VAD only, no Conformer download)")
            diarizer = None
            try:
                diarizer = nemo_asr.models.NeuralDiarizer(cfg=diarizer_config)
                # Run diarization
                self.logger.info("Running diarization...")
                diarizer.diarize()
                self.logger.info("Diarization completed successfully")
            except Exception as e:
                self.logger.error(f"Error during diarizer.diarize(): {str(e)}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # Continue with the rest of the function to try to use any output files that might have been created
            finally:
                # L04: VAD/Titanet/MSDD váhy pryč z VRAM hned po diarize(),
                # ne až na konci process() (souběh s Whisper na 8GB).
                if diarizer is not None:
                    del diarizer
                import gc
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            
            # Process results from RTTM file
            # NeMo creates RTTM files in the pred_rttms subdirectory with the processed filename
            base_filename = os.path.splitext(os.path.basename(preprocessed_audio_file))[0]
            audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
            
            self.logger.info(f"Base filename: {base_filename}")
            self.logger.info(f"Audio basename: {audio_basename}")
            
            # Check for RTTM files with various possible naming patterns
            possible_rttm_files = [
                # Most likely pattern based on observed files
                os.path.join(output_dir, 'pred_rttms', f"{audio_basename}_processed.rttm"),
                
                # Other possible patterns
                os.path.join(output_dir, 'pred_rttms', f"{base_filename}.rttm"),
                os.path.join(output_dir, 'pred_rttms', f"{base_filename}_processed.rttm"),
                os.path.join(output_dir, 'pred_rttms', f"{audio_basename}.rttm")
            ]
            
            # Also check for any RTTM files in the directory
            pred_rttms_dir = os.path.join(output_dir, 'pred_rttms')
            if os.path.exists(pred_rttms_dir):
                self.logger.info(f"Contents of {pred_rttms_dir}:")
                for file in os.listdir(pred_rttms_dir):
                    self.logger.info(f"  - {file}")
                    if file.endswith('.rttm'):
                        possible_rttm_files.append(os.path.join(pred_rttms_dir, file))
            
            rttm_file = None
            for possible_file in possible_rttm_files:
                self.logger.info(f"Checking for RTTM file at: {possible_file}")
                if os.path.exists(possible_file):
                    rttm_file = possible_file
                    self.logger.info(f"Found RTTM file: {rttm_file}")
                    break
            
            # If we found an RTTM file, parse it
            if rttm_file:
                try:
                    segments = []
                    with open(rttm_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 8:
                                start_time = float(parts[3])
                                duration = float(parts[4])
                                end_time = start_time + duration
                                speaker = parts[7]
                                
                                segments.append({
                                    "start": start_time,
                                    "end": end_time,
                                    "speaker": speaker.upper(),  # Convert to uppercase to match format
                                    "duration": duration
                                })
                    
                    # If we found segments in the RTTM file, consolidate them and return
                    if segments:
                        self.logger.info(f"Parsed {len(segments)} segments from RTTM file")
                        consolidated_segments = self._consolidate_segments(segments)
                        self.logger.info(f"Consolidated to {len(consolidated_segments)} segments")
                        return consolidated_segments
                except Exception as e:
                    self.logger.error(f"Error parsing RTTM file: {str(e)}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # If we didn't find an RTTM file or it had no segments, check the speaker_outputs directory
            self.logger.warning("No valid RTTM file found or it contained no segments")
            
            # Try to use the cluster label file as a backup
            cluster_label_file = os.path.join(output_dir, 'speaker_outputs', 'subsegments_scale0_cluster.label')
            subsegments_file = os.path.join(output_dir, 'speaker_outputs', 'subsegments_scale0.json')
            
            if os.path.exists(cluster_label_file) and os.path.exists(subsegments_file):
                try:
                    self.logger.info(f"Using cluster label file: {cluster_label_file}")
                    segments = []
                    
                    # Load subsegments to get timestamps
                    import json
                    with open(subsegments_file, 'r') as f:
                        subsegments = json.load(f)
                    
                    # Load cluster labels
                    with open(cluster_label_file, 'r') as f:
                        cluster_labels = f.readlines()
                    
                    # Process each segment with its label
                    if len(subsegments) == len(cluster_labels):
                        for i, (subsegment, label_line) in enumerate(zip(subsegments, cluster_labels)):
                            if 'start' in subsegment and 'end' in subsegment:
                                start_time = float(subsegment['start'])
                                end_time = float(subsegment['end'])
                                duration = end_time - start_time
                                speaker = label_line.strip()
                                
                                segments.append({
                                    "start": start_time,
                                    "end": end_time,
                                    "speaker": f"SPEAKER_{speaker}",  # Format speaker ID
                                    "duration": duration
                                })
                        
                        # If we found segments, consolidate them and return
                        if segments:
                            self.logger.info(f"Parsed {len(segments)} segments from cluster labels")
                            consolidated_segments = self._consolidate_segments(segments)
                            self.logger.info(f"Consolidated to {len(consolidated_segments)} segments")
                            return consolidated_segments
                    else:
                        self.logger.warning(f"Mismatch between subsegments ({len(subsegments)}) and cluster labels ({len(cluster_labels)})")
                except Exception as e:
                    self.logger.error(f"Error processing cluster labels: {str(e)}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                if not os.path.exists(cluster_label_file):
                    self.logger.warning(f"Cluster label file not found: {cluster_label_file}")
                if not os.path.exists(subsegments_file):
                    self.logger.warning(f"Subsegments file not found: {subsegments_file}")
            
            # If we reach here, we couldn't find any valid diarization results
            self.logger.error("Could not find any valid diarization results")
            
            # Create a single segment for the entire audio as a last resort
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                duration = info.duration
            except Exception as e:
                self.logger.error(f"Error getting audio duration: {str(e)}")
                # If we can't determine the duration, use a default value
                duration = 60.0
            
            self.logger.warning(f"Creating a single segment for the entire audio ({duration:.2f} seconds)")
            return [{
                "start": 0.0,
                "end": duration,
                "speaker": "SPEAKER_0",
                "duration": duration
            }]
            
        except Exception as e:
            self.logger.error(f"Diarization process failed: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Create a single segment for the entire audio as a last resort
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                duration = info.duration
            except:
                # If we can't determine the duration, use a default value
                duration = 60.0
            
            self.logger.warning(f"Creating a single segment for the entire audio ({duration:.2f} seconds)")
            return [{
                "start": 0.0,
                "end": duration,
                "speaker": "SPEAKER_0",
                "duration": duration
            }]
        finally:
            # L04: temp adresář (preprocessed wav + manifest) + temp soubory
            # uklidit na VŠECH cestách včetně early-returnů a výjimek.
            if temp_dir:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            try:
                self._cleanup_temp_files()
            except Exception:
                pass
    
    def _consolidate_segments(self, segments: List[Dict[str, any]], gap_threshold: float = 0.5) -> List[Dict[str, any]]:
        """
        Consolidate segments for the same speaker that are close together.
        This helps create longer, more meaningful segments for transcription alignment.
        
        Args:
            segments: List of diarization segments
            gap_threshold: Maximum gap between segments to consolidate (in seconds)
            
        Returns:
            List of consolidated segments
        """
        if not segments:
            return []
            
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        
        consolidated = []
        current = sorted_segments[0].copy()
        
        for next_segment in sorted_segments[1:]:
            # If same speaker and gap is small enough, merge segments
            if (next_segment["speaker"] == current["speaker"] and 
                next_segment["start"] - current["end"] <= gap_threshold):
                current["end"] = next_segment["end"]
                current["duration"] = current["end"] - current["start"]
            else:
                # Add current segment to consolidated list and start a new one
                consolidated.append(current)
                current = next_segment.copy()
        
        # Add the last segment
        consolidated.append(current)
        
        return consolidated

    def _process_rttm_results(self, out_dir: str, base_filename: str) -> List[Dict[str, any]]:
        """
        Process RTTM results from NeMo diarization
        
        Args:
            out_dir: Directory containing the RTTM files
            base_filename: Base filename of the processed audio
            
        Returns:
            List of diarization segments with start, end, and speaker information
        """
        try:
            # Construct the RTTM file path
            base_name = os.path.splitext(base_filename)[0]
            rttm_file = os.path.join(out_dir, f"{base_name}.rttm")
            
            self.logger.info(f"Processing RTTM results from: {rttm_file}")
            
            if not os.path.exists(rttm_file):
                self.logger.warning(f"RTTM file not found: {rttm_file}")
                return []
            
            # Parse RTTM file
            segments = []
            with open(rttm_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        start_time = float(parts[3])
                        duration = float(parts[4])
                        end_time = start_time + duration
                        speaker = parts[7]
                        
                        segments.append({
                            "start": start_time,
                            "end": end_time,
                            "speaker": speaker,
                            "duration": duration
                        })
            
            self.logger.info(f"Processed {len(segments)} segments from RTTM file")
            return segments
            
        except Exception as e:
            self.logger.error(f"Failed to process RTTM results: {str(e)}")
            return self._fallback_simple_diarization(os.path.join(out_dir, base_filename))

    def _verify_and_fix_audio(self, wav_path: str) -> str:
        """Verify and fix audio tensor dimensions"""
        try:
            audio, sr = sf.read(wav_path)
            
            # Check for empty audio
            if len(audio) == 0:
                raise ValueError("Empty audio file detected")
            
            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Ensure proper shape (n_samples,)
            audio = audio.reshape(-1)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                from librosa import resample
                audio = resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Save fixed file
            fixed_path = wav_path.replace('.wav', '_fixed.wav')
            sf.write(fixed_path, audio, sr, subtype='PCM_16')
            self.temp_files.append(fixed_path)
            return fixed_path
            
        except Exception as e:
            self.logger.error(f"Audio verification failed: {str(e)}")
            raise

    def _force_audio_conversion(self, wav_path: str) -> str:
        """Force audio conversion with more aggressive settings"""
        try:
            # Load with pydub for robust conversion
            audio = AudioSegment.from_file(wav_path)
            
            # Force mono and 16kHz
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Normalize volume
            audio = audio.normalize()
            
            # Export with high quality settings
            forced_path = wav_path.replace('.wav', '_forced.wav')
            audio.export(forced_path, format='wav', 
                        parameters=["-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le"])
            
            self.temp_files.append(forced_path)
            return forced_path
            
        except Exception as e:
            self.logger.error(f"Force conversion failed: {str(e)}")
            raise

    def _create_manifest(self, audio_path: str) -> str:
        """Create a temporary manifest file for NeMo"""
        import json
        
        manifest_path = os.path.splitext(audio_path)[0] + '_manifest.json'
        
        with open(manifest_path, 'w') as f:
            entry = {
                "audio_filepath": audio_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": self.config.diarization.num_speakers if hasattr(self.config.diarization, 'num_speakers') else None,
                "rttm_filepath": None,
                "uem_filepath": None
            }
            f.write(json.dumps(entry))
        
        self.temp_files.append(manifest_path)
        return manifest_path
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary file {file_path}: {str(e)}")
    
    def _pad_audio_if_needed(self, wav_path: str) -> str:
        """Add silence padding to very short audio files to prevent tensor issues."""
        try:
            info = sf.info(wav_path)
            if info.duration >= 1.5:
                return wav_path
                
            self.logger.info(f"Padding audio file {wav_path} to minimum duration (1.5 seconds).")
            
            # Read the original audio
            audio, sr = sf.read(wav_path)
            
            # Calculate padding
            min_samples = int(1.5 * sr)
            current_samples = len(audio) if len(audio.shape) == 1 else len(audio[:, 0])
            padding_needed = max(0, min_samples - current_samples)
            
            if padding_needed > 0:
                # Create silence padding
                if len(audio.shape) == 1:  # Mono
                    padding = np.zeros(padding_needed)
                    padded_audio = np.concatenate([audio, padding])
                else:  # Multi-channel
                    padding = np.zeros((padding_needed, audio.shape[1]))
                    padded_audio = np.concatenate([audio, padding])
                
                # Save padded audio to a new file
                padded_path = wav_path.replace('.wav', '_padded.wav')
                sf.write(padded_path, padded_audio, sr)
                
                self.temp_files.append(padded_path)
                return padded_path
            
            return wav_path
            
        except Exception as e:
            self.logger.warning(f"Failed to pad audio file {wav_path}: {str(e)}")
            return wav_path

    def _ensure_wav_format(self, audio_path: str) -> str:
        """Ensure audio is in mono 16kHz WAV format required by NeMo"""
        try:
            # First check if the file is already in the correct format
            try:
                info = sf.info(audio_path)
                if info.channels == 1 and info.samplerate == 16000:
                    return audio_path
            except Exception as e:
                self.logger.warning(f"Could not read audio info: {str(e)}")

            self.logger.info(f"Converting audio file to mono 16kHz WAV: {audio_path}")
            temp_wav_path = os.path.splitext(audio_path)[0] + '_nemo_temp.wav'
            
            # Load audio with pydub
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono and resample to 16kHz
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            
            # Export with high-quality resampling
            audio.export(temp_wav_path, format="wav", parameters=[
                "-ac", "1",
                "-ar", "16000",
                "-acodec", "pcm_s16le"
            ])
            
            # Verify conversion
            try:
                info = sf.info(temp_wav_path)
                if info.channels != 1 or info.samplerate != 16000:
                    raise ValueError("Conversion failed - incorrect format")
            except Exception as e:
                self.logger.error(f"Post-conversion verification failed: {str(e)}")
                raise
            
            self.temp_files.append(temp_wav_path)
            return temp_wav_path
        
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {str(e)}")
            raise

    def _fix_audio_format(self, wav_path: str) -> str:
        """Fix audio format to ensure correct tensor dimensions"""
        try:
            self.logger.info(f"Fixing audio format for compatibility: {wav_path}")
            fixed_path = wav_path.replace('.wav', '_fixed.wav')
            
            # Load audio
            audio, sr = sf.read(wav_path)
            self.logger.info(f"Original audio shape: {audio.shape}, SR: {sr}")
            
            # Ensure mono and 16kHz
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                self.logger.info(f"Converted to mono shape: {audio.shape}")
            
            # Resample if needed
            if sr != 16000:
                from librosa import resample
                audio = resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
                sr = 16000
                self.logger.info(f"Resampled to 16kHz")
            
            # Ensure 2D shape (samples, 1)
            if len(audio.shape) == 1:
                audio = audio.reshape(-1, 1)
                self.logger.info(f"Reshaped to 2D: {audio.shape}")
            
            # Save with correct format
            sf.write(fixed_path, audio, sr, subtype='PCM_16')
            self.temp_files.append(fixed_path)
            return fixed_path
            
        except Exception as e:
            self.logger.error(f"Failed to fix audio format: {str(e)}")
            raise
            
    def _fallback_simple_diarization(self, audio_path: str) -> List[Dict[str, any]]:
        """Simple fallback diarization for when the main pipeline fails"""
        try:
            import librosa
            import numpy as np
            
            self.logger.info(f"Running simple fallback diarization for: {audio_path}")
            
            # Load the audio file
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Calculate energy
            energy = librosa.feature.rms(y=audio)[0]
            
            # Set a threshold for voice activity
            threshold = np.mean(energy) * 1.2
            
            # Find the segments with voice
            is_speech = energy > threshold
            
            # Create speech segments
            segments = []
            min_segment_length = 0.5  # minimum segment length in seconds
            speech_frames = int(min_segment_length * sr / librosa.get_duration(y=audio) * len(is_speech))
            
            # Simple clustering for speaker assignment
            speaker_id = 0
            speech_start = None
            
            for i, speech in enumerate(is_speech):
                frame_time = librosa.get_duration(y=audio) * i / len(is_speech)
                
                if speech and speech_start is None:
                    speech_start = frame_time
                elif not speech and speech_start is not None:
                    if frame_time - speech_start > min_segment_length:
                        segments.append({
                            "start": speech_start,
                            "end": frame_time,
                            "speaker": f"SPEAKER_{speaker_id}",
                            "duration": frame_time - speech_start
                        })
                    speech_start = None
            
            # Add final segment if needed
            if speech_start is not None:
                frame_time = librosa.get_duration(y=audio)
                segments.append({
                    "start": speech_start,
                    "end": frame_time,
                    "speaker": f"SPEAKER_{speaker_id}",
                    "duration": frame_time - speech_start
                })
            
            # If no segments found, return whole audio as one segment
            if not segments:
                duration = librosa.get_duration(y=audio)
                segments.append({
                    "start": 0.0,
                    "end": duration,
                    "speaker": "UNKNOWN",
                    "duration": duration
                })
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Simple fallback diarization failed: {str(e)}")
            
            # Return a simple single-speaker segment as last resort
            try:
                info = sf.info(audio_path)
                duration = info.duration
            except:
                duration = 10.0
            
            return [{
                "start": 0.0,
                "end": duration,
                "speaker": "SPEAKER_0",
                "duration": duration
            }]

    def _preprocess_audio(self, audio_file: str, temp_dir: str) -> str:
        """
        Preprocess audio file to ensure compatibility with NeMo diarization.
        Matches the implementation in diarize_audio.py
        """
        try:
            import soundfile as sf
            import librosa
            import numpy as np
            
            self.logger.info(f"Preprocessing audio file: {os.path.basename(audio_file)}")
            
            # Load audio file with librosa
            y, sr = librosa.load(audio_file, sr=16000, mono=True)
            
            # Ensure the audio is in the correct format (16kHz, mono)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Save as WAV file with soundfile
            base_filename = os.path.splitext(os.path.basename(audio_file))[0]
            temp_wav_path = os.path.join(temp_dir, f"{base_filename}_processed.wav")
            sf.write(temp_wav_path, y, sr, subtype='PCM_16')
            
            self.logger.info(f"Audio preprocessed and saved to: {temp_wav_path}")
            return temp_wav_path
        
        except Exception as e:
            self.logger.error(f"Error preprocessing audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return audio_file  # Return original file if preprocessing fails
    
    def _create_diarize_manifest(self, audio_file: str, temp_dir: str) -> str:
        """
        Creates a temporary manifest file for NeMo diarization.
        Matches the implementation in diarize_audio.py
        """
        import json
        
        manifest_content = {
            'audio_filepath': audio_file,
            'offset': 0,
            'duration': None,
            'label': 'infer',
            'text': '-',
            'num_speakers': self.config.diarization.num_speakers if hasattr(self.config.diarization, 'num_speakers') else None,
            'rttm_filepath': None,
            'uem_filepath': None
        }
        
        # Save to JSON file
        manifest_filepath = os.path.join(temp_dir, 'manifest.json')
        with open(manifest_filepath, 'w') as f:
            json.dump(manifest_content, f)
        
        self.logger.info(f"Created manifest file: {manifest_filepath}")
        return manifest_filepath
