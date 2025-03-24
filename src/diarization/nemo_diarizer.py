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
            
            # Load VAD model
            vad_model_name = self.config.diarization.nemo.vad_model
            self.logger.info(f"Loading VAD model: {vad_model_name}")
            self.vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
                model_name=vad_model_name
            )
            
            # Load speaker embedding model
            speaker_model_name = self.config.diarization.nemo.speaker_model
            self.logger.info(f"Loading speaker model: {speaker_model_name}")
            self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name=speaker_model_name
            )
            
            # Load multiscale diarization decoder model if specified
            if self.config.diarization.nemo.msdd_model:
                msdd_model_name = self.config.diarization.nemo.msdd_model
                self.logger.info(f"Loading MSDD model: {msdd_model_name}")
                self.msdd_model = nemo_asr.models.ClusteringDiarizer.from_pretrained(
                    model_name=msdd_model_name
                )
            
            # Move models to appropriate device
            if torch.cuda.is_available() and self.config.performance.device == "cuda":
                self.logger.info("Moving NeMo models to CUDA")
                if self.vad_model:
                    self.vad_model = self.vad_model.to(torch.device("cuda"))
                if self.speaker_model:
                    self.speaker_model = self.speaker_model.to(torch.device("cuda"))
                if self.msdd_model:
                    self.msdd_model = self.msdd_model.to(torch.device("cuda"))
            
            self.logger.info("Successfully initialized NeMo diarization models")
            
        except ImportError as e:
            self.logger.error(f"Failed to import NeMo: {str(e)}. Is NeMo properly installed?")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize NeMo diarization: {str(e)}")
            raise
    

    def process(self, audio_path: str) -> List[Dict[str, any]]:
        """Process audio file using NeMo diarization pipeline"""
        try:
            import nemo.collections.asr as nemo_asr

            # Convert to WAV format with proper specs
            wav_path = self._ensure_wav_format(audio_path)
            
            # Verify and fix audio tensor dimensions
            wav_path = self._verify_and_fix_audio(wav_path)
            
            # Create manifest file
            manifest_file = self._create_manifest(wav_path)
            
            # Create output directory
            out_dir = os.path.join(os.path.dirname(wav_path), 'diar_outputs')
            os.makedirs(out_dir, exist_ok=True)

            # Configuration with proper tensor handling
            diar_config = {
                'device': 'cuda' if torch.cuda.is_available() and self.config.performance.device == "cuda" else 'cpu',
                'num_workers': 0,  # Avoid multiprocessing issues
                'diarizer': {
                    'manifest_filepath': manifest_file,
                    'out_dir': out_dir,
                    'oracle_vad': False,
                    'speaker_embeddings': {
                        'model_path': self.config.diarization.nemo.speaker_model,
                        'parameters': {
                            'window_length_in_sec': 1.5,
                            'shift_length_in_sec': 0.75,
                            'multiscale_weights': [1.0],
                            'save_embeddings': False
                        }
                    },
                    'vad': {
                        'model_path': self.config.diarization.nemo.vad_model,
                        'parameters': {
                            'threshold': 0.5,
                            'window_length_in_sec': 0.15,
                            'shift_length_in_sec': 0.01,
                            'smoothing': False,
                            'smoothing_method': "median"
                        }
                    },
                    'clustering': {
                        'parameters': {
                            'method': 'sc',
                            'min_samples': 2,
                            'oracle_num_speakers': False,
                            'max_num_speakers': 8
                        }
                    }
                }
            }

            # Handle number of speakers if specified
            if hasattr(self.config.diarization, 'num_speakers') and self.config.diarization.num_speakers:
                diar_config['diarizer']['clustering']['parameters']['oracle_num_speakers'] = True
                diar_config['diarizer']['clustering']['parameters']['oracle_value'] = self.config.diarization.num_speakers

            # Convert to OmegaConf
            cfg = OmegaConf.create(diar_config)

            # Create diarizer
            diarizer = nemo_asr.models.ClusteringDiarizer(cfg=cfg)
            
            # Set models directly to avoid serialization issues
            diarizer.speaker_model = self.speaker_model
            diarizer.vad_model = self.vad_model
            
            # Run diarization with tensor shape validation
            try:
                diarizer.diarize()
            except RuntimeError as e:
                if "0-d tensor" in str(e):
                    self.logger.error("Empty tensor detected - checking audio file")
                    # Re-process audio with more aggressive conversion
                    wav_path = self._force_audio_conversion(wav_path)
                    cfg.diarizer.manifest_filepath = self._create_manifest(wav_path)
                    diarizer.diarize()
                else:
                    raise

            # Process results...
            return self._process_rttm_results(out_dir, os.path.basename(wav_path))

        except Exception as e:
            self.logger.error(f"Diarization failed: {str(e)}")
            return self._fallback_simple_diarization(audio_path)

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
