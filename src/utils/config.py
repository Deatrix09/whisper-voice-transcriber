import yaml

from dataclasses import dataclass

from typing import Optional, Dict, Any



@dataclass
class ModelConfig:

    name: str
    language: str



@dataclass
class WhisperConfig:

    sample_rate: int



@dataclass
class AudioConfig:

    chunk_size: int



@dataclass
class PerformanceConfig:

    device: str
    num_workers: int



@dataclass
class TranscriptionConfig:

    task: str
    temperature: float
    compression_ratio_threshold: float
    logprob_threshold: float
    no_speech_threshold: float
    fp16: bool
    verbose: bool
    multi_language: bool



@dataclass
class DiarizationConfig:

    enabled: bool
    model: str
    auth_token: str
    min_duration_off: float
    min_cluster_size: int
    method: str



@dataclass
class OutputConfig:

    format: str
    style: str



@dataclass
class LoggingConfig:

    level: str
    file: str
    console_level: str
    file_level: str
    retention_days: int



@dataclass
class AppConfig:

    model: ModelConfig
    whisper: WhisperConfig
    audio: AudioConfig
    performance: PerformanceConfig
    transcription: TranscriptionConfig
    diarization: DiarizationConfig
    output: OutputConfig
    logging: LoggingConfig



    @classmethod

    def from_yaml(cls, path: str) -> 'AppConfig':

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

            

        return cls(

            model=ModelConfig(**config_dict['model']),
            whisper=WhisperConfig(**config_dict['whisper']),
            audio=AudioConfig(**config_dict['audio']),
            performance=PerformanceConfig(**config_dict['performance']),
            transcription=TranscriptionConfig(**config_dict['transcription']),
            diarization=DiarizationConfig(**config_dict['diarization']),
            output=OutputConfig(**config_dict['output']),
            logging=LoggingConfig(**config_dict['logging'])

        ) 
