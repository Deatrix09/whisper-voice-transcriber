import yaml

from dataclasses import dataclass, field

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
    chunk_threshold_mb: int
    chunk_overlap_sec: int



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
class AsrConfig:
    """FÁZE 4: sjednocené ASR nastavení (nová `asr:` sekce, vše s defaulty).

    Staré YAML bez `asr:` sekce fungují dál: model se bere z `model.name`,
    compute_type = auto. `engine: auto|parakeet` aktivuje až FÁZE 5 router
    (do té doby warning + faster-whisper fallback v transcriberu).
    """

    engine: str = "faster-whisper"  # auto|faster-whisper|parakeet
    model: str = "large-v3-turbo"
    compute_type: Optional[str] = None  # None = auto (int8_float16 CUDA / int8 CPU)
    batch_size: int = 4  # pyannote segmentation/embedding batche; parakeet batch
    align_words: bool = False  # True = word_timestamps (volitelný align krok)
    parakeet_model: Optional[str] = None  # None = nvidia/parakeet-tdt-0.6b-v3



@dataclass
class PyAnnoteConfig:

    enabled: bool
    model: str
    auth_token: str
    min_duration_off: float
    min_cluster_size: int
    method: str



@dataclass
class NemoConfig:

    enabled: bool
    vad_model: str
    speaker_model: str
    msdd_model: Optional[str] = None
    clustering_params: Optional[Dict[str, Any]] = None
    vad_params: Optional[Dict[str, Any]] = None



@dataclass
class SortformerConfig:

    enabled: bool
    model: str = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    batch_size: int = 1



@dataclass
class NemotronConfig:

    enabled: bool
    model: str = "nvidia/Nemotron-3-Diarization"



@dataclass
class DiarizationConfig:

    enabled: bool
    module: str  # "pyannote", "sortformer", "nemotron-3", "nemo" (legacy)
    num_speakers: Optional[int] = None
    pyannote: Optional[PyAnnoteConfig] = None
    nemo: Optional[NemoConfig] = None
    sortformer: Optional[SortformerConfig] = None
    nemotron: Optional[NemotronConfig] = None



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
    asr: AsrConfig = field(default_factory=AsrConfig)



    @classmethod

    def from_yaml(cls, path: str) -> 'AppConfig':

        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # Convert nested diarization configs (chybějící sekce -> None, staré YAML fungují)
        if 'diarization' in config_dict:
            if 'pyannote' in config_dict['diarization']:
                config_dict['diarization']['pyannote'] = PyAnnoteConfig(**config_dict['diarization']['pyannote'])
            if 'nemo' in config_dict['diarization']:
                config_dict['diarization']['nemo'] = NemoConfig(**config_dict['diarization']['nemo'])
            if 'sortformer' in config_dict['diarization']:
                config_dict['diarization']['sortformer'] = SortformerConfig(**config_dict['diarization']['sortformer'])
            if 'nemotron' in config_dict['diarization']:
                config_dict['diarization']['nemotron'] = NemotronConfig(**config_dict['diarization']['nemotron'])

        # FÁZE 4: `asr:` sekce je volitelná (defaulty AsrConfig), staré YAML fungují.
        asr_raw = config_dict.get('asr')
        asr_cfg = AsrConfig(**asr_raw) if isinstance(asr_raw, dict) else AsrConfig()

        return cls(

            model=ModelConfig(**config_dict['model']),
            whisper=WhisperConfig(**config_dict['whisper']),
            audio=AudioConfig(**config_dict['audio']),
            performance=PerformanceConfig(**config_dict['performance']),
            transcription=TranscriptionConfig(**config_dict['transcription']),
            diarization=DiarizationConfig(**config_dict['diarization']),
            output=OutputConfig(**config_dict['output']),
            logging=LoggingConfig(**config_dict['logging']),
            asr=asr_cfg

        )
