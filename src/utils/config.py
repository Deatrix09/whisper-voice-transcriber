# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import re
import yaml

from dataclasses import dataclass, field

from typing import List, Optional, Dict, Any


# "${HF_TOKEN}" (celá hodnota) -> os.environ, jinak hodnota beze změny.
# P0 0.1: token nikdy natvrdo v yaml, jen placeholder + .env.
_ENV_PLACEHOLDER_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def expand_env(value):
    """Expanduj "${VAR}" placeholder z prostředí (jinak vrať beze změny)."""
    if isinstance(value, str):
        m = _ENV_PLACEHOLDER_RE.match(value.strip())
        if m:
            return os.environ.get(m.group(1), "")
    return value


class ConfigError(Exception):
    """P0 0.3: chyba konfigurace s kontextem (soubor, případně řádek a sekce)."""

    def __init__(self, path: str, reason: str, section: str = "", line=None):
        where = path
        if line is not None:
            where += f":{line}"
        if section:
            where += f" [{section}]"
        super().__init__(f"{where}: {reason}")
        self.path = path
        self.reason = reason
        self.section = section
        self.line = line


def _key_lines(path: str) -> Dict[str, int]:
    """Best-effort mapa 'sekce[.podsekce]' -> řádek (pro hlášky)."""
    try:
        from yaml.nodes import MappingNode
        with open(path, 'r', encoding='utf-8') as f:
            root = yaml.compose(f)
        lines: Dict[str, int] = {}

        def walk(node, prefix: str):
            if not isinstance(node, MappingNode):
                return
            for key_node, value_node in node.value:
                dotted = f"{prefix}.{key_node.value}" if prefix else str(key_node.value)
                try:
                    lines[dotted] = int(key_node.start_mark.line) + 1
                except Exception:
                    pass
                walk(value_node, dotted)

        walk(root, "")
        return lines
    except Exception:
        return {}


def resolve_device(device_setting: str) -> str:
    """P0 0.3: auto|cuda|cpu -> efektivní cuda|cpu (auto = cuda když jde)."""
    dev = (device_setting or "cpu").strip().lower()
    if dev not in ("auto", "cuda", "cpu"):
        raise ValueError(f"neznámé performance.device={device_setting!r} (auto|cuda|cpu)")
    if dev == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"



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
    chunk_overlap_sec: int = 0  # P0 0.3 default (zpětná kompatibilita)



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

        try:
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigError(path, "soubor neexistuje")
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            line = (int(mark.line) + 1) if mark is not None else None
            raise ConfigError(path, f"YAML syntax error: {e}", line=line)
        if not isinstance(config_dict, dict):
            raise ConfigError(path, "kořen musí být mapa sekcí")

        lines = _key_lines(path)

        def need(section: str):
            if section not in config_dict or not isinstance(config_dict[section], dict):
                raise ConfigError(path, f"chybí povinná sekce '{section}'",
                                  section=section, line=lines.get(section))

        for section in ("model", "audio", "performance", "diarization",
                        "output", "logging"):
            need(section)

        try:
            # Convert nested diarization configs (chybějící sekce -> None, staré YAML fungují)
            if 'pyannote' in config_dict['diarization']:
                pyannote_raw = dict(config_dict['diarization']['pyannote'])
                # P0 0.1: "${HF_TOKEN}" -> env (nikdy hardcoded secret v yaml).
                if 'auth_token' in pyannote_raw:
                    pyannote_raw['auth_token'] = expand_env(pyannote_raw['auth_token'])
                config_dict['diarization']['pyannote'] = PyAnnoteConfig(**pyannote_raw)
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
        except TypeError as e:
            raise ConfigError(path, f"špatné klíče v sekci ({e})")

    KNOWN_WHISPER_MODELS = ("tiny", "base", "small", "medium", "large",
                            "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo")
    KNOWN_MODULES = ("pyannote", "sortformer", "nemotron-3", "nemo")
    KNOWN_FORMATS = ("json", "txt", "subtitle")
    KNOWN_DEVICES = ("auto", "cuda", "cpu")
    KNOWN_ENGINES = ("auto", "faster-whisper", "parakeet")

    def validate(self) -> List[str]:
        """P0 0.3: sémantická kontrola. Vrátí warnings; fatální = ConfigError.

        device=cuda bez CUDA = warning + fallback (ne crash).
        Neznámý model.name = warning (budoucí modely).
        """
        warnings: List[str] = []
        if self.model.name not in self.KNOWN_WHISPER_MODELS:
            warnings.append(f"model.name={self.model.name!r} není v známých "
                            f"{list(self.KNOWN_WHISPER_MODELS)} (pokračuji)")
        device = str(getattr(self.performance, "device", "cpu"))
        if device not in self.KNOWN_DEVICES:
            raise ConfigError("", f"performance.device={device!r} "
                                  f"(auto|cuda|cpu)", section="performance")
        if device in ("cuda", "auto"):
            try:
                import torch
                has_cuda = torch.cuda.is_available()
            except Exception:
                has_cuda = False
            if not has_cuda:
                warnings.append(f"performance.device={device} ale CUDA není "
                                f"k dispozici -> fallback cpu")
        module = str(getattr(self.diarization, "module", "")).lower()
        if self.diarization.enabled and module not in self.KNOWN_MODULES:
            raise ConfigError("", f"diarization.module={module!r} "
                                  f"{list(self.KNOWN_MODULES)}", section="diarization")
        ns = getattr(self.diarization, "num_speakers", None)
        if ns is not None and (not isinstance(ns, int) or ns < 1):
            raise ConfigError("", f"diarization.num_speakers={ns!r} musí být int >= 1",
                              section="diarization")
        chunk = getattr(self.audio, "chunk_size", 30)
        overlap = getattr(self.audio, "chunk_overlap_sec", 0)
        if chunk <= 10:
            raise ConfigError("", f"audio.chunk_size={chunk} <= 10s zvedá WER "
                                  f"(viz processor guard)", section="audio")
        if overlap < 0 or overlap >= chunk:
            raise ConfigError("", f"audio.chunk_overlap_sec={overlap} musí být "
                                  f"v [0, chunk_size)", section="audio")
        if self.output.format not in self.KNOWN_FORMATS:
            raise ConfigError("", f"output.format={self.output.format!r} "
                                  f"{list(self.KNOWN_FORMATS)}", section="output")
        engine = str(getattr(getattr(self, "asr", None), "engine", "faster-whisper"))
        if engine not in self.KNOWN_ENGINES:
            raise ConfigError("", f"asr.engine={engine!r} {list(self.KNOWN_ENGINES)}",
                              section="asr")
        return warnings


def validate_file(path: str) -> List[str]:
    """Načti + validuj config. Vrátí warnings; chyba = ConfigError."""
    return AppConfig.from_yaml(path).validate()
