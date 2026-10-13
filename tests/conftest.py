"""Společné fixtures pro performance validační testy.

Vše běží bez GPU / bez stahování modelů (mock Pipeline, tiny wav generovaný za běhu).
"""
import sys
from pathlib import Path

import pytest

# Root projektu na sys.path, aby šlo `from src... import` i `from algorithm... import`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import (
    AppConfig, ModelConfig, WhisperConfig, AudioConfig, PerformanceConfig,
    TranscriptionConfig, DiarizationConfig, PyAnnoteConfig, NemoConfig,
    OutputConfig, LoggingConfig,
)


@pytest.fixture
def app_config(tmp_path):
    """Lehký AppConfig pro testy (žádné čtení YAML, žádné modely)."""
    return AppConfig(
        model=ModelConfig(name="tiny", language=""),
        whisper=WhisperConfig(sample_rate=16000),
        audio=AudioConfig(chunk_size=360, chunk_threshold_mb=20, chunk_overlap_sec=0),
        performance=PerformanceConfig(device="cpu", num_workers=4),
        transcription=TranscriptionConfig(
            task="transcribe", temperature=0.0,
            compression_ratio_threshold=2.4, logprob_threshold=-1.0,
            no_speech_threshold=0.6, fp16=False, verbose=False,
            multi_language=False,
        ),
        diarization=DiarizationConfig(
            enabled=False, module="pyannote", num_speakers=None,
            pyannote=PyAnnoteConfig(
                enabled=False, model="pyannote/speaker-diarization-3.1",
                auth_token="test", min_duration_off=0.5,
                min_cluster_size=5, method="average",
            ),
            nemo=NemoConfig(enabled=False, vad_model="x", speaker_model="y"),
        ),
        output=OutputConfig(format="txt", style="timestamped"),
        logging=LoggingConfig(
            level="WARNING", file="test_log",
            console_level="WARNING", file_level="WARNING", retention_days=1,
        ),
    )


@pytest.fixture
def comparator(app_config):
    from algorithm.levenshtein_comparator import LevenshteinComparator
    return LevenshteinComparator(app_config)


@pytest.fixture
def sample_transcriptions():
    return {
        "tiny": "hello world this is a test",
        "base": "hello world this is a test!",
        "small": "completely different sentence here",
    }


def make_wav(path: Path, seconds: float = 1.0, sample_rate: int = 16000):
    """Vygeneruj tichý 16-bit mono WAV bez ffmpeg/pydub (čistý stdlib)."""
    import wave
    import struct
    n_frames = int(seconds * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * n_frames, *([0] * n_frames)))
    return path
