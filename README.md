# Voice Transcription with faster-whisper and Speaker Diarization

[![tests](https://github.com/Deatrix09/whisper-voice-transcriber/actions/workflows/tests.yml/badge.svg?branch=Production)](https://github.com/Deatrix09/whisper-voice-transcriber/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

CLI app for transcribing audio files with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(CTranslate2, INT8) + speaker diarization (Nemotron-3 / pyannote / Sortformer / NeMo).
Optimized for long recordings: streaming 30s chunks, Silero VAD pre-filter, lazy model loading.

## Features

- **Dual ASR engines** (`asr.engine`): `faster-whisper` (default, `large-v3-turbo`) and
  `parakeet` (NeMo Parakeet-TDT-0.6B-v3, 25 languages incl. Czech) with `auto` router
  and graceful fallback. Swappable via `ASRProvider` factory, no pipeline changes needed.
- **Streaming audio pipeline**: PyAV decode to 16 kHz mono without loading the whole file
  into RAM, 30 s chunks with 2 s overlap (chunks ≤10 s are rejected — they raise WER),
  per-chunk timing + RTFx logs.
- **Diarization modules** (`diarization.module`): `nemotron-3` (default, open-weight,
  no HF token, up to 8 speakers), `pyannote` (community-1, needs HF token),
  `sortformer` (max 4 speakers), `nemo` (legacy clustering).
- **Evaluation**: `run_comparison.py` compares model sizes with Levenshtein similarity,
  WER/CER/cpWER (jiwer) and RTFx timing columns.
- **128 pytest tests**, reproducible builds (`pyproject.toml` + `uv.lock`), GPU Docker image.

## Setup

### Prerequisites

1. **Python 3.11** (see `pyproject.toml`: `requires-python >=3.11,<3.12`).
2. **FFmpeg** (required for audio conversion/chunking):
   - Windows: `choco install ffmpeg` (or `winget install ffmpeg`)
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
3. **NVIDIA GPU driver supporting CUDA 12.6** for GPU runs (CPU works too, slower).
   PyTorch is installed from the CUDA 12.6 wheel index, because faster-whisper
   (CTranslate2) needs the CUDA 12 cuBLAS/cuDNN libraries.

### Installation

```bash
# Option A: uv (reproducible, recommended) - exact versions from uv.lock
uv sync                     # CLI
uv sync --extra ui          # + Streamlit UI
uv sync --extra test        # + pytest

# Option B: pip + venv (requirements*.txt are generated from uv.lock)
python -m venv venv
venv/Scripts/activate  # Windows; source venv/bin/activate on Linux
pip install -r requirements-cuda.txt   # NVIDIA GPU (CUDA 12.6 torch wheels)
# pip install -r requirements-cpu.txt  # CPU-only boxes / CI
```

Optional NeMo engines (`parakeet` ASR, `sortformer` and legacy `nemo` diarization)
are **not** part of the lockfile: `nemo-toolkit` 3.0 pins `lightning<=2.4.0` and
`hydra-core<=1.3.2`, versions with known security advisories. Install it only if
you need these modules, and load only models and configs from sources you trust:

```bash
pip install "nemo-toolkit[asr]"   # typically Linux + CUDA
```

Secrets (only needed for the `pyannote` module — Nemotron-3 default needs none):

```bash
copy .env.example .env   # Windows; cp .env.example .env on Linux
# then put your token into .env: HF_TOKEN=<token from https://hf.co/settings/tokens>
```

Verify without heavy downloads (`--help` needs no torch):

```bash
python main.py --help            # <1 s
python main.py --validate --config config/config.yaml   # config check, exit 0/2
```

## Usage

```bash
python main.py --input <path_to_audio> [--config config/config.yaml] [--format txt|json|subtitle]
```

Examples:

```bash
python main.py --input data/input/example.mp3 --format txt
python main.py --input data/input/podcast.mp3 --config config/config.yaml --format json
```

- `--input`: required (except with `--validate`).
- `--config`: default `config/config.yaml`.
- `--format`: `txt` (default from config), `json`, or `subtitle`; overrides config.
- Output lands in `data/output/<input>_<timestamp>.<format>`.

## Graphical interface (Streamlit)

Optional UI over the same Python API. The CLI commands stay as they are
(`python main.py`, `python run_comparison.py`, `python run_multiple_configs.py`).

Streamlit is included in `requirements*.txt` (pip) and in the `ui` extra
(`uv sync --extra ui`). FFmpeg must be installed and on `PATH` (see Prerequisites). From the project root:

```bash
streamlit run app.py
```

## Configuration (`config/config.yaml`)

```yaml
asr:
  engine: "faster-whisper"  # auto|faster-whisper|parakeet (auto: lang in 25 -> parakeet)
  model: "large-v3-turbo"   # has priority over model.name; empty = use model.name
  compute_type: null        # null = auto (int8_float16 CUDA / int8 CPU)
  batch_size: 4             # pyannote batches; parakeet batch (8 recommended)
  align_words: false        # true = word-level timestamps
  parakeet_model: null      # null = nvidia/parakeet-tdt-0.6b-v3
model:
  name: "large-v3-turbo"   # fallback tiny/base/small/medium/large-v2/large-v3/turbo
  language: ""              # e.g. "cs"; empty = auto-detect
audio:
  chunk_size: 30            # seconds; <=10 rejected (raises WER)
  chunk_threshold_mb: 20    # files above this size are chunked
  chunk_overlap_sec: 2
performance:
  device: "cuda"            # auto|cuda|cpu (auto = cuda when available)
  num_workers: 4            # >=2 = diarization + transcription in parallel
diarization:
  enabled: true
  module: "nemotron-3"      # nemotron-3|pyannote|sortformer|nemo (legacy)
  num_speakers: null        # exact count or null for auto-detection
```

Run `python main.py --validate --config <file>` to check a config (prints warnings,
exit 2 on errors such as unknown module, bad chunk sizes, or wrong format).

## Model Comparison Tool

Compares transcription engines/sizes on the same audio (Levenshtein + WER/CER/cpWER + RTFx):

```bash
python run_comparison.py --input path/to/audio.mp3
python run_comparison.py --input path/to/audio.mp3 --models tiny,turbo,parakeet
```

Results go to `data/algorithm_output/<audio>/<timestamp>/`:

- `levenshtein_distances.csv` / `similarity_scores.csv` — pairwise matrices
- `wer_cer.csv` — columns `model,wer,cer,cpwer,reference` (reference = largest model)
- `timing.csv` — columns `model,seconds,audio_seconds,rtfx`
- `summary_report.txt` — averages, most/least similar pair, RTFx and accuracy sections
- `text_comparisons.txt` — full texts + pairwise distances

## Docker (GPU)

```bash
cp .env.example .env   # HF_TOKEN only needed for pyannote module
docker compose build
docker compose run --rm transcriber --input data/input/example.mp3 --format txt
# CPU-only box: docker compose run --rm transcriber-cpu --input ... --format txt
```

Mounts: `./data/input` (ro), `./data/output`, `hf-cache` volume. Healthcheck verifies torch CUDA.

## Benchmarking (before/after on the same file)

```powershell
Measure-Command { python main.py --input data/input/example.mp3 --format txt }
# watch the log: "Chunk done: ... RTFx=...", "VRAM [after-transcribe]", per-model timing
nvidia-smi dmon -s mu -d 1 -o DT   # VRAM sampling in a second terminal
```

## Project Structure

```
main.py                      # CLI (--input/--config/--format/--validate)
run_comparison.py            # model comparison (--models tiny,turbo,parakeet)
run_multiple_configs.py      # batch runner over config/ folder
src/
  core/pipeline.py           # orchestration (lazy ASR/diarizer, parallel, merge)
  audio/processor.py         # PyAV streaming, Silero VAD prefilter, 30s chunks
  transcription/
    base_transcriber.py      # ASRProvider interface (transcribe/unload)
    asr_factory.py           # create_asr_provider router
    transcriber.py           # faster-whisper engine
    parakeet_transcriber.py  # NeMo Parakeet-TDT engine
  diarization/               # pyannote/sortformer/nemotron-3/nemo + factory
  utils/                     # config (validation), logger, formatters, paths
algorithm/levenshtein_comparator.py  # similarity + WER/CER/cpWER + timing
config/config.yaml           # default config (HF token via ${HF_TOKEN} + .env, never hardcoded)
tests/                       # 128 pytest tests (mocked heavies + tiny smoke)
Dockerfile.cuda / docker-compose.yml / pyproject.toml / uv.lock
```

## Development

```bash
python -m pytest tests/ -q
python -m compileall -q src algorithm
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first —
all changes go through pull requests against the `Production` branch.
This project follows a [Code of Conduct](CODE_OF_CONDUCT.md).

Found a security issue? Please report it privately, see [SECURITY.md](SECURITY.md).

## License

Copyright (c) 2024-2026 Kleveta Lukas

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

See [LICENSE](LICENSE) for the full text. Every source file carries an
`SPDX-License-Identifier: Apache-2.0` header.
