# Voice Transcription using Whisper AI and Pyannote.audio (OpenAI and Hugging Face)

This project provides a Python script for transcribing audio files using OpenAI's Whisper AI model. It supports various features including speaker diarization, multiple output formats, and optimizations for long recordings.



## Setup

### Prerequisites

1. **Chocolatey** (for Windows users):
   ```
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```

2. **FFmpeg**:
   - Windows (using Chocolatey):
     ```
     choco install ffmpeg
     ```
   - macOS (using Homebrew):
     ```
     brew install ffmpeg
     ```
   - Ubuntu/Debian:
     ```
     sudo apt update && sudo apt install ffmpeg
     ```

3. **PyTorch**:
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Whisper AI**:
   ```
   pip install -U openai-whisper
   ```

### Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Place your audio files in the `data/input/` directory.

Run the transcription:

```
python main.py --input <path_to_audio_file> [--config <path_to_config>] [--format <format>]
```

Default configuration is loaded from the `config/` directory.

### Parameters

**Required:**
- `--input <path_to_audio_file>`: Path to the audio file you want to transcribe.

**Optional:**
- `--config <path_to_config>`: Path to custom configuration file.
  - **Default:** "config/config.yaml"
  - **Example:** `--config my_config.yaml`

- `--format <format>`: Specifies the output format for the transcription.
  - **Default:** Format specified in config.yaml
  - **Choices:** "txt", "json", "subtitle"
  - **Example:** `--format json`

### Example

```
python main.py --input data/input/example.mp3 --format txt
```

To use a different Whisper model, edit the `model.name` setting in `config/config.yaml`.

## Output

- The transcribed text will be saved in the `data/output/` directory.

## Configuration

The transcription process uses settings from `config/config.yaml`. You can modify this file to adjust various parameters of the Whisper model, output format, and chunking size for long recordings.



# Model Comparison Tool

The project includes a powerful model comparison tool that allows you to evaluate and compare the performance of different Whisper model sizes on the same audio input.

### What It Does

1. **Multi-Model Processing**: 
   - Processes the same audio file through multiple Whisper models
   - Compares: tiny, base, small, medium, large-v2, and large-v3 models

2. **Comparison Metrics**:
   - Calculates Levenshtein distance between transcriptions
   - Generates similarity scores (0-100%) between model outputs
   - Creates detailed comparison matrices

3. **Output Analysis**:
   - Generates CSV files with comparison matrices
   - Creates a detailed summary report
   - Provides visual text comparisons
   - Shows statistics for each model's performance

### Output Files

The tool generates several files in `data/algorithm_output/<audio_name>/<timestamp>/`:
- `levenshtein_distances.csv`: Raw distance scores between models
- `similarity_scores.csv`: Normalized similarity percentages
- `summary_report.txt`: Overall analysis and key findings
- `text_comparisons.txt`: Detailed text comparisons with statistics

### How to Run

1. **Basic Usage**:
   ```bash
   python run_comparison.py --input path/to/audio/file.mp3
   ```

2. **With Custom Config**:
   ```bash
   python run_comparison.py --input path/to/audio/file.mp3 --config path/to/config.yaml
   ```

### Example Output

When you run the comparison tool, you'll get output files like this:

1. **summary_report.txt**:
```
Shrnutí srovnání modelů
=====================

Průměrné skóre podobností:
----------------------
tiny: 85.23%
base: 88.45%
small: 91.67%
medium: 93.82%
large-v2: 95.44%
large-v3: 96.12%

Nejvíc podobný model:
large-v2 - large-v3
Podobnost: 98.45%

Nejvíc odlišný model:
tiny - large-v3
Podobnost: 82.34%
```

2. **text_comparisons.txt**:
```
Porovnání transkripcí mezi modely
================================

TINY
----
Délka textu: 1250 znaků
Text: "Hello everyone, welcome to the podcast. Today we're going to discuss..."

BASE
----
Délka textu: 1280 znaků
Text: "Hello everyone, welcome to the podcast. Today we're going to discuss..."

LARGE-V3
--------
Délka textu: 1295 znaků
Text: "Hello everyone, welcome to the podcast. Today we're going to discuss..."

Porovnání rozdílů mezi modely
============================

tiny vs base
Podobnost: 88.45%
Levenshteinova vzdálenost: 145

base vs large-v3
Podobnost: 94.23%
Levenshteinova vzdálenost: 74
```

3. **levenshtein_distances.csv**:
```csv
Model,tiny,base,small,medium,large-v2,large-v3
tiny,0,145,167,189,201,215
base,145,0,89,102,124,138
small,167,89,0,45,67,82
medium,189,102,45,0,34,41
large-v2,201,124,67,34,0,18
large-v3,215,138,82,41,18,0
```

4. **similarity_scores.csv**:
```csv
Model,tiny,base,small,medium,large-v2,large-v3
tiny,100,88.45,86.23,84.56,83.12,82.34
base,88.45,100,92.67,91.23,90.12,89.45
small,86.23,92.67,100,96.34,94.56,93.45
medium,84.56,91.23,96.34,100,97.23,96.78
large-v2,83.12,90.12,94.56,97.23,100,98.45
large-v3,82.34,89.45,93.45,96.78,98.45,100
```

The tool creates these files in the following directory structure:
```
data/
└── algorithm_output/
    └── your_audio_name/
        └── timestamp/
            ├── levenshtein_distances.csv
            ├── similarity_scores.csv
            ├── summary_report.txt
            └── text_comparisons.txt
```

This output helps you:
- Compare accuracy between different model sizes
- Identify which models are most similar/different
- See detailed transcription differences
- Make informed decisions about model selection