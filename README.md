# Voice Transcription using Whisper AI and Pyannote.audio (OpenAI and Hugging Face)

This project provides a Python script for transcribing audio files using OpenAI's Whisper AI model. It supports various features including speaker diarization, multiple output formats, and optimizations for long recordings.

## Features

1. **Multi-Language Support**: Automatically detects and transcribes multiple languages in the same audio file.
2. **Speaker Diarization**: Optional speaker identification with improved alignment between transcription and diarization.
3. **Multiple Output Formats**: Supports txt, json, csv, and srt output formats.
4. **Long Recording Optimization**: 
   - Processes audio in chunks to manage memory efficiently.
   - Saves intermediate results to prevent data loss in case of interruptions.

## Setup

### Prerequisites

1. **Chocolatey** (for Windows users):
   ```
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```

2. **PyTorch**:
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Whisper AI**:
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
python main.py <path_to_audio_file> [--model MODEL] [--diarization] [--formats FORMAT [FORMAT ...]]
```


### Parameters

**Required:**
- `<path_to_audio_file>`: Path to the audio file you want to transcribe.

**Optional:**
- `--model MODEL`: Specifies the Whisper model to use for transcription.
  - **Default:** "base"
  - **Choices:** "tiny", "base", "small", "medium", "large"
  - **Example:** `--model medium`

- `--diarization`: Enables speaker diarization (speaker identification).
  - **Default:** Disabled
  - **Usage:** Include this flag to enable diarization

- `--formats FORMAT [FORMAT ...]`: Specifies the output formats for the transcription.
  - **Default:** "txt"
  - **Choices:** "txt", "json", "srt"
  - **Example:** `--formats txt json srt`

### Example

```
python main.py data/input/example.mp3 --model medium --diarization --formats txt json srt
```

## Output

- The transcribed text will be saved in the `data/output/` directory.
- Intermediate results (for long recordings) are saved in `data/output/intermediate/`.

## Configuration

The transcription process uses settings from `config/whisper_config.yaml`. You can modify this file to adjust various parameters of the Whisper model, output format, and chunking size for long recordings.

## Note

- Larger models generally provide better accuracy but require more computational resources and time. Choose the model size based on your needs and available resources.
- The script now always uses multi-language detection and transcription, providing better support for audio files with multiple languages.
- The alignment between transcription and diarization has been improved for better accuracy in speaker identification.