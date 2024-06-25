# Whisper AI Transcription Application

This application uses Whisper AI from OpenAI to transcribe audio files into text. Below are the detailed instructions for setting up and using the application.

## Prerequisites

### 1. Chocolatey - The Package Manager for Windows
Chocolatey is a package manager for Windows that simplifies the installation of software.

To install Chocolatey, open PowerShell with administrative privileges and run the following command:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```


### 2. Python and Pip
Ensure you have Python (version 3.7 or higher) and pip installed. You can install Python from the official website.

### 3. Pytorch
PyTorch is an open-source machine learning library. Install it using pip:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### 4. Whisper AI
Whisper AI is the transcription model used by this application. Install it using pip:
```bash
pip install -U openai-whisper
```


## Application Setup
Clone the Repository

Clone the repository containing the code:

```bash
git clone https://github.com/Deatrix09/whisper-voice-transcriber.git
```



## Configuration
The application uses a configuration file transcribe_config.py located in the config directory. Ensure that the configurations such as LANGUAGE, WHISPER_MODEL, and OUTPUT_PATH are set appropriately in this file.

### Usage
Run the application from the command line with the following command:

```bash
python main.py <path-to-audio-file> --language <language-code> --model <whisper-model>
```
#### Example

```bash
python main.py sample_audio.wav --language en --model base
```

#### Arguments
```
    audio_file (required): Path to the audio file to be transcribed.
    --language (optional): Language of the audio. Default is set in transcribe_config.py.
    --model (optional): Model to use for transcription. Default is set in transcribe_config.py.
```