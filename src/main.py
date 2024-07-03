import os
import torch
from config.transcribe_config import TranscribeConfig as Config
from utils.utils import ensure_directory_exists
from utils.transcribe import transcribe_audio
from utils.detect_language import detect_language_of_text
from utils.transcription import write_transcription
from utils.segmentation import simple_segmentation


def main(audio_file, model=Config.WHISPER_MODEL):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ensure_directory_exists(Config.OUTPUT_PATH)

    language_output = detect_language_of_text(audio_file)
    print(f"Detected language: {language_output}")

    text, segments, language = transcribe_audio(audio_file, model)
    print(f"Transcription completed. Detected language: {language}")

    output_file = os.path.join(Config.OUTPUT_PATH, f"{os.path.basename(audio_file)}_transcipted.txt")
    write_transcription(output_file, audio_file, language_output, segments)

    print(f"Transcription saved to {output_file}")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Transcribe audio files with simple speaker segmentation.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file.")
    parser.add_argument("--model", type=str, default=Config.WHISPER_MODEL, help="Model to use for transcription.")
    
    args = parser.parse_args()
    main(args.audio_file, args.model)