
import os
from config.transcribe_config import TranscribeConfig as Config
from utils.utils import ensure_directory_exists
from utils.transcribe import transcribe_audio
from utils.detect_language import detect_language_of_text


def main(audio_file, model=Config.WHISPER_MODEL):

    ## Ensure the output directory exists
    ensure_directory_exists(Config.OUTPUT_PATH)

    ## Detect the language of the audio file
    language_output = detect_language_of_text(audio_file)

    ## Transcribe the audio file
    text, segments, language = transcribe_audio(audio_file, model=model)


    #Save the transcription to a file
    output_file = os.path.join(Config.OUTPUT_PATH, os.path.basename(audio_file) + ".txt")
    with open(output_file, "w") as f:
        f.write(f"Transcription of {audio_file}\n")
        f.write(f"Language: {language_output}\n")
        f.write("\n")
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            f.write(f"[{start:.2f} - {end:.2f}]: {text}\n")


    print(f"Transcription saved to {output_file}")




# This is the main function that will be called from the command line
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Transcribe audio files.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file.")
    parser.add_argument("--model", type=str, default=Config.WHISPER_MODEL, help="Model to use for transcription.")
    
    args = parser.parse_args()
    main(args.audio_file, args.model)