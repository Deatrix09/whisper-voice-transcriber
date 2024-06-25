
class TranscribeConfig:
    WHISPER_MODEL = "base"   # Možné hodnoty: tiny, base, small, medium, large
    LANGUAGE = "en" # Základní jazyk pro transkripci
    AUDIO_PATH = "data/audio_files/"
    OUTPUT_PATH = "data/output_files/"
    