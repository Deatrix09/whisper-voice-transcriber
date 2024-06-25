import whisper
from config.transcribe_config import TranscribeConfig as Config

def transcribe_audio(file_path, model=Config.WHISPER_MODEL):
    model = whisper.load_model(model)


    result = model.transcribe(file_path, verbose=False)

    return result["text"], result["segments"], result["language"]

