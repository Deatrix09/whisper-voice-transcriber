import whisper
import torch
from config.transcribe_config import TranscribeConfig as Config
import warnings


def transcribe_audio(file_path, model=Config.WHISPER_MODEL):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device: {device} with {dtype} precision \n")

    # Suppress the specific UserWarning
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


    # Load the model with the specified data type
    model = whisper.load_model(model).to(device)
    
    # Set the model to use the appropriate data type
    model.eval()
    model.to(dtype)

    # Transcribe the audio
    result = model.transcribe(file_path)
    return result["text"], result["segments"], result["language"]

