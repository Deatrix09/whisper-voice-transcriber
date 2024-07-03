import datetime
from utils.segmentation import simple_segmentation

def write_transcription(output_file, audio_file, language_output, segments):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Transcription of {audio_file}\n")
        f.write(f"Language: {language_output}\n\n")
        
        for current_speaker, segment in simple_segmentation(segments):
            start = segment["start"]
            text = segment["text"]
            timestamp = str(datetime.timedelta(seconds=int(start)))
            f.write(f"[{timestamp}] {current_speaker}: {text}\n")