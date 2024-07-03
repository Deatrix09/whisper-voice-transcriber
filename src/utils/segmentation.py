def simple_segmentation(segments, segment_length=10):
    current_speaker = "Speaker 1"
    for segment in segments:
        if int(segment["start"]) % segment_length == 0:
            current_speaker = f"Speaker {(int(segment['start']) // segment_length) % 2 + 1}"
        yield current_speaker, segment