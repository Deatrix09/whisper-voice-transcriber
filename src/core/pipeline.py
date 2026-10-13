from typing import Dict, List
import os
from src.utils.logger import setup_logger

# Pozn. (F): torch se importuje lazy v metodách, aby `import src.core.pipeline`
# fungoval i bez torch (testy, --help cesty). torch je v sys.modules cachován.


def _create_diarizer(config):
    from src.diarization.diarizer import Diarizer
    return Diarizer(config)


def _create_transcriber(config):
    # FÁZE 5: dual-engine router (faster-whisper | parakeet | auto).
    # Vrací inicializovaný transcriber; při pádu parakeet init je uvnitř
    # graceful fallback na faster-whisper. Stejné rozhraní (.transcribe/.model).
    from src.transcription.parakeet_transcriber import create_asr_transcriber
    return create_asr_transcriber(config)


class Pipeline:
    def __init__(self, config):
        from src.audio.processor import AudioProcessor
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file,
                                     file_level=getattr(config.logging, "file_level", None),
                                     console_level=getattr(config.logging, "console_level", None))

        self.logger.info("Initializing pipeline components...")
        self.audio_processor = AudioProcessor(config)
        self.diarizer = _create_diarizer(config)
        self.transcriber = _create_transcriber(config)

    @staticmethod
    def release_model_resources(pipeline) -> None:
        """Uvolni těžké modely (Whisper/pyannote) z RAM/VRAM. Idempotentní."""
        import gc
        try:
            transcriber = getattr(pipeline, "transcriber", None)
            if transcriber is not None and getattr(transcriber, "model", None) is not None:
                del transcriber.model
                transcriber.model = None
        except Exception:
            pass
        try:
            diarizer = getattr(pipeline, "diarizer", None)
            inner = getattr(diarizer, "diarizer", None) if diarizer is not None else None
            if inner is not None:
                # pyannote drží .pipeline, sortformer/nemotron drží .model
                for attr in ("pipeline", "model"):
                    if getattr(inner, attr, None) is not None:
                        try:
                            delattr(inner, attr)
                            setattr(inner, attr, None)
                        except Exception:
                            pass
                try:
                    del diarizer.diarizer
                except Exception:
                    pass
                diarizer.diarizer = None
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
    def merge_diarization_with_transcription(
        self, 
        diarization_results: List[Dict], 
        transcription_result: Dict
    ) -> List[Dict]:
        """Merge diarization and transcription results."""
        try:
            self.logger.info("Starting to merge diarization and transcription results...")
            self.logger.info(f"Diarization segments: {len(diarization_results)}")
            self.logger.info(f"Transcription segments: {len(transcription_result['segments'])}")
            
            merged_results = []
            unmatched_segments = 0
            multiple_speakers = 0

            # E: index diarizace pro O(log D + K log K) lookup místo O(D) scanu.
            # Podmínka překryvu max(s,ds) < min(e,de) <=> ds < e a s < de.
            # order[:hi] (bisect na starty) dá přesně ds < e, filtr de > s pak
            # přesně překryv. Setřídění kandidátů zpět do původního pořadí
            # zachovává i tie-break v max() níž (bitová shoda se starým kódem).
            import bisect
            order = sorted(
                range(len(diarization_results)),
                key=lambda k: diarization_results[k]["start"],
            )
            diar_starts = [diarization_results[k]["start"] for k in order]
            by_speaker: Dict[str, List[Dict]] = {}
            for d in diarization_results:
                by_speaker.setdefault(d["speaker"], []).append(d)
            
            # Process segments in sequential order
            prev_segment_text = None
            
            for segment in transcription_result["segments"]:
                segment_start = segment["start"]
                segment_end = segment["end"]
                segment_text = segment["text"].strip()
                
                # Skip near-duplicate segments
                if prev_segment_text and self._is_similar_text(segment_text, prev_segment_text):
                    self.logger.debug(f"Skipping duplicate segment text: '{segment_text}'")
                    continue
                
                prev_segment_text = segment_text
                
                # Find matching speakers (překryv s [segment_start, segment_end))
                hi = bisect.bisect_left(diar_starts, segment_end)
                cand = sorted(
                    k for k in order[:hi]
                    if diarization_results[k]["end"] > segment_start
                )
                matching_speakers = [diarization_results[k]["speaker"] for k in cand]

                if not matching_speakers:
                    unmatched_segments += 1
                    speaker = "UNKNOWN"
                elif len(matching_speakers) > 1:
                    multiple_speakers += 1
                    # Use the speaker with the longest overlap.
                    # Součet jen přes segmenty daného speakera (včetně záporných
                    # příspěvků nepřekrývajících se) = identický výsledek jako
                    # dřívější součet přes celé diarization_results.
                    speaker = max(
                        matching_speakers,
                        key=lambda s: sum(
                            min(segment_end, d["end"]) - max(segment_start, d["start"])
                            for d in by_speaker[s]
                        )
                    )
                else:
                    speaker = matching_speakers[0]
                
                # Skip empty or very short segments (less than 0.5 seconds with no useful content)
                if not segment_text or (segment_end - segment_start < 0.5 and len(segment_text.split()) <= 1):
                    continue
                    
                # Skip segments with repetitive content
                if self._is_repetitive_content(segment_text):
                    self.logger.debug(f"Skipping repetitive segment: '{segment_text}'")
                    continue
                    
                merged_results.append({
                    "start": segment_start,
                    "end": segment_end,
                    "speaker": speaker,
                    "text": segment_text,
                    "duration": segment_end - segment_start
                })
            
            # Log merge statistics
            self.logger.info("Merge Results Summary:")
            self.logger.info(f"Total merged segments: {len(merged_results)}")
            self.logger.info(f"Unmatched segments: {unmatched_segments}")
            self.logger.info(f"Multiple speaker segments: {multiple_speakers}")
            
            # Log speaker statistics
            speaker_stats = {}
            for segment in merged_results:
                speaker = segment["speaker"]
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {
                        "count": 0,
                        "total_duration": 0,
                        "total_words": 0
                    }
                speaker_stats[speaker]["count"] += 1
                speaker_stats[speaker]["total_duration"] += segment["duration"]
                speaker_stats[speaker]["total_words"] += len(segment["text"].split())
            
            for speaker, stats in speaker_stats.items():
                self.logger.debug(f"Speaker {speaker} statistics:")
                self.logger.debug(f"  - Segments: {stats['count']}")
                self.logger.debug(f"  - Total duration: {stats['total_duration']:.2f} seconds")
                self.logger.debug(f"  - Total words: {stats['total_words']}")
                self.logger.debug(f"  - Average words per segment: {stats['total_words']/stats['count']:.2f}")
            
            return merged_results
            
        except Exception as e:
            self.logger.error(f"Error merging results: {str(e)}")
            raise
            
    def _is_similar_text(self, text1: str, text2: str) -> bool:
        """Check if two text segments are similar enough to be considered duplicates."""
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # If either text is very short, use direct comparison
        if len(text1) < 10 or len(text2) < 10:
            return text1 == text2
            
        # If texts are identical, they're obviously duplicates
        if text1 == text2:
            return True
            
        # Check for significant overlap (more than 70% of shorter text)
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        overlap = words1.intersection(words2)
        shorter_len = min(len(words1), len(words2))
        
        if shorter_len == 0:
            return False
        
        overlap_ratio = len(overlap) / shorter_len
        return overlap_ratio > 0.7

    def _is_repetitive_content(self, text: str) -> bool:
        """Check if a segment contains repetitive content that should be filtered out."""
        text = text.lower().strip()
        
        # If text is very short, don't consider it repetitive
        if len(text) < 10:
            return False
            
        # Check for repeated phrases
        words = text.split()
        if len(words) <= 1:
            return False
            
        # Check if the same phrase is repeated multiple times
        if len(set(words)) == 1:
            return True
            
        # Check for repeating patterns
        # Example: "I am a native, I am a native, I am a native"
        for phrase_length in range(2, min(5, len(words) // 2 + 1)):
            for start in range(0, len(words) - phrase_length * 2 + 1):
                phrase1 = " ".join(words[start:start + phrase_length])
                phrase2 = " ".join(words[start + phrase_length:start + phrase_length * 2])
                if phrase1 == phrase2:
                    return True
                    
        return False

    def _log_memory_summary(self, prefix: str) -> None:
        """torch.cuda.memory_summary() do debug logu (FÁZE 4 VRAM management)."""
        try:
            import torch as _torch
            if not _torch.cuda.is_available():
                return
            self.logger.debug(
                f"CUDA memory_summary [{prefix}]:\n"
                + _torch.cuda.memory_summary(abbreviated=True)
            )
        except Exception as e:
            self.logger.debug(f"memory_summary nedostupné: {e}")

    def _transcribe_and_diarize_parallel(
        self, audio_path: str
    ):
        """Transkripce + diarizace paralelně (nezávislé větve).

        Šířku řídí performance.num_workers (dřív ignorováno, L08):
        >= 2 = 2 vlákna, jinak sekvenční fallback ve stejném pořadí jako dřív.
        Transkripce běží v hlavním vlákně (CUDA kontext Whisperu),
        diarizace ve workeru (GIL se uvolňuje v torch/CUDA/ffmpeg).
        """
        num_workers = getattr(self.config.performance, "num_workers", 0) or 0
        if num_workers < 2:
            self.logger.info("Starting diarization...")
            diarization_results = self.diarizer.process(audio_path)
            self.logger.info("Starting transcription...")
            transcription_result = self.transcriber.transcribe(audio_path)
            return transcription_result, diarization_results

        from concurrent.futures import ThreadPoolExecutor
        self.logger.info("Starting diarization + transcription in parallel...")
        # Pozn.: při pádu transkripce `with` počká na doběh diarizace
        # (stejná sémantika jako sekvenčně: obě větve se pokusí doběhnout).
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="wvt-pipe") as pool:
            diar_future = pool.submit(self.diarizer.process, audio_path)
            transcription_result = self.transcriber.transcribe(audio_path)
            diarization_results = diar_future.result()
        return transcription_result, diarization_results

    def process_file(self, audio_path: str) -> List[Dict]:
        try:
            self.logger.info(f"Starting processing of {audio_path}")
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if not os.path.isfile(audio_path):
                raise ValueError(f"Path is not a file: {audio_path}")
            
            # Check file size
            file_size = os.path.getsize(audio_path)
            self.logger.info(f"Audio file size: {file_size / (1024*1024):.2f} MB")
            
            # Load and preprocess audio
            # Note: waveform se cíleně nedekóduje (load_waveform=False) — transkripce
            # i diarizace čtou audio z disku samy. Šetří ~460MB RAM u 2h nahrávky (L01).
            # FÁZE 4 stage order: STAGE 1 audio/VAD-gate -> STAGE 2 ASR+VAD chunky
            # (paralelně STAGE 2b diarizace) -> STAGE 3 merge -> STAGE 4 output.
            self.logger.info("STAGE 1/4: loading audio (VAD prefiltr běží ve splitu)...")
            _, sample_rate, converted_path = self.audio_processor.load_audio(
                audio_path, load_waveform=False
            )

            # Use converted path if available
            if converted_path:
                self.logger.info(f"Using converted WAV file: {converted_path}")
                audio_path = converted_path

            # Log audio properties
            self.logger.info(f"Audio ready: {audio_path}, Sample rate: {sample_rate}")
            
            try:
                import torch as _torch
                cuda_ok = _torch.cuda.is_available()
            except Exception:
                cuda_ok = False
            if cuda_ok and self.config.performance.device == "cuda":
                self.logger.info("CUDA is available and will be used")
            else:
                self.logger.warning("CUDA is not available, using CPU")
            
            # STAGE 2: ASR (VAD-gatované 30s chunky, model 1x) + STAGE 2b diarizace.
            # Nezávislé větve běží paralelně, viz _transcribe_and_diarize_parallel.
            diarization_results = []
            if self.config.diarization.enabled:
                self.logger.info("STAGE 2/4: ASR + diarization (parallel)...")
                transcription_result, diarization_results = (
                    self._transcribe_and_diarize_parallel(audio_path)
                )
                self.logger.info(f"Diarization completed: {len(diarization_results)} segments")
                self.logger.info("Transcription completed")
            else:
                # Perform transcription only
                self.logger.info("STAGE 2/4: transcription only...")
                transcription_result = self.transcriber.transcribe(audio_path)
                self.logger.info("Transcription completed")
            self._log_memory_summary("after-asr-diar")
            
            # Merge results if diarization is enabled
            if self.config.diarization.enabled:
                self.logger.info("STAGE 3/4: merging diarization and transcription...")
                final_results = self.merge_diarization_with_transcription(
                    diarization_results,
                    transcription_result
                )
            else:
                self.logger.info("STAGE 3/4: results without diarization...")
                final_results = [{
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": "UNKNOWN",
                    "text": segment["text"].strip()
                } for segment in transcription_result["segments"]]
            self._log_memory_summary("after-merge")
            
            self.logger.info(f"Successfully processed {audio_path}")
            self.logger.info(f"Generated {len(final_results)} segments")
            
            return final_results
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found error: {str(e)}")
            raise
        except ValueError as e:
            self.logger.error(f"Value error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}", exc_info=True)
            raise
        finally:
            # STAGE 4/4: cleanup po KAŽDÉM souboru (i při pádu): temp wav/chunky
            # (fix orphan m4a konverzí) + CUDA cache. Modely schválně zůstávají
            # nahrané pro rychlý další soubor (uvolnění jen přes release_*).
            try:
                self.audio_processor.cleanup()
            except Exception:
                pass
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass