"""P0 0.4: factory ASR providerů (faster-whisper | parakeet | auto).

Eager init zvoleného enginu + graceful fallback na faster-whisper.
Vrací inicializovaný ASRProvider (model 1x, žádný reload mezi chunky).
"""
import logging

from src.transcription.base_transcriber import ASRProvider


def create_asr_provider(config) -> ASRProvider:
    from src.transcription.parakeet_transcriber import (
        ParakeetTranscriber, supports_language)
    from src.transcription.transcriber import Transcriber
    logger = logging.getLogger("asr_factory")
    asr = getattr(config, "asr", None)
    engine = (getattr(asr, "engine", "") or "faster-whisper").strip().lower()
    lang = (config.model.language or "").strip().lower()

    use_parakeet = (engine == "parakeet") or \
        (engine == "auto" and supports_language(lang))
    if engine == "auto":
        logger.info(f"ASR router: engine=auto, lang={lang or 'auto-detect'} -> "
                    f"{'parakeet' if use_parakeet else 'faster-whisper'}")
    if use_parakeet:
        try:
            transcriber = ParakeetTranscriber(config)
            transcriber.initialize_model()
            return transcriber
        except Exception as e:
            logger.warning(f"Parakeet init selhal ({e}), fallback faster-whisper-turbo")

    transcriber = Transcriber(config)
    transcriber.initialize_model()
    return transcriber
