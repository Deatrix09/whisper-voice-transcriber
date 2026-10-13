from typing import Optional
from src.utils.logger import setup_logger
from src.diarization.base_diarizer import BaseDiarizer
from src.utils.config import AppConfig

class DiarizationFactory:
        
    @staticmethod
    def create_diarizer(config: AppConfig) -> Optional[BaseDiarizer]:
        """
        Create and return appropriate diarizer based on configuration
        
        Args:
            config: Application configuration
            
        Returns:
            A diarizer instance or None if diarization is disabled
        """
        # Use the same logger as the rest of the app
        logger = setup_logger(__name__, config.logging.level, config.logging.file,
                                file_level=getattr(config.logging, "file_level", None),
                                console_level=getattr(config.logging, "console_level", None))
        
        if not config.diarization.enabled:
            logger.info("Diarization is disabled in configuration")
            return None
        
        # Create appropriate diarizer based on module name
        module = config.diarization.module.lower()
        
        if module == "pyannote":
            # Check if PyAnnote is configured
            if not hasattr(config.diarization, 'pyannote') or not config.diarization.pyannote.enabled:
                logger.error("PyAnnote module selected but not properly configured")
                raise ValueError("PyAnnote module selected but not properly configured")
                
            logger.info("Creating PyAnnote diarizer")
            from src.diarization.pyannote_diarizer import PyAnnoteDiarizer
            return PyAnnoteDiarizer(config)
            
        elif module == "sortformer":
            # NeMo Sortformer (end-to-end, offline). NeMo se importuje lazy
            # až v initialize() adapteru, tady jen konstrukce.
            logger.info("Creating Sortformer diarizer")
            from src.diarization.sortformer_diarizer import SortformerDiarizer
            return SortformerDiarizer(config)

        elif module == "nemotron-3":
            # NVIDIA Nemotron-3-Diarization přes 🤗 transformers (ne NeMo!).
            # transformers se importuje lazy až v initialize() adapteru.
            logger.info("Creating Nemotron-3 diarizer")
            from src.diarization.nemotron_diarizer import NemotronDiarizer
            return NemotronDiarizer(config)

        elif module == "nemo":
            # Check if NeMo is configured
            if not hasattr(config.diarization, 'nemo') or not config.diarization.nemo.enabled:
                logger.error("NeMo module selected but not properly configured")
                raise ValueError("NeMo module selected but not properly configured")
                
            logger.info("Creating NeMo diarizer")
            try:
                import nemo.collections.asr as nemo_asr
                from src.diarization.nemo_diarizer import NemoDiarizer
                return NemoDiarizer(config)
            except ImportError:
                logger.error("Failed to import NeMo. Is it properly installed?")
                logger.error("Please run 'python scripts/setup_nemo.py' to install NeMo")
                raise ImportError("Failed to import NeMo. Please run the setup script.")
                
        else:
            logger.error(f"Unknown diarization module: {module}")
            raise ValueError(f"Unknown diarization module: {module}")
