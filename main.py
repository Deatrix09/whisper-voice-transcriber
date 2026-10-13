# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
import json
import os
import sys
from typing import List, Dict

from src.utils.logger import setup_logger
from src.utils.formatters import OutputFormatter
from src.utils.path_utils import PathUtils
from src.utils.log_cleanup import LogCleanup

# Pozn. (F): torch / pyannote / Pipeline / AppConfig se importují až v main()
# za parse_arguments, aby --help a chyba v CLI neplatily cold start
# těžkých závislostí (torch má přes 1 s, whisper/pyannote další).



def parse_arguments():

    parser = argparse.ArgumentParser(description="Audio Transcription with Speaker Diarization")

    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")

    parser.add_argument("--input", type=str, required=False, help="Path to input audio file (povinný kromě --validate)")

    parser.add_argument(
        "--validate",
        action="store_true",
        help="P0 0.3: jen ověř config (tisk + exit 0/2), bez transkripce"
    )

    parser.add_argument(

        "--format", 

        type=str, 

        choices=["json", "txt", "subtitle"],

        help="Output format (overrides config file setting)"

    )

    return parser.parse_args()



def validate_config_file(path: str) -> int:
    """P0 0.3: ověř config bez těžkých importů. Vrací exit kód (0 ok, 2 chyba)."""
    from src.utils.config import AppConfig, ConfigError
    try:
        warnings = AppConfig.from_yaml(path).validate()
    except ConfigError as e:
        print(f"NEPLATNÝ config: {e}")
        return 2
    print(f"Config {path}: OK")
    for w in warnings:
        print(f"  warning: {w}")
    return 0


def main():

    # Parse command line arguments first so --help works without torch.
    args = parse_arguments()

    if args.validate:
        raise SystemExit(validate_config_file(args.config))

    if not args.input:
        print("main.py: error: --input je povinný (nebo použij --validate --config ...)",
              file=sys.stderr)
        raise SystemExit(2)

    # P0 0.1: lokální .env (HF_TOKEN) - před čtením configu.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Heavy imports lazy (F) - až po parsingu argumentů.
    import torch
    from torch.serialization import add_safe_globals
    from torch.torch_version import TorchVersion
    try:
        # Jen pro pyannote větev (safe_globals pro torch.load 2.6+).
        # Sortformer/nemotron-only běh pyannote nepotřebuje. Široký except:
        # rozbitá kombinace pyannote/torchaudio (např. chybějící AudioMetaData)
        # nesmí shodit non-pyannote běhy.
        from pyannote.audio.core.task import Specifications, Problem, Resolution
        add_safe_globals([TorchVersion, Specifications, Problem, Resolution])
    except Exception:
        pass
    from src.utils.config import AppConfig
    from src.core.pipeline import Pipeline

    # Load configuration
    config = AppConfig.from_yaml(args.config)

    # Setup logging directory
    logs_dir = PathUtils.ensure_directory_exists(os.path.join(os.getcwd(), 'logs'))

    
    # Setup logger
    logger = setup_logger("main", config.logging.level, config.logging.file,
                          file_level=getattr(config.logging, "file_level", None),
                          console_level=getattr(config.logging, "console_level", None))

    # P0 0.3: auto-validace configu (fatální = jasná chyba + exit 2, warnings do logu)
    from src.utils.config import ConfigError
    try:
        for w in config.validate():
            logger.warning(f"config: {w}")
    except ConfigError as e:
        logger.error(f"NEPLATNÝ config: {e}")
        raise SystemExit(2)

    

    try:
        # Get output format
        output_format = args.format if args.format else config.output.format

        # Validate and prepare paths (output_path will be generated automatically)
        input_path, output_path = PathUtils.validate_and_prepare_paths(

            args.input,

            output_format=output_format

        )

        

        log_cleanup = LogCleanup(logs_dir, config.logging.retention_days)

        log_cleanup.cleanup_old_logs()

        

        # Get log statistics
        log_stats = log_cleanup.get_log_stats()

        logger.info("Log Statistics:")

        logger.info(f"Total logs: {log_stats['total_logs']}")

        logger.info(f"Total size: {log_stats['total_size'] / 1024:.2f} KB")

        if log_stats['oldest_log']:

            logger.info(f"Oldest log: {log_stats['oldest_log']}")

        if log_stats['newest_log']:

            logger.info(f"Newest log: {log_stats['newest_log']}")

        # Log system information
        logger.info("Starting transcription pipeline")

        logger.info(f"Input file: {input_path}")

        logger.info(f"Output file: {output_path}")

        logger.info(f"Using model: {config.model.name}")

        logger.info(f"Device: {config.performance.device}")

        

        # Initialize pipeline
        pipeline = Pipeline(config)

        

        # Process audio file
        logger.info(f"Processing file: {input_path}")

        results = pipeline.process_file(input_path)

        

        # Save results in specified format
        formatter = OutputFormatter()

        formatter.save_output(results, output_path, output_format)

        logger.info(f"Results saved to: {output_path} in {output_format} format")

        

        # Log recent logs

        recent_logs = log_cleanup.get_recent_logs()

        if recent_logs:

            logger.info("Recent log files:")

            for log_file in recent_logs[:5]:

                logger.info(f"  - {log_file}")

        

    except Exception as e:

        logger.error(f"Processing failed: {str(e)}", exc_info=True)

        raise

    finally:

        logger.info("Pipeline execution completed")



if __name__ == "__main__":

    main() 














