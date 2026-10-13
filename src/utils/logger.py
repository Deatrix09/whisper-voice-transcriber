# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
import os
from datetime import datetime
from typing import Optional

# Global dictionary to track initialized loggers
_initialized_loggers = {}
# Global variable to store current log file path
_current_log_file = None

def setup_logging_directory():
    """Create logs directory if it doesn't exist."""
    logs_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    return logs_dir

def get_log_filename(base_filename: str) -> str:
    """Generate a log filename with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_filename}_{timestamp}.log"

def _coerce_level(value: Optional[str], default: int) -> int:
    """Převeď 'INFO'/'DEBUG'/... na logging level, jinak default."""
    if not value:
        return default
    level = getattr(logging, str(value).upper(), None)
    return level if isinstance(level, int) else default


def setup_logger(
    name: str,
    level: str,
    log_file: Optional[str] = None,
    file_level: Optional[str] = None,
    console_level: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a logger instance with both file and console handlers.

    file_level/console_level jsou volitelné (default INFO konzole / DEBUG soubor
    = původní chování). Volající předává config.logging.file_level/console_level.
    """
    global _initialized_loggers, _current_log_file
    
    # Check if this is a new execution or continuation
    if name in _initialized_loggers:
        return _initialized_loggers[name]
        
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        _initialized_loggers[name] = logger
        return logger
        
    logger.setLevel(level.upper())

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(_coerce_level(console_level, logging.INFO))
    logger.addHandler(console_handler)

    # File handler - reuse the same log file for the entire execution
    if log_file:
        logs_dir = setup_logging_directory()
        
        # If this is the first logger being set up, generate the log file path
        if _current_log_file is None:
            _current_log_file = os.path.join(logs_dir, get_log_filename(log_file))
            is_first_logger = True
        else:
            is_first_logger = False

        file_handler = logging.FileHandler(_current_log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(_coerce_level(file_level, logging.DEBUG))
        logger.addHandler(file_handler)
        
        if is_first_logger:
            logger.info(f"Logging initialized. Log file: {_current_log_file}")

    # Store in initialized loggers dictionary
    _initialized_loggers[name] = logger
    
    return logger

def get_current_log_file() -> Optional[str]:
    """Return the current log file path being used."""
    global _current_log_file
    return _current_log_file
