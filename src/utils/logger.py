import logging

import os

from datetime import datetime

from typing import Optional



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



def setup_logger(name: str, level: str, log_file: Optional[str] = None) -> logging.Logger:

    """Configure and return a logger instance with both file and console handlers."""
    logger = logging.getLogger(name)


    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

        
    logger.setLevel(level.upper())

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'

    )

    

    # Console handler
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    console_handler.setLevel(logging.INFO)  # Console shows INFO and above

    logger.addHandler(console_handler)

    

    # File handler
    if log_file:

        logs_dir = setup_logging_directory()

        log_path = os.path.join(logs_dir, get_log_filename(log_file))

        

        file_handler = logging.FileHandler(log_path, encoding='utf-8')

        file_handler.setFormatter(formatter)

        file_handler.setLevel(logging.DEBUG)  # File logs everything

        logger.addHandler(file_handler)

        

        # Log the initialization
        logger.info(f"Logging initialized. Log file: {log_path}")

    

    return logger 
