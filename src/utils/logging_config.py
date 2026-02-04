"""
Logging configuration.

Setup logging for the project.
"""

import logging
import logging.handlers
from pathlib import Path


def setup_logging(
    log_dir: str = 'logs',
    log_level: int = logging.INFO,
    log_file: str = 'plant_disease_detection.log'
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save logs
        log_level: Logging level
        log_file: Log file name
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)


# Setup logging on import
setup_logging()
