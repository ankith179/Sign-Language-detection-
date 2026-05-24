"""Logging utilities for the Sign Language Recognition System."""

import logging
import os
from datetime import datetime


def get_logger(name, log_dir='logs', level=logging.INFO):
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        log_dir: Directory to save logs
        level: Logging level
    
    Returns:
        Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def log_config(logger, config_dict):
    """Log configuration parameters."""
    logger.info("="*70)
    logger.info("Configuration:")
    logger.info("="*70)
    
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("="*70)


def log_metrics(logger, metrics_dict):
    """Log evaluation metrics."""
    logger.info("="*70)
    logger.info("Evaluation Metrics:")
    logger.info("="*70)
    
    for key, value in metrics_dict.items():
        if key != 'report':
            logger.info(f"  {key}: {value:.4f}")
    
    logger.info("="*70)
