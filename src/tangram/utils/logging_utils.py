"""
Logging utilities for the TANGRAM pipeline.

Provides consistent logging setup and function call tracking across all modules.
"""

import logging
import functools
import time
from pathlib import Path
from typing import Any, Callable
from config import LOGS_DIR, LOGGING_CONFIG

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path. If None, uses console only.
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, LOGGING_CONFIG["level"]))
    
    # Create formatter
    formatter = logging.Formatter(LOGGING_CONFIG["format"])
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = LOGS_DIR / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_function_call(logger: logging.Logger = None):
    """
    Decorator to log function calls with execution time.
    
    Args:
        logger: Logger instance. If None, creates a default logger.
        
    Usage:
        @log_function_call()
        def my_function(arg1, arg2):
            return result
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = setup_logger(func.__module__)
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            # Log function start
            logger.info(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"Completed {func.__name__} in {execution_time:.2f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {execution_time:.2f}s: {e}")
                raise
                
        return wrapper
    return decorator

def log_pipeline_step(step_name: str, logger: logging.Logger = None):
    """
    Decorator for logging pipeline steps with clear formatting.
    
    Args:
        step_name: Name of the pipeline step
        logger: Logger instance
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = setup_logger(func.__module__)
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger.info("=" * 60)
            logger.info(f"STEP: {step_name}")
            logger.info("=" * 60)
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"✓ {step_name} completed successfully in {execution_time:.2f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"✗ {step_name} failed after {execution_time:.2f}s: {e}")
                raise
                
        return wrapper
    return decorator