"""
File handling utilities for the TANGRAM pipeline.

Provides consistent file I/O operations with error handling and validation.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union
import logging

logger = logging.getLogger(__name__)

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object of the created directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def save_json(data: Union[Dict, List], file_path: Union[str, Path], 
              indent: int = 2, ensure_dir: bool = True) -> bool:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save (dict or list)
        file_path: Output file path
        indent: JSON indentation level
        ensure_dir: Whether to create parent directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        
        if ensure_dir:
            ensure_directory(file_path.parent)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            
        logger.debug(f"Saved JSON data to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        return False

def load_json(file_path: Union[str, Path]) -> Union[Dict, List, None]:
    """
    Load data from JSON file with error handling.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data or None if failed
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"JSON file not found: {file_path}")
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        logger.debug(f"Loaded JSON data from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return None

def validate_file_exists(file_path: Union[str, Path], 
                        file_type: str = "file") -> bool:
    """
    Validate that a file exists and is accessible.
    
    Args:
        file_path: Path to validate
        file_type: Type description for logging
        
    Returns:
        True if file exists and is accessible
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"{file_type} not found: {file_path}")
        return False
        
    if not file_path.is_file():
        logger.error(f"Path is not a file: {file_path}")
        return False
        
    if not os.access(file_path, os.R_OK):
        logger.error(f"File not readable: {file_path}")
        return False
        
    return True

def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB, or 0 if file doesn't exist
    """
    try:
        file_path = Path(file_path)
        if file_path.exists():
            return file_path.stat().st_size / (1024 * 1024)
        return 0.0
    except Exception:
        return 0.0

def clean_filename(filename: str) -> str:
    """
    Clean filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename safe for filesystem
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()

def list_files_by_extension(directory: Union[str, Path], 
                           extension: str) -> List[Path]:
    """
    List all files with specific extension in directory.
    
    Args:
        directory: Directory to search
        extension: File extension (e.g., '.mp4', '.json')
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
        
    if not extension.startswith('.'):
        extension = '.' + extension
        
    return list(directory.glob(f"*{extension}"))

def backup_file(file_path: Union[str, Path], 
                backup_suffix: str = ".backup") -> bool:
    """
    Create a backup copy of a file.
    
    Args:
        file_path: Original file path
        backup_suffix: Suffix to add to backup filename
        
    Returns:
        True if backup created successfully
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Cannot backup non-existent file: {file_path}")
            return False
            
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        
        import shutil
        shutil.copy2(file_path, backup_path)
        
        logger.info(f"Created backup: {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create backup of {file_path}: {e}")
        return False