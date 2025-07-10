"""
TANGRAM Simplified Configuration

Essential configuration parameters only.
Reduced from 200+ lines to core settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Core directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Essential subdirectories
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "frames"
TRACKING_DIR = DATA_DIR / "tracking"
GRAPHS_DIR = DATA_DIR / "graphs"

YOLO_MODELS_DIR = MODELS_DIR / "yolo"

# Core YOLO Configuration
YOLO_CONFIG = {
    "model_name": "yolov8n.pt",  # Nano model for speed
    "confidence_threshold": 0.5,
    "device": "mps",  # Apple Silicon GPU
}

# LLM Configuration 
LLM_CONFIG = {
    "host": "localhost",
    "port": 11434,
    "model": "deepseek-r1:7b",
    "temperature": 0.1,
    "timeout": 30
}

# Robot Configuration
ROBOT_CONFIG = {
    "gui": False,  # Set to True for PyBullet GUI
    "workspace_bounds": {
        "x": [0, 8],
        "y": [0, 6], 
        "z": [0, 4]
    },
    "home_position": [0, 3, 1.5]
}

# Hardware Configuration
HARDWARE_CONFIG = {
    "device": "mps",  # Apple Silicon
    "num_workers": 4,
    "memory_limit_gb": 8
}

def ensure_directories():
    """Create required directories if they don't exist."""
    directories = [
        DATA_DIR, RAW_VIDEOS_DIR, FRAMES_DIR, TRACKING_DIR, GRAPHS_DIR,
        MODELS_DIR, YOLO_MODELS_DIR, RESULTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_api_key(service: str) -> str:
    """Get API key for specified service from environment variables."""
    env_keys = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "google": "GOOGLE_API_KEY"
    }
    
    key = os.getenv(env_keys.get(service))
    if not key:
        print(f"Warning: {env_keys.get(service)} not found in environment variables")
    
    return key

# Initialize directories on import
ensure_directories()