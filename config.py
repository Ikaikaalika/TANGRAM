"""
TANGRAM Configuration Module

This module contains all configuration parameters for the robotic scene understanding pipeline.
Modify these settings to customize the behavior of different pipeline components.

Author: TANGRAM Team
License: MIT
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"
FRAMES_DIR = DATA_DIR / "frames"
TRACKING_DIR = DATA_DIR / "tracking"
MASKS_DIR = DATA_DIR / "masks"
RECONSTRUCTION_DIR = DATA_DIR / "3d_points"
GRAPHS_DIR = DATA_DIR / "graphs"
SIMULATION_DIR = DATA_DIR / "simulation"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
SAM_MODELS_DIR = MODELS_DIR / "sam"
YOLO_MODELS_DIR = MODELS_DIR / "yolo"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results"
EXPORTS_DIR = RESULTS_DIR / "exports"
VIDEOS_DIR = RESULTS_DIR / "videos"
LOGS_DIR = RESULTS_DIR / "logs"

# YOLO Configuration
YOLO_CONFIG = {
    "model_name": "yolov8n.pt",  # Use nano model for faster inference on M1
    "confidence_threshold": 0.5,
    "iou_threshold": 0.7,
    "max_detections": 50,
    "device": "mps",  # Apple Silicon GPU acceleration
    "tracker_config": "bytetrack.yaml"
}

# SAM Configuration
SAM_CONFIG = {
    "model_type": "vit_b",  # vit_b, vit_l, or vit_h (balance speed vs accuracy)
    "checkpoint_url": {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    },
    "device": "mps",  # Apple Silicon GPU acceleration
    "points_per_side": 32,  # For automatic mask generation
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95
}

# COLMAP Configuration
COLMAP_CONFIG = {
    "feature_extractor": {
        "camera_model": "PINHOLE",
        "single_camera": True,
        "max_image_size": 1600
    },
    "matcher": {
        "guided_matching": True,
        "max_ratio": 0.8
    },
    "mapper": {
        "ba_refine_focal_length": False,
        "ba_refine_principal_point": False,
        "min_num_matches": 15
    },
    "dense_reconstruction": {
        "max_image_size": 1000,
        "enable_dense": True  # Set to False to skip dense reconstruction for speed
    }
}

# Scene Graph Configuration
SCENE_GRAPH_CONFIG = {
    "spatial_thresholds": {
        "near": 0.5,      # meters
        "touching": 0.1,   # meters
        "on_surface": 0.05  # height difference for "on" relationship
    },
    "temporal_thresholds": {
        "interaction_frames": 5,  # Minimum frames for interaction
        "bbox_overlap_threshold": 0.1
    },
    "scene_classification": {
        "kitchen": ["cup", "bottle", "bowl", "spoon", "knife", "plate"],
        "office": ["book", "laptop", "mouse", "keyboard", "pen"],
        "bedroom": ["bed", "pillow", "clock"]
    }
}

# LLM Configuration
LLM_CONFIG = {
    "provider": "deepseek",  # Currently supports: deepseek, openai
    "model": "deepseek-reasoner",
    "base_url": "https://api.deepseek.com",
    "api_key_env": "DEEPSEEK_API_KEY",
    "temperature": 0.1,
    "max_tokens": 2000,
    "timeout": 30  # seconds
}

# PyBullet Configuration
PYBULLET_CONFIG = {
    "gui": True,  # Set to False for headless simulation
    "gravity": -9.81,
    "time_step": 1.0/240.0,  # Simulation time step
    "robot_urdf": "assets/robots/ur5e/ur5e.urdf",  # Robot model
    "workspace_bounds": {
        "x": [-0.8, 0.8],
        "y": [-0.8, 0.8], 
        "z": [0.0, 1.0]
    },
    "simulation_speed": 1.0,  # Real-time multiplier
    "max_simulation_time": 300  # Maximum simulation time in seconds
}

# Frame Extraction Configuration
FRAME_EXTRACTION_CONFIG = {
    "interval": 5,  # Extract every Nth frame
    "max_frames": 200,  # Maximum frames to extract
    "format": "jpg",
    "quality": 90  # JPEG quality (1-100)
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "node_size": 1500,
    "font_size": 10,
    "edge_width": 2,
    "layout": "spring",  # spring, circular, kamada_kawai
    "save_formats": ["png", "pdf"]
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_rotation": "daily",
    "max_log_files": 7
}

# Hardware Configuration
HARDWARE_CONFIG = {
    "use_gpu": True,
    "gpu_device": "mps",  # For Apple Silicon
    "num_workers": 4,  # For multiprocessing
    "memory_limit_gb": 8,  # Memory usage limit
    "thunder_compute": {
        "enabled": False,  # Set to True when using Thunder Compute
        "ssh_host": None,  # Thunder Compute SSH host
        "ssh_user": None,  # SSH username
        "remote_data_dir": "/tmp/tangram_data"
    }
}

def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR, RAW_VIDEOS_DIR, SAMPLE_VIDEOS_DIR, FRAMES_DIR,
        TRACKING_DIR, MASKS_DIR, RECONSTRUCTION_DIR, GRAPHS_DIR,
        SIMULATION_DIR, MODELS_DIR, SAM_MODELS_DIR, YOLO_MODELS_DIR,
        RESULTS_DIR, EXPORTS_DIR, VIDEOS_DIR, LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_api_key(service: str) -> str:
    """Get API key for specified service from environment variables."""
    env_keys = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY"
    }
    
    key = os.getenv(env_keys.get(service))
    if not key:
        print(f"Warning: {env_keys.get(service)} not found in environment variables")
    
    return key

# Initialize directories on import
ensure_directories()