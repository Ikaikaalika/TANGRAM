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

# Data directories - New logical structure
DATA_DIR = PROJECT_ROOT / "data"

# Input directories
INPUTS_DIR = DATA_DIR / "inputs"
MEDIA_DIR = INPUTS_DIR / "media"  # Formerly raw_videos
SAMPLES_DIR = INPUTS_DIR / "samples"  # Formerly sample_videos
DEMOS_DIR = INPUTS_DIR / "demos"

# Processing directories
PROCESSING_DIR = DATA_DIR / "processing"
FRAMES_DIR = PROCESSING_DIR / "frames"
TRACKING_DIR = PROCESSING_DIR / "tracking"
MASKS_DIR = PROCESSING_DIR / "segmentation"  # Formerly masks
COLMAP_DIR = PROCESSING_DIR / "colmap_reconstruction"  # Formerly 3d_reconstruction_*

# Output directories
OUTPUTS_DIR = DATA_DIR / "outputs"
RECONSTRUCTION_DIR = OUTPUTS_DIR / "point_clouds"  # Formerly 3d_points
GRAPHS_DIR = OUTPUTS_DIR / "scene_graphs"  # Formerly graphs
SIMULATION_DIR = OUTPUTS_DIR / "simulation_results"  # Formerly simulation
EXPORTS_DIR = OUTPUTS_DIR / "exports"

# Cache directories
CACHE_DIR = DATA_DIR / "cache"
TEMP_DIR = CACHE_DIR / "temp"
SESSIONS_DIR = CACHE_DIR / "sessions"

# Backward compatibility aliases
RAW_VIDEOS_DIR = MEDIA_DIR
SAMPLE_VIDEOS_DIR = SAMPLES_DIR

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
SAM_MODELS_DIR = MODELS_DIR / "sam"
YOLO_MODELS_DIR = MODELS_DIR / "yolo"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results"
EXPORTS_DIR = RESULTS_DIR / "exports"
VIDEOS_DIR = RESULTS_DIR / "videos"
LOGS_DIR = RESULTS_DIR / "logs"

# YOLO Configuration - Updated to YOLOv11
YOLO_CONFIG = {
    "model_name": "yolo11x.pt",  # YOLOv11 extra-large - latest and most accurate
    "model_fallbacks": ["yolo11l.pt", "yolo11m.pt", "yolo11s.pt", "yolo11n.pt", "yolov8x.pt"],  # Fallback models
    "confidence_threshold": 0.25,  # Optimized for YOLOv11's improved accuracy
    "iou_threshold": 0.7,
    "max_detections": 100,  # More detections for photos
    "device": "mps",  # Apple Silicon GPU acceleration
    "tracker_config": "bytetrack.yaml",
    "imgsz": 1280,  # Higher resolution for better photo detection
    "agnostic_nms": True,  # YOLOv11 feature for better multi-class detection
    "half": True  # Use FP16 for faster inference on Apple Silicon
}

# SAM Configuration
SAM_CONFIG = {
    "model_type": "vit_b",  # vit_b, vit_l, or vit_h (balance speed vs accuracy)
    "checkpoint_url": {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    },
    "device": "cpu",  # Use CPU for stability (SAM has MPS issues on Apple Silicon)
    "points_per_side": 16,  # Reduced for memory efficiency
    "pred_iou_thresh": 0.86,
    "stability_score_thresh": 0.92
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

# LLM Configuration - Support for multiple backends
LLM_CONFIG = {
    "provider": "auto",  # auto, huggingface, ollama, gemini
    "prompt_for_api_key": True,  # Ask for API key at start if needed
    "prefer_local": False,  # Prefer cloud APIs for faster startup
    
    # Hugging Face (preferred local option)
    "huggingface": {
        "enabled": True,
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  # DeepSeek R1 7B
        "device": "mps",  # Apple Silicon GPU acceleration
        "torch_dtype": "float16",
        "max_new_tokens": 2000,
        "temperature": 0.1,
        "do_sample": True,
        "timeout": 180
    },
    
    # Ollama (fallback local option)
    "ollama": {
        "enabled": True,
        "host": "localhost",
        "port": 11434,
        "model": "deepseek-r1:7b",  # Use 7B for M1 Mac with 8GB RAM
        "timeout": 180
    },
    
    # Google Gemini API (cloud fallback)
    "gemini": {
        "enabled": True,
        "model": "gemini-1.5-flash",  # gemini-1.5-flash, gemini-1.5-pro
        "api_key_env": "GOOGLE_API_KEY",  # Environment variable for API key
        "timeout": 30
    },
    
    # Global settings
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
    "level": "DEBUG",  # DEBUG for detailed debugging information
    "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    "file_rotation": "daily",
    "max_log_files": 7,
    "console_level": "INFO",  # Less verbose on console
    "file_level": "DEBUG"     # Full debug info in files
}

# Hardware Configuration
HARDWARE_CONFIG = {
    "use_gpu": True,
    "gpu_device": "mps",  # For Apple Silicon
    "num_workers": 4,  # For multiprocessing
    "memory_limit_gb": 8,  # Memory usage limit
    "thunder_compute": {
        "enabled": True,  # Set to True when using Thunder Compute
        "ssh_host": None,  # Will be set dynamically from tnr status
        "ssh_user": "ubuntu",  # Default Thunder Compute user
        "remote_data_dir": "/tmp/tangram_data",
        "cli_command": "tnr",  # Thunder Compute CLI command
        "auto_create_instance": True,  # Auto-create instance if none exists
        "instance_config": {
            "gpu": "a100",  # or "t4", "a100xl"
            "vcpus": 4,
            "mode": "prototyping"
        }
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