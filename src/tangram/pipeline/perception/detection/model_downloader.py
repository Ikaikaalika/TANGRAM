#!/usr/bin/env python3
"""
Model Downloader for TANGRAM

Downloads and manages pretrained models for YOLO and SAM.
Provides automated model downloading with progress tracking.

Author: TANGRAM Team
License: MIT
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Optional, Callable
import hashlib
import time
import logging

logger = logging.getLogger(__name__)

class ModelDownloader:
    """Downloads and manages pretrained models for TANGRAM"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model downloader
        
        Args:
            models_dir: Directory to store downloaded models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            # YOLO v8 models (auto-download via ultralytics)
            "yolo": {
                "yolov8n.pt": {"size": "nano", "params": "3.2M", "auto": True},
                "yolov8s.pt": {"size": "small", "params": "11.2M", "auto": True},
                "yolov8m.pt": {"size": "medium", "params": "25.9M", "auto": True},
                "yolov8l.pt": {"size": "large", "params": "43.7M", "auto": True},
                "yolov8x.pt": {"size": "extra-large", "params": "68.2M", "auto": True},
            },
            
            # SAM models (manual download)
            "sam": {
                "sam_vit_b_01ec64.pth": {
                    "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                    "size": "375MB",
                    "model_type": "vit_b",
                    "description": "SAM ViT-B encoder - balanced performance"
                },
                "sam_vit_l_0b3195.pth": {
                    "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    "size": "1.25GB",
                    "model_type": "vit_l",
                    "description": "SAM ViT-L encoder - higher accuracy"
                },
                "sam_vit_h_4b8939.pth": {
                    "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    "size": "2.56GB",
                    "model_type": "vit_h",
                    "description": "SAM ViT-H encoder - highest accuracy"
                }
            },
            
            # SAM 2 models (latest 2024)
            "sam2": {
                "sam2_hiera_tiny.pt": {
                    "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
                    "size": "38MB",
                    "model_type": "hiera_tiny",
                    "description": "SAM 2.1 Hiera Tiny - fastest"
                },
                "sam2_hiera_small.pt": {
                    "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
                    "size": "183MB",
                    "model_type": "hiera_small",
                    "description": "SAM 2.1 Hiera Small - balanced"
                },
                "sam2_hiera_base_plus.pt": {
                    "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
                    "size": "319MB",
                    "model_type": "hiera_base_plus",
                    "description": "SAM 2.1 Hiera Base Plus - good accuracy"
                },
                "sam2_hiera_large.pt": {
                    "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
                    "size": "899MB",
                    "model_type": "hiera_large",
                    "description": "SAM 2.1 Hiera Large - best accuracy"
                }
            }
        }
    
    def download_model(self, model_type: str, model_name: str, 
                      progress_callback: Optional[Callable] = None) -> Path:
        """
        Download a model if not already present
        
        Args:
            model_type: Type of model (yolo, sam, sam2)
            model_name: Name of the model file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the downloaded model
        """
        if model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if model_name not in self.model_configs[model_type]:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.model_configs[model_type][model_name]
        
        # Create model type directory
        model_dir = self.models_dir / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / model_name
        
        # Check if model already exists
        if model_path.exists():
            logger.info(f"Model already exists: {model_path}")
            return model_path
        
        # Handle auto-download models (YOLO)
        if model_config.get("auto", False):
            logger.info(f"YOLO model {model_name} will be auto-downloaded by ultralytics")
            return model_path
        
        # Download model
        url = model_config["url"]
        logger.info(f"Downloading {model_name} from {url}")
        
        try:
            self._download_with_progress(url, model_path, progress_callback)
            logger.info(f"Successfully downloaded: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            if model_path.exists():
                model_path.unlink()  # Remove partial download
            raise
    
    def _download_with_progress(self, url: str, output_path: Path, 
                              progress_callback: Optional[Callable] = None):
        """Download file with progress tracking"""
        
        def progress_hook(block_num: int, block_size: int, total_size: int):
            if progress_callback:
                downloaded = block_num * block_size
                progress = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
                progress_callback(progress, downloaded, total_size)
        
        try:
            urllib.request.urlretrieve(url, str(output_path), progress_hook)
        except urllib.error.URLError as e:
            raise Exception(f"Download failed: {e}")
    
    def list_available_models(self) -> Dict[str, Dict[str, Dict]]:
        """List all available models"""
        return self.model_configs
    
    def get_model_info(self, model_type: str, model_name: str) -> Dict:
        """Get information about a specific model"""
        if model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if model_name not in self.model_configs[model_type]:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.model_configs[model_type][model_name]
    
    def is_model_downloaded(self, model_type: str, model_name: str) -> bool:
        """Check if a model is already downloaded"""
        model_path = self.models_dir / model_type / model_name
        return model_path.exists()
    
    def get_model_path(self, model_type: str, model_name: str) -> Path:
        """Get the path where a model should be/is stored"""
        return self.models_dir / model_type / model_name
    
    def download_recommended_models(self, progress_callback: Optional[Callable] = None):
        """Download recommended models for TANGRAM"""
        recommended = [
            ("yolo", "yolov8n.pt"),  # Fast YOLO for real-time detection
            ("sam", "sam_vit_b_01ec64.pth"),  # Balanced SAM model
        ]
        
        for model_type, model_name in recommended:
            try:
                logger.info(f"Downloading recommended model: {model_name}")
                self.download_model(model_type, model_name, progress_callback)
            except Exception as e:
                logger.warning(f"Failed to download {model_name}: {e}")
    
    def print_model_summary(self):
        """Print summary of available models"""
        print("\n🤖 TANGRAM Model Repository")
        print("=" * 50)
        
        for model_type, models in self.model_configs.items():
            print(f"\n{model_type.upper()} Models:")
            print("-" * 20)
            
            for model_name, config in models.items():
                status = "✓" if self.is_model_downloaded(model_type, model_name) else "○"
                size = config.get("size", "Unknown")
                desc = config.get("description", config.get("size", ""))
                
                print(f"{status} {model_name:<25} {size:<8} {desc}")
        
        print("\n📋 Legend:")
        print("✓ Downloaded    ○ Not downloaded")
        print("\n💡 Recommended for TANGRAM:")
        print("  - yolov8n.pt (fast object detection)")
        print("  - sam_vit_b_01ec64.pth (balanced segmentation)")

def main():
    """CLI interface for model downloader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TANGRAM Model Downloader")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--download", help="Download specific model (format: type/name)")
    parser.add_argument("--recommended", action="store_true", help="Download recommended models")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.models_dir)
    
    if args.list:
        downloader.print_model_summary()
        
    elif args.download:
        try:
            model_type, model_name = args.download.split("/")
            
            def progress_callback(progress, downloaded, total):
                print(f"\rDownloading: {progress:.1f}% ({downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB)", end="")
            
            downloader.download_model(model_type, model_name, progress_callback)
            print("\nDownload complete!")
            
        except ValueError as e:
            print(f"Error: {e}")
            print("Format: type/name (e.g., sam/sam_vit_b_01ec64.pth)")
            
    elif args.recommended:
        def progress_callback(progress, downloaded, total):
            print(f"\rDownloading: {progress:.1f}% ({downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB)", end="")
        
        downloader.download_recommended_models(progress_callback)
        print("\nRecommended models downloaded!")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()