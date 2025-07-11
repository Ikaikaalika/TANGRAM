#!/usr/bin/env python3
"""
TANGRAM 3D Reconstruction Pipeline

This module provides a complete 3D reconstruction pipeline that:
1. Extracts frames from video
2. Runs COLMAP for camera pose estimation
3. Triangulates object positions using detected objects
4. Integrates with the scene graph system

Author: TANGRAM Team
License: MIT
"""

import os
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.tangram.core.reconstruction.extract_frames import extract_frames
from src.tangram.core.reconstruction.triangulate import PointTriangulator
from src.tangram.utils.logging_utils import setup_logger
from config import COLMAP_CONFIG, FRAME_EXTRACTION_CONFIG, PROJECT_ROOT

logger = setup_logger(__name__)

class ReconstructionPipeline:
    """Complete 3D reconstruction pipeline for TANGRAM"""
    
    def __init__(self, 
                 frames_dir: str = None,
                 colmap_output_dir: str = None,
                 skip_dense: bool = True):
        """
        Initialize reconstruction pipeline.
        
        Args:
            frames_dir: Directory for extracted frames
            colmap_output_dir: Directory for COLMAP output
            skip_dense: Skip dense reconstruction (faster, uses less memory)
        """
        self.frames_dir = frames_dir or str(PROJECT_ROOT / "data" / "processing" / "frames")
        self.colmap_output_dir = colmap_output_dir or str(PROJECT_ROOT / "data" / "outputs" / "point_clouds")
        self.skip_dense = skip_dense
        
        # Create directories
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.colmap_output_dir, exist_ok=True)
        
        # Initialize triangulator
        self.triangulator = PointTriangulator()
        
    def process_video(self, 
                      video_path: str,
                      detection_results: Dict[str, Any] = None,
                      force_recompute: bool = False) -> Dict[str, Any]:
        """
        Complete 3D reconstruction pipeline from video.
        
        Args:
            video_path: Path to input video
            detection_results: Object detection results from YOLO+SAM
            force_recompute: Force recomputation of COLMAP results
            
        Returns:
            Dictionary with reconstruction results
        """
        logger.info(f"Starting 3D reconstruction pipeline for {video_path}")
        
        results = {
            "success": False,
            "frames_extracted": 0,
            "colmap_success": False,
            "triangulation_success": False,
            "object_positions": {},
            "camera_poses": [],
            "point_cloud_path": None,
            "error": None
        }
        
        try:
            # Step 1: Extract frames
            logger.info("Step 1: Extracting frames from video")
            frames_extracted = self._extract_frames(video_path)
            results["frames_extracted"] = frames_extracted
            
            if frames_extracted < 2:
                raise ValueError(f"Need at least 2 frames for reconstruction. Got {frames_extracted}")
            
            # Step 2: Run COLMAP reconstruction
            logger.info("Step 2: Running COLMAP reconstruction")
            colmap_success = self._run_colmap_reconstruction(force_recompute)
            results["colmap_success"] = colmap_success
            
            if not colmap_success:
                raise RuntimeError("COLMAP reconstruction failed")
            
            # Step 3: Load COLMAP results
            logger.info("Step 3: Loading COLMAP results")
            self.triangulator.load_colmap_results(self.colmap_output_dir)
            results["camera_poses"] = len(self.triangulator.camera_poses)
            
            # Step 4: Triangulate object positions
            if detection_results:
                logger.info("Step 4: Triangulating object positions")
                object_positions = self._triangulate_objects(detection_results)
                results["object_positions"] = object_positions
                results["triangulation_success"] = len(object_positions) > 0
                
                # Step 5: Export point cloud
                if object_positions:
                    logger.info("Step 5: Exporting point cloud")
                    point_cloud_path = self._export_point_cloud(object_positions)
                    results["point_cloud_path"] = point_cloud_path
            else:
                logger.warning("No detection results provided. Skipping object triangulation.")
            
            results["success"] = True
            logger.info("3D reconstruction pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"3D reconstruction pipeline failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _extract_frames(self, video_path: str) -> int:
        """Extract frames from video for COLMAP processing"""
        # Clear existing frames
        if os.path.exists(self.frames_dir):
            for file in os.listdir(self.frames_dir):
                if file.endswith(('.jpg', '.png')):
                    os.remove(os.path.join(self.frames_dir, file))
        
        # Extract frames
        success = extract_frames(
            video_path=video_path,
            output_dir=self.frames_dir,
            frame_interval=FRAME_EXTRACTION_CONFIG["interval"],
            max_frames=FRAME_EXTRACTION_CONFIG["max_frames"]
        )
        
        if not success:
            raise RuntimeError("Frame extraction failed")
        
        # Count extracted frames
        frame_count = len([f for f in os.listdir(self.frames_dir) 
                          if f.endswith(('.jpg', '.png'))])
        
        logger.info(f"Extracted {frame_count} frames for reconstruction")
        return frame_count
    
    def _run_colmap_reconstruction(self, force_recompute: bool = False) -> bool:
        """Run COLMAP reconstruction pipeline"""
        
        # Check if reconstruction already exists
        sparse_model_path = os.path.join(self.colmap_output_dir, "sparse", "0")
        if os.path.exists(sparse_model_path) and not force_recompute:
            logger.info("COLMAP reconstruction already exists. Skipping...")
            return True
        
        # Clean output directory if recomputing
        if force_recompute:
            import shutil
            if os.path.exists(self.colmap_output_dir):
                shutil.rmtree(self.colmap_output_dir)
            os.makedirs(self.colmap_output_dir, exist_ok=True)
        
        database_path = os.path.join(self.colmap_output_dir, "database.db")
        
        try:
            # Step 1: Feature extraction
            logger.info("Running COLMAP feature extraction...")
            cmd = [
                "colmap", "feature_extractor",
                "--database_path", database_path,
                "--image_path", self.frames_dir,
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "PINHOLE",
                "--SiftExtraction.max_image_size", str(COLMAP_CONFIG["feature_extractor"]["max_image_size"])
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Feature extraction failed: {result.stderr}")
                return False
            
            # Step 2: Feature matching
            logger.info("Running COLMAP feature matching...")
            cmd = [
                "colmap", "exhaustive_matcher",
                "--database_path", database_path,
                "--SiftMatching.guided_matching", "1" if COLMAP_CONFIG["matcher"]["guided_matching"] else "0"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Feature matching failed: {result.stderr}")
                return False
            
            # Step 3: Structure from Motion
            logger.info("Running COLMAP structure from motion...")
            sparse_dir = os.path.join(self.colmap_output_dir, "sparse")
            os.makedirs(sparse_dir, exist_ok=True)
            
            cmd = [
                "colmap", "mapper",
                "--database_path", database_path,
                "--image_path", self.frames_dir,
                "--output_path", sparse_dir,
                "--Mapper.ba_refine_focal_length", "0" if not COLMAP_CONFIG["mapper"]["ba_refine_focal_length"] else "1",
                "--Mapper.ba_refine_principal_point", "0" if not COLMAP_CONFIG["mapper"]["ba_refine_principal_point"] else "1",
                "--Mapper.min_num_matches", str(COLMAP_CONFIG["mapper"]["min_num_matches"])
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Structure from motion failed: {result.stderr}")
                return False
            
            # Check if reconstruction was successful
            if not os.path.exists(sparse_model_path):
                logger.error("COLMAP reconstruction failed - no model generated")
                return False
            
            # Step 4: Convert to text format
            logger.info("Converting COLMAP model to text format...")
            cmd = [
                "colmap", "model_converter",
                "--input_path", sparse_model_path,
                "--output_path", sparse_model_path,
                "--output_type", "TXT"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Model conversion failed: {result.stderr}")
                # This is not critical, continue anyway
            
            # Step 5: Dense reconstruction (optional)
            if not self.skip_dense and COLMAP_CONFIG["dense_reconstruction"]["enable_dense"]:
                logger.info("Running dense reconstruction...")
                self._run_dense_reconstruction()
            
            logger.info("COLMAP reconstruction completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"COLMAP reconstruction failed: {e}")
            return False
    
    def _run_dense_reconstruction(self) -> bool:
        """Run COLMAP dense reconstruction (optional)"""
        try:
            sparse_model_path = os.path.join(self.colmap_output_dir, "sparse", "0")
            dense_dir = os.path.join(self.colmap_output_dir, "dense")
            os.makedirs(dense_dir, exist_ok=True)
            
            # Image undistortion
            cmd = [
                "colmap", "image_undistorter",
                "--image_path", self.frames_dir,
                "--input_path", sparse_model_path,
                "--output_path", dense_dir,
                "--output_type", "COLMAP"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Image undistortion failed: {result.stderr}")
                return False
            
            # Patch match stereo
            cmd = [
                "colmap", "patch_match_stereo",
                "--workspace_path", dense_dir,
                "--PatchMatchStereo.max_image_size", str(COLMAP_CONFIG["dense_reconstruction"]["max_image_size"])
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Patch match stereo failed: {result.stderr}")
                return False
            
            # Stereo fusion
            fused_ply_path = os.path.join(dense_dir, "fused.ply")
            cmd = [
                "colmap", "stereo_fusion",
                "--workspace_path", dense_dir,
                "--output_path", fused_ply_path,
                "--StereoFusion.max_image_size", str(COLMAP_CONFIG["dense_reconstruction"]["max_image_size"])
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Stereo fusion failed: {result.stderr}")
                return False
            
            logger.info(f"Dense reconstruction completed: {fused_ply_path}")
            return True
            
        except Exception as e:
            logger.error(f"Dense reconstruction failed: {e}")
            return False
    
    def _triangulate_objects(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Triangulate 3D positions of detected objects"""
        try:
            # Convert detection results to format expected by triangulator
            mask_data = self._convert_detection_to_mask_data(detection_results)
            
            # Triangulate object positions
            object_positions = self.triangulator.triangulate_object_positions(
                mask_data, 
                output_dir=self.colmap_output_dir
            )
            
            return object_positions
            
        except Exception as e:
            logger.error(f"Object triangulation failed: {e}")
            return {}
    
    def _convert_detection_to_mask_data(self, detection_results: Dict[str, Any]) -> List[Dict]:
        """Convert YOLO detection results to mask data format"""
        mask_data = []
        
        # This assumes detection_results has the structure from YOLO+SAM pipeline
        # Adjust based on actual detection result format
        for frame_id, detections in detection_results.items():
            if isinstance(detections, dict) and "detections" in detections:
                frame_masks = []
                
                for detection in detections["detections"]:
                    mask_info = {
                        "track_id": detection.get("track_id", 0),
                        "class_name": detection.get("class_name", "unknown"),
                        "bbox": detection.get("bbox", [0, 0, 100, 100]),
                        "confidence": detection.get("confidence", 0.5),
                        "mask_file": detection.get("mask_file", "")
                    }
                    frame_masks.append(mask_info)
                
                mask_data.append({
                    "frame_id": int(frame_id),
                    "masks": frame_masks
                })
        
        return mask_data
    
    def _export_point_cloud(self, object_positions: Dict[str, Any]) -> str:
        """Export object positions as point cloud"""
        point_cloud_path = os.path.join(self.colmap_output_dir, "object_point_cloud.ply")
        
        try:
            self.triangulator.export_point_cloud(object_positions, point_cloud_path)
            logger.info(f"Point cloud exported to {point_cloud_path}")
            return point_cloud_path
            
        except Exception as e:
            logger.error(f"Point cloud export failed: {e}")
            return None
    
    def get_reconstruction_stats(self) -> Dict[str, Any]:
        """Get statistics about the reconstruction"""
        stats = {
            "frames_dir": self.frames_dir,
            "colmap_output_dir": self.colmap_output_dir,
            "num_frames": 0,
            "num_cameras": 0,
            "num_registered_images": 0,
            "num_3d_points": 0,
            "reconstruction_exists": False
        }
        
        # Count frames
        if os.path.exists(self.frames_dir):
            stats["num_frames"] = len([f for f in os.listdir(self.frames_dir) 
                                     if f.endswith(('.jpg', '.png'))])
        
        # Check COLMAP results
        sparse_model_path = os.path.join(self.colmap_output_dir, "sparse", "0")
        if os.path.exists(sparse_model_path):
            stats["reconstruction_exists"] = True
            
            # Count cameras
            cameras_file = os.path.join(sparse_model_path, "cameras.txt")
            if os.path.exists(cameras_file):
                with open(cameras_file, 'r') as f:
                    stats["num_cameras"] = len([line for line in f 
                                              if not line.startswith('#') and line.strip()])
            
            # Count registered images
            images_file = os.path.join(sparse_model_path, "images.txt")
            if os.path.exists(images_file):
                with open(images_file, 'r') as f:
                    lines = [line for line in f if not line.startswith('#') and line.strip()]
                    stats["num_registered_images"] = len(lines) // 2  # Each image has 2 lines
            
            # Count 3D points
            points_file = os.path.join(sparse_model_path, "points3D.txt")
            if os.path.exists(points_file):
                with open(points_file, 'r') as f:
                    stats["num_3d_points"] = len([line for line in f 
                                                if not line.startswith('#') and line.strip()])
        
        return stats

def main():
    """Test the reconstruction pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TANGRAM 3D Reconstruction Pipeline")
    parser.add_argument("--video", "-v", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="data/outputs/point_clouds", help="Output directory")
    parser.add_argument("--force", "-f", action="store_true", help="Force recomputation")
    parser.add_argument("--dense", action="store_true", help="Enable dense reconstruction")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ReconstructionPipeline(
        colmap_output_dir=args.output,
        skip_dense=not args.dense
    )
    
    # Run reconstruction
    results = pipeline.process_video(args.video, force_recompute=args.force)
    
    # Print results
    print("\n" + "="*50)
    print("TANGRAM 3D Reconstruction Results")
    print("="*50)
    print(f"Success: {results['success']}")
    print(f"Frames extracted: {results['frames_extracted']}")
    print(f"COLMAP success: {results['colmap_success']}")
    print(f"Camera poses: {results.get('camera_poses', 0)}")
    print(f"Object positions: {len(results.get('object_positions', {}))}")
    print(f"Point cloud: {results.get('point_cloud_path', 'Not generated')}")
    
    if results.get('error'):
        print(f"Error: {results['error']}")
    
    # Print statistics
    stats = pipeline.get_reconstruction_stats()
    print(f"\nStatistics:")
    print(f"  Frames: {stats['num_frames']}")
    print(f"  Cameras: {stats['num_cameras']}")
    print(f"  Registered images: {stats['num_registered_images']}")
    print(f"  3D points: {stats['num_3d_points']}")

if __name__ == "__main__":
    main()