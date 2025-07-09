#!/usr/bin/env python3
"""
Object 3D Mapper for TANGRAM

Combines object detection with 3D reconstruction to create accurate 3D object positions.
Integrates YOLO+SAM detections with COLMAP camera poses for spatial understanding.

Author: TANGRAM Team
License: MIT
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass

from .triangulate import PointTriangulator
from ..detection.yolo_sam_detector import YOLOSAMDetector

logger = logging.getLogger(__name__)

@dataclass
class Object3D:
    """3D object representation"""
    id: int
    track_id: int
    class_name: str
    confidence: float
    position_3d: np.ndarray
    bbox_2d: List[int]
    mask: Optional[np.ndarray] = None
    observations: List[Dict] = None
    uncertainty: float = 0.0
    
    def __post_init__(self):
        if self.observations is None:
            self.observations = []

class Object3DMapper:
    """Maps 2D object detections to 3D space using camera poses"""
    
    def __init__(self, 
                 colmap_output_dir: str,
                 detector: YOLOSAMDetector = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize 3D object mapper
        
        Args:
            colmap_output_dir: Directory containing COLMAP reconstruction
            detector: YOLO+SAM detector instance
            confidence_threshold: Minimum confidence for object detection
        """
        self.colmap_output_dir = Path(colmap_output_dir)
        self.confidence_threshold = confidence_threshold
        
        # Initialize detector
        if detector is None:
            self.detector = YOLOSAMDetector()
        else:
            self.detector = detector
        
        # Initialize triangulator
        self.triangulator = PointTriangulator()
        
        # Load COLMAP reconstruction
        self.load_colmap_data()
        
        # Object tracking
        self.objects_3d = {}  # track_id -> Object3D
        self.frame_detections = {}  # frame_id -> detections
        
    def load_colmap_data(self):
        """Load COLMAP reconstruction data"""
        try:
            self.triangulator.load_colmap_results(str(self.colmap_output_dir))
            logger.info(f"Loaded COLMAP data: {len(self.triangulator.camera_poses)} cameras")
        except Exception as e:
            logger.error(f"Failed to load COLMAP data: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> List[Dict[str, Any]]:
        """
        Process a single frame for object detection
        
        Args:
            frame: Input frame
            frame_id: Frame identifier
            
        Returns:
            List of detection results
        """
        # Run object detection
        detection_results = self.detector.detect_objects(frame, segment=True, track=True)
        
        # Filter by confidence
        valid_detections = [
            det for det in detection_results["detections"]
            if det["confidence"] >= self.confidence_threshold
        ]
        
        # Store detections for this frame
        self.frame_detections[frame_id] = valid_detections
        
        logger.info(f"Frame {frame_id}: {len(valid_detections)} objects detected")
        
        return valid_detections
    
    def process_video(self, video_path: str, frame_interval: int = 5) -> Dict[str, Any]:
        """
        Process entire video for object detection
        
        Args:
            video_path: Path to video file
            frame_interval: Process every Nth frame
            
        Returns:
            Processing results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = 0
        processed_frames = 0
        total_detections = 0
        
        logger.info(f"Processing video: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % frame_interval == 0:
                detections = self.process_frame(frame, processed_frames)
                total_detections += len(detections)
                processed_frames += 1
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"Processed {processed_frames} frames, {total_detections} total detections")
        
        return {
            "processed_frames": processed_frames,
            "total_detections": total_detections,
            "frame_detections": self.frame_detections
        }
    
    def triangulate_objects(self) -> Dict[int, Object3D]:
        """
        Triangulate 3D positions of detected objects
        
        Returns:
            Dictionary mapping track_id to Object3D
        """
        logger.info("Triangulating 3D object positions...")
        
        # Group detections by track_id
        track_groups = {}
        for frame_id, detections in self.frame_detections.items():
            for detection in detections:
                track_id = detection.get("track_id", -1)
                if track_id not in track_groups:
                    track_groups[track_id] = []
                
                track_groups[track_id].append({
                    "frame_id": frame_id,
                    "detection": detection
                })
        
        # Triangulate each track
        for track_id, observations in track_groups.items():
            if len(observations) < 2:
                logger.warning(f"Track {track_id} has only {len(observations)} observations, skipping")
                continue
            
            # Triangulate 3D position
            position_3d = self._triangulate_track(observations)
            
            if position_3d is not None:
                # Create Object3D
                latest_detection = observations[-1]["detection"]
                
                obj_3d = Object3D(
                    id=len(self.objects_3d),
                    track_id=track_id,
                    class_name=latest_detection["class_name"],
                    confidence=np.mean([obs["detection"]["confidence"] for obs in observations]),
                    position_3d=position_3d,
                    bbox_2d=latest_detection["bbox"],
                    mask=latest_detection.get("mask"),
                    observations=observations,
                    uncertainty=self._calculate_uncertainty(observations)
                )
                
                self.objects_3d[track_id] = obj_3d
                logger.info(f"Triangulated {obj_3d.class_name} at {position_3d}")
        
        logger.info(f"Successfully triangulated {len(self.objects_3d)} objects")
        return self.objects_3d
    
    def _triangulate_track(self, observations: List[Dict]) -> Optional[np.ndarray]:
        """
        Triangulate 3D position from multiple 2D observations
        
        Args:
            observations: List of observations with frame_id and detection
            
        Returns:
            3D position or None if triangulation fails
        """
        if len(observations) < 2:
            return None
        
        # Use first two observations for triangulation
        obs1 = observations[0]
        obs2 = observations[1]
        
        frame_id1 = obs1["frame_id"]
        frame_id2 = obs2["frame_id"]
        
        # Get camera poses
        if frame_id1 >= len(self.triangulator.camera_poses) or frame_id2 >= len(self.triangulator.camera_poses):
            logger.warning(f"Invalid frame IDs: {frame_id1}, {frame_id2}")
            return None
        
        pose1 = self.triangulator.camera_poses[frame_id1]
        pose2 = self.triangulator.camera_poses[frame_id2]
        
        # Get 2D points (object centers)
        center1 = obs1["detection"]["center"]
        center2 = obs2["detection"]["center"]
        
        # Get camera intrinsics
        if len(self.triangulator.camera_matrices) == 0:
            logger.error("No camera matrices available")
            return None
        
        K = self.triangulator.camera_matrices[0]  # Assume single camera
        
        # Create projection matrices
        P1 = K @ pose1["pose"]
        P2 = K @ pose2["pose"]
        
        # Triangulate
        try:
            point1 = np.array(center1, dtype=np.float32).reshape(2, 1)
            point2 = np.array(center2, dtype=np.float32).reshape(2, 1)
            
            points_4d = cv2.triangulatePoints(P1, P2, point1, point2)
            
            # Convert to 3D
            if points_4d[3, 0] != 0:
                points_3d = points_4d[:3, 0] / points_4d[3, 0]
                return points_3d
            else:
                logger.warning("Triangulation failed: homogeneous coordinate is zero")
                return None
                
        except Exception as e:
            logger.error(f"Triangulation error: {e}")
            return None
    
    def _calculate_uncertainty(self, observations: List[Dict]) -> float:
        """
        Calculate uncertainty in 3D position based on observations
        
        Args:
            observations: List of observations
            
        Returns:
            Uncertainty value (0-1)
        """
        if len(observations) < 2:
            return 1.0
        
        # Calculate confidence variance
        confidences = [obs["detection"]["confidence"] for obs in observations]
        confidence_var = np.var(confidences)
        
        # Calculate position variance (2D)
        centers = [obs["detection"]["center"] for obs in observations]
        center_var = np.var(centers, axis=0).mean()
        
        # Combine uncertainties
        uncertainty = min(confidence_var + center_var / 10000, 1.0)
        
        return uncertainty
    
    def get_objects_in_region(self, center: np.ndarray, radius: float) -> List[Object3D]:
        """
        Get objects within a spherical region
        
        Args:
            center: Center of the region
            radius: Radius of the region
            
        Returns:
            List of objects in the region
        """
        objects_in_region = []
        
        for obj_3d in self.objects_3d.values():
            distance = np.linalg.norm(obj_3d.position_3d - center)
            if distance <= radius:
                objects_in_region.append(obj_3d)
        
        return objects_in_region
    
    def get_objects_by_class(self, class_name: str) -> List[Object3D]:
        """
        Get all objects of a specific class
        
        Args:
            class_name: Name of the class
            
        Returns:
            List of objects of the specified class
        """
        return [obj for obj in self.objects_3d.values() if obj.class_name == class_name]
    
    def export_objects_3d(self, output_path: str):
        """
        Export 3D objects to JSON file
        
        Args:
            output_path: Path to output JSON file
        """
        export_data = {
            "objects": [],
            "metadata": {
                "total_objects": len(self.objects_3d),
                "colmap_output_dir": str(self.colmap_output_dir),
                "confidence_threshold": self.confidence_threshold
            }
        }
        
        for obj_3d in self.objects_3d.values():
            obj_data = {
                "id": obj_3d.id,
                "track_id": obj_3d.track_id,
                "class_name": obj_3d.class_name,
                "confidence": obj_3d.confidence,
                "position_3d": obj_3d.position_3d.tolist(),
                "bbox_2d": obj_3d.bbox_2d,
                "uncertainty": obj_3d.uncertainty,
                "num_observations": len(obj_3d.observations)
            }
            export_data["objects"].append(obj_data)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(self.objects_3d)} objects to {output_path}")
    
    def create_scene_description(self) -> Dict[str, Any]:
        """
        Create a structured scene description for LLM
        
        Returns:
            Scene description dictionary
        """
        scene = {
            "objects": [],
            "spatial_relationships": [],
            "scene_bounds": self._calculate_scene_bounds(),
            "object_counts": self._get_object_counts(),
            "summary": ""
        }
        
        # Add objects
        for obj_3d in self.objects_3d.values():
            scene["objects"].append({
                "id": obj_3d.id,
                "track_id": obj_3d.track_id,
                "name": obj_3d.class_name,
                "position": obj_3d.position_3d.tolist(),
                "confidence": obj_3d.confidence,
                "uncertainty": obj_3d.uncertainty
            })
        
        # Calculate spatial relationships
        scene["spatial_relationships"] = self._calculate_spatial_relationships()
        
        # Create summary
        scene["summary"] = self._create_scene_summary()
        
        return scene
    
    def _calculate_scene_bounds(self) -> Dict[str, float]:
        """Calculate bounding box of all objects"""
        if not self.objects_3d:
            return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0}
        
        positions = [obj.position_3d for obj in self.objects_3d.values()]
        positions = np.array(positions)
        
        return {
            "min_x": float(positions[:, 0].min()),
            "max_x": float(positions[:, 0].max()),
            "min_y": float(positions[:, 1].min()),
            "max_y": float(positions[:, 1].max()),
            "min_z": float(positions[:, 2].min()),
            "max_z": float(positions[:, 2].max())
        }
    
    def _get_object_counts(self) -> Dict[str, int]:
        """Get count of each object class"""
        counts = {}
        for obj_3d in self.objects_3d.values():
            counts[obj_3d.class_name] = counts.get(obj_3d.class_name, 0) + 1
        return counts
    
    def _calculate_spatial_relationships(self) -> List[Dict]:
        """Calculate spatial relationships between objects"""
        relationships = []
        
        objects = list(self.objects_3d.values())
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                distance = np.linalg.norm(obj1.position_3d - obj2.position_3d)
                
                # Determine relationship
                if distance < 0.1:  # 10cm
                    relation = "touching"
                elif distance < 0.3:  # 30cm
                    relation = "near"
                elif distance < 1.0:  # 1m
                    relation = "close"
                else:
                    relation = "far"
                
                # Check vertical relationship
                z_diff = obj1.position_3d[2] - obj2.position_3d[2]
                if abs(z_diff) > 0.05:  # 5cm height difference
                    if z_diff > 0:
                        vertical_relation = "above"
                    else:
                        vertical_relation = "below"
                else:
                    vertical_relation = "same_level"
                
                relationships.append({
                    "object1": obj1.id,
                    "object2": obj2.id,
                    "distance": float(distance),
                    "relation": relation,
                    "vertical_relation": vertical_relation
                })
        
        return relationships
    
    def _create_scene_summary(self) -> str:
        """Create natural language scene summary"""
        if not self.objects_3d:
            return "The scene is empty."
        
        counts = self._get_object_counts()
        total_objects = len(self.objects_3d)
        
        summary = f"The scene contains {total_objects} objects: "
        
        count_strs = []
        for class_name, count in counts.items():
            if count == 1:
                count_strs.append(f"1 {class_name}")
            else:
                count_strs.append(f"{count} {class_name}s")
        
        summary += ", ".join(count_strs) + "."
        
        # Add spatial information
        bounds = self._calculate_scene_bounds()
        scene_width = bounds["max_x"] - bounds["min_x"]
        scene_depth = bounds["max_y"] - bounds["min_y"]
        
        summary += f" The objects are distributed over a {scene_width:.1f}m × {scene_depth:.1f}m area."
        
        return summary

def main():
    """Test the object 3D mapper"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Object 3D Mapper")
    parser.add_argument("--video", "-v", required=True, help="Input video path")
    parser.add_argument("--colmap-dir", "-c", required=True, help="COLMAP output directory")
    parser.add_argument("--output", "-o", default="objects_3d.json", help="Output JSON file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--interval", type=int, default=5, help="Frame interval")
    
    args = parser.parse_args()
    
    # Initialize mapper
    mapper = Object3DMapper(args.colmap_dir, confidence_threshold=args.confidence)
    
    # Process video
    print(f"Processing video: {args.video}")
    video_results = mapper.process_video(args.video, args.interval)
    
    # Triangulate objects
    print("Triangulating 3D object positions...")
    objects_3d = mapper.triangulate_objects()
    
    # Export results
    mapper.export_objects_3d(args.output)
    
    # Create scene description
    scene_desc = mapper.create_scene_description()
    
    # Print results
    print(f"\n🎯 Results:")
    print(f"  Objects detected: {len(objects_3d)}")
    print(f"  Scene summary: {scene_desc['summary']}")
    print(f"  Output saved to: {args.output}")
    
    # Print object details
    for obj in objects_3d.values():
        print(f"  {obj.class_name} at {obj.position_3d} (confidence: {obj.confidence:.2f})")

if __name__ == "__main__":
    main()