#!/usr/bin/env python3

import cv2
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple, Union
from ultralytics import YOLO
from pathlib import Path
import logging

# Import config for YOLO settings
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import YOLO_CONFIG

logger = logging.getLogger(__name__)

class YOLOByteTracker:
    def __init__(self, model_path: str = None):
        """Initialize YOLO with fallback model support"""
        if model_path is None:
            model_path = YOLO_CONFIG["model_name"]
        
        self.yolo_model = self._load_model_with_fallback(model_path)
        self.tracker_results = {}
        self.frame_count = 0
        self.confidence_threshold = YOLO_CONFIG["confidence_threshold"]
        self.iou_threshold = YOLO_CONFIG["iou_threshold"]
        self.max_detections = YOLO_CONFIG["max_detections"]
        self.imgsz = YOLO_CONFIG["imgsz"]
        
    def _load_model_with_fallback(self, primary_model: str) -> YOLO:
        """Load YOLO model with fallback options"""
        models_to_try = [primary_model] + YOLO_CONFIG.get("model_fallbacks", [])
        
        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to load YOLO model: {model_name}")
                model = YOLO(model_name)
                logger.info(f"Successfully loaded YOLO model: {model_name}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        # If all else fails, try the smallest model
        logger.error("All YOLO models failed to load, falling back to yolov8n.pt")
        return YOLO("yolov8n.pt")
        
    def detect_objects(self, frame: np.ndarray, use_tracking: bool = True) -> Dict[str, Any]:
        """
        Detect objects with YOLO, optionally with tracking
        Returns detection results with optional tracking IDs
        """
        try:
            if use_tracking:
                results = self.yolo_model.track(
                    frame, 
                    persist=True, 
                    tracker="bytetrack.yaml",
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    max_det=self.max_detections
                )
            else:
                results = self.yolo_model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    max_det=self.max_detections
                )
            
            frame_data = {
                "frame_id": self.frame_count,
                "detections": []
            }
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Get tracking IDs if available
                track_ids = None
                if use_tracking and results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    x, y, w, h = box
                    detection = {
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "confidence": float(conf),
                        "class": int(cls),
                        "class_name": self.yolo_model.names[cls]
                    }
                    
                    # Add tracking ID if available
                    if track_ids is not None:
                        detection["track_id"] = int(track_ids[i])
                    else:
                        detection["track_id"] = i  # Use index as pseudo-ID for photos
                    
                    frame_data["detections"].append(detection)
            
            self.frame_count += 1
            return frame_data
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {"frame_id": self.frame_count, "detections": []}
    
    def detect_and_track(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility
        """
        return self.detect_objects(frame, use_tracking=True)
    
    def process_video(self, video_path: str, output_dir: str = "data/processing/tracking"):
        """Process entire video and save tracking results"""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        all_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_data = self.detect_and_track(frame)
            all_frames.append(frame_data)
            
            # Optional: visualize results
            annotated_frame = self.visualize_tracks(frame, frame_data)
            
        cap.release()
        
        # Save tracking results
        output_file = os.path.join(output_dir, "tracking_results.json")
        with open(output_file, 'w') as f:
            json.dump(all_frames, f, indent=2)
            
        print(f"Tracking results saved to {output_file}")
        return all_frames
    
    def process_image(self, image_path: Union[str, Path], output_dir: str = "data/processing/tracking") -> List[Dict]:
        """
        Process a single image for object detection
        Returns detection results in the same format as video processing
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Reset frame count for single image
        self.frame_count = 0
        
        # Load image
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return []
        
        try:
            frame = cv2.imread(str(image_path))
            if frame is None:
                logger.error(f"Could not load image: {image_path}")
                return []
            
            # Process image (no tracking for single images)
            frame_data = self.detect_objects(frame, use_tracking=False)
            all_frames = [frame_data]
            
            # Optional: visualize results
            annotated_frame = self.visualize_tracks(frame, frame_data)
            
            # Save annotated image
            annotated_path = Path(output_dir) / f"annotated_{image_path.name}"
            cv2.imwrite(str(annotated_path), annotated_frame)
            
            # Save detection results
            output_file = os.path.join(output_dir, "tracking_results.json")
            with open(output_file, 'w') as f:
                json.dump(all_frames, f, indent=2)
                
            logger.info(f"Image processing complete. Results saved to {output_file}")
            logger.info(f"Annotated image saved to {annotated_path}")
            
            return all_frames
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return []
    
    def process_media(self, media_path: Union[str, Path], output_dir: str = "data/processing/tracking") -> List[Dict]:
        """
        Process either image or video automatically based on file extension
        """
        media_path = Path(media_path)
        
        # Image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        # Video extensions  
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        file_ext = media_path.suffix.lower()
        
        if file_ext in image_extensions:
            logger.info(f"Processing image: {media_path}")
            return self.process_image(media_path, output_dir)
        elif file_ext in video_extensions:
            logger.info(f"Processing video: {media_path}")
            return self.process_video(str(media_path), output_dir)
        else:
            logger.error(f"Unsupported file format: {file_ext}")
            return []
    
    def visualize_tracks(self, frame: np.ndarray, frame_data: Dict) -> np.ndarray:
        """Draw bounding boxes and track IDs on frame"""
        annotated = frame.copy()
        
        for detection in frame_data["detections"]:
            x, y, w, h = detection["bbox"]
            track_id = detection["track_id"]
            class_name = detection["class_name"]
            conf = detection["confidence"]
            
            # Convert center coordinates to corner coordinates
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated

def main():
    print("YOLO + ByteTrack Object Detection and Tracking Module")
    
    # Example usage
    tracker = YOLOByteTracker()
    
    # Check if sample video exists
    video_path = "data/inputs/samples/tabletop_manipulation.mp4"
    if os.path.exists(video_path):
        print(f"Processing video: {video_path}")
        results = tracker.process_video(video_path)
        print(f"Processed {len(results)} frames")
    else:
        print(f"Sample video not found at {video_path}")

if __name__ == "__main__":
    main()