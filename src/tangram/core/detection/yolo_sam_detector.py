#!/usr/bin/env python3
"""
YOLO + SAM Object Detection and Segmentation

This module integrates YOLO for object detection with SAM for precise segmentation.
Provides real-time object detection and segmentation for robotic scene understanding.

Author: TANGRAM Team
License: MIT
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOSAMDetector:
    """Integrated YOLO + SAM object detection and segmentation"""
    
    def __init__(self, 
                 yolo_model_path: str = "yolov8n.pt",
                 sam_model_path: str = None,
                 sam_model_type: str = "vit_b",
                 device: str = "auto"):
        """
        Initialize YOLO + SAM detector
        
        Args:
            yolo_model_path: Path to YOLO model weights
            sam_model_path: Path to SAM model weights (auto-download if None)
            sam_model_type: SAM model type (vit_b, vit_l, vit_h)
            device: Device to run on (auto, cpu, cuda, mps)
        """
        self.device = self._get_device(device)
        
        # Initialize YOLO
        self.yolo = YOLO(yolo_model_path)
        logger.info(f"YOLO model loaded: {yolo_model_path}")
        
        # Initialize SAM
        self.sam = self._initialize_sam(sam_model_path, sam_model_type)
        self.sam_predictor = SamPredictor(self.sam)
        logger.info(f"SAM model loaded: {sam_model_type}")
        
        # Detection parameters
        self.conf_threshold = 0.5
        self.iou_threshold = 0.7
        self.max_detections = 50
        
        # Track IDs for consistent object tracking
        self.next_track_id = 1
        self.track_history = {}
        
    def _get_device(self, device: str) -> str:
        """Get appropriate device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _initialize_sam(self, model_path: str, model_type: str):
        """Initialize SAM model"""
        if model_path is None:
            # Auto-download SAM model
            model_urls = {
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            }
            
            models_dir = Path("models/sam")
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / f"sam_{model_type}.pth"
            
            if not model_path.exists():
                logger.info(f"Downloading SAM model: {model_type}")
                import urllib.request
                urllib.request.urlretrieve(model_urls[model_type], str(model_path))
        
        # Load SAM model
        sam = sam_model_registry[model_type](checkpoint=str(model_path))
        sam.to(device=self.device)
        return sam
    
    def detect_objects(self, image: np.ndarray, 
                      segment: bool = True,
                      track: bool = True) -> Dict[str, Any]:
        """
        Detect and optionally segment objects in image
        
        Args:
            image: Input image (BGR format)
            segment: Whether to perform segmentation
            track: Whether to track objects across frames
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        # YOLO detection
        results = self.yolo(image, 
                          conf=self.conf_threshold,
                          iou=self.iou_threshold,
                          max_det=self.max_detections)
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            
            # Set image for SAM
            if segment:
                self.sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box
                class_name = self.yolo.names[int(cls_id)]
                
                detection = {
                    "id": i,
                    "class_name": class_name,
                    "class_id": int(cls_id),
                    "confidence": float(conf),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "center": [(int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2],
                    "area": (int(x2) - int(x1)) * (int(y2) - int(y1))
                }
                
                # Add tracking ID
                if track:
                    track_id = self._get_track_id(detection)
                    detection["track_id"] = track_id
                
                # Add segmentation mask
                if segment:
                    mask = self._segment_object(box)
                    detection["mask"] = mask
                    detection["mask_area"] = np.sum(mask)
                
                detections.append(detection)
        
        processing_time = time.time() - start_time
        
        return {
            "detections": detections,
            "num_detections": len(detections),
            "processing_time": processing_time,
            "image_shape": image.shape,
            "timestamp": time.time()
        }
    
    def _get_track_id(self, detection: Dict[str, Any]) -> int:
        """Get or assign tracking ID for object"""
        # Simple tracking based on center position and class
        # In production, use more sophisticated tracking like ByteTrack
        
        center = detection["center"]
        class_name = detection["class_name"]
        
        # Look for existing track
        for track_id, track_info in self.track_history.items():
            if track_info["class_name"] == class_name:
                # Calculate distance from last position
                last_center = track_info["last_center"]
                distance = np.sqrt((center[0] - last_center[0])**2 + 
                                 (center[1] - last_center[1])**2)
                
                # If close enough, update track
                if distance < 100:  # Threshold for track continuity
                    track_info["last_center"] = center
                    track_info["last_seen"] = time.time()
                    return track_id
        
        # Create new track
        track_id = self.next_track_id
        self.next_track_id += 1
        
        self.track_history[track_id] = {
            "class_name": class_name,
            "last_center": center,
            "last_seen": time.time(),
            "created": time.time()
        }
        
        return track_id
    
    def _segment_object(self, box: np.ndarray) -> np.ndarray:
        """Segment object using SAM"""
        try:
            # Use bounding box as prompt for SAM
            input_box = np.array([box])
            
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False,
            )
            
            # Return the best mask
            return masks[0]
            
        except Exception as e:
            logger.warning(f"SAM segmentation failed: {e}")
            # Return empty mask
            return np.zeros((box[3] - box[1], box[2] - box[0]), dtype=bool)
    
    def visualize_detections(self, image: np.ndarray, 
                           detections: List[Dict[str, Any]],
                           show_masks: bool = True,
                           show_boxes: bool = True,
                           show_labels: bool = True) -> np.ndarray:
        """
        Visualize detection results on image
        
        Args:
            image: Input image
            detections: List of detection results
            show_masks: Whether to show segmentation masks
            show_boxes: Whether to show bounding boxes
            show_labels: Whether to show class labels
            
        Returns:
            Annotated image
        """
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            track_id = detection.get("track_id", -1)
            
            # Draw mask
            if show_masks and "mask" in detection:
                mask = detection["mask"]
                color = self._get_color_for_class(detection["class_id"])
                
                # Apply mask overlay
                mask_overlay = np.zeros_like(result_image)
                mask_overlay[mask] = color
                result_image = cv2.addWeighted(result_image, 0.7, mask_overlay, 0.3, 0)
            
            # Draw bounding box
            if show_boxes:
                color = self._get_color_for_class(detection["class_id"])
                cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            color, 2)
            
            # Draw label
            if show_labels:
                label = f"{class_name} {confidence:.2f}"
                if track_id >= 0:
                    label += f" ID:{track_id}"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(result_image, (bbox[0], bbox[1] - label_size[1] - 10),
                            (bbox[0] + label_size[0], bbox[1]), 
                            self._get_color_for_class(detection["class_id"]), -1)
                cv2.putText(result_image, label, (bbox[0], bbox[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for class"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 128), (128, 128, 0), (128, 0, 0), (0, 128, 0)
        ]
        return colors[class_id % len(colors)]
    
    def cleanup_old_tracks(self, max_age: float = 5.0):
        """Remove old tracks that haven't been seen recently"""
        current_time = time.time()
        tracks_to_remove = []
        
        for track_id, track_info in self.track_history.items():
            if current_time - track_info["last_seen"] > max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_history[track_id]
    
    def get_detection_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of detection results"""
        class_counts = {}
        total_confidence = 0
        
        for detection in detections:
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
            total_confidence += confidence
        
        return {
            "total_objects": len(detections),
            "class_counts": class_counts,
            "average_confidence": total_confidence / max(len(detections), 1),
            "unique_classes": len(class_counts)
        }

def main():
    """Test the YOLO + SAM detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test YOLO + SAM detector")
    parser.add_argument("--input", "-i", required=True, help="Input image or video path")
    parser.add_argument("--output", "-o", help="Output path for results")
    parser.add_argument("--no-segment", action="store_true", help="Disable segmentation")
    parser.add_argument("--no-track", action="store_true", help="Disable tracking")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLOSAMDetector()
    
    # Process input
    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Process image
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not load image {args.input}")
            return
        
        # Detect objects
        results = detector.detect_objects(image, 
                                        segment=not args.no_segment,
                                        track=not args.no_track)
        
        # Visualize results
        annotated_image = detector.visualize_detections(image, results["detections"])
        
        # Show results
        print(f"Detection Results:")
        print(f"  Objects found: {results['num_detections']}")
        print(f"  Processing time: {results['processing_time']:.3f}s")
        
        summary = detector.get_detection_summary(results["detections"])
        print(f"  Class counts: {summary['class_counts']}")
        
        # Save or display
        if args.output:
            cv2.imwrite(args.output, annotated_image)
            print(f"Result saved to: {args.output}")
        else:
            cv2.imshow("YOLO + SAM Detection", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    else:
        # Process video
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Could not open video {args.input}")
            return
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            results = detector.detect_objects(frame,
                                            segment=not args.no_segment,
                                            track=not args.no_track)
            
            # Visualize results
            annotated_frame = detector.visualize_detections(frame, results["detections"])
            
            # Show frame
            cv2.imshow("YOLO + SAM Detection", annotated_frame)
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {results['num_detections']} objects detected")
            
            frame_count += 1
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Cleanup old tracks periodically
            if frame_count % 100 == 0:
                detector.cleanup_old_tracks()
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()