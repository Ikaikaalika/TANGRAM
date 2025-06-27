#!/usr/bin/env python3

import cv2
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
from pathlib import Path

class YOLOByteTracker:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.yolo_model = YOLO(model_path)
        self.tracker_results = {}
        self.frame_count = 0
        
    def detect_and_track(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect objects with YOLO and track with ByteTrack
        Returns detection results with tracking IDs
        """
        results = self.yolo_model.track(frame, persist=True, tracker="bytetrack.yaml")
        
        frame_data = {
            "frame_id": self.frame_count,
            "detections": []
        }
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, track_id, conf, cls) in enumerate(zip(boxes, track_ids, confidences, classes)):
                x, y, w, h = box
                detection = {
                    "track_id": int(track_id),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "confidence": float(conf),
                    "class": int(cls),
                    "class_name": self.yolo_model.names[cls]
                }
                frame_data["detections"].append(detection)
        
        self.frame_count += 1
        return frame_data
    
    def process_video(self, video_path: str, output_dir: str = "data/tracking"):
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
    video_path = "data/sample_videos/tabletop_manipulation.mp4"
    if os.path.exists(video_path):
        print(f"Processing video: {video_path}")
        results = tracker.process_video(video_path)
        print(f"Processed {len(results)} frames")
    else:
        print(f"Sample video not found at {video_path}")

if __name__ == "__main__":
    main()