#!/usr/bin/env python3

import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path: str, output_dir: str, 
                  frame_interval: int = 1, max_frames: int = None):
    """
    Extract frames from video for COLMAP processing
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (1 = all frames)
        max_frames: Maximum number of frames to extract
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    frame_count = 0
    extracted_count = 0
    
    print(f"Extracting frames from {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Frame interval: {frame_interval}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{extracted_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Save frame
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
            
            if extracted_count % 50 == 0:
                print(f"Extracted {extracted_count} frames...")
            
            # Stop if max frames reached
            if max_frames and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"Extraction complete. Total frames extracted: {extracted_count}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video for COLMAP")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="data/processing/frames", help="Output directory")
    parser.add_argument("--interval", type=int, default=5, 
                       help="Extract every Nth frame (default: 5)")
    parser.add_argument("--max-frames", type=int, default=200,
                       help="Maximum frames to extract (default: 200)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input video {args.input} not found")
        return
    
    extract_frames(args.input, args.output, args.interval, args.max_frames)

if __name__ == "__main__":
    main()