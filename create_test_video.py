#!/usr/bin/env python3
"""
Create a simple test video with objects on a table for TANGRAM pipeline testing.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def create_tabletop_scene(frame_num, total_frames):
    """Create a frame showing objects on a tabletop"""
    # Create base image (640x480)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 220  # Light gray background
    
    # Draw table (brown rectangle)
    cv2.rectangle(img, (50, 300), (590, 450), (139, 69, 19), -1)  # Brown table
    
    # Table edge highlight
    cv2.rectangle(img, (50, 300), (590, 310), (180, 100, 50), -1)
    
    # Animate a red cup moving slightly
    cup_x = 150 + int(20 * np.sin(frame_num * 0.1))
    cup_y = 250
    cv2.circle(img, (cup_x, cup_y), 30, (0, 0, 255), -1)  # Red cup
    cv2.circle(img, (cup_x, cup_y), 30, (0, 0, 200), 3)   # Cup outline
    
    # Blue block (static)
    block_x, block_y = 350, 260
    cv2.rectangle(img, (block_x-25, block_y-25), (block_x+25, block_y+25), (255, 0, 0), -1)  # Blue block
    cv2.rectangle(img, (block_x-25, block_y-25), (block_x+25, block_y+25), (200, 0, 0), 3)   # Block outline
    
    # Green plate (static)
    plate_x, plate_y = 500, 280
    cv2.ellipse(img, (plate_x, plate_y), (40, 15), 0, 0, 360, (0, 255, 0), -1)  # Green plate
    cv2.ellipse(img, (plate_x, plate_y), (40, 15), 0, 0, 360, (0, 200, 0), 3)   # Plate outline
    
    # Add some texture/shadows for realism
    cv2.ellipse(img, (cup_x, cup_y+40), (35, 8), 0, 0, 360, (180, 180, 180), -1)  # Cup shadow
    cv2.rectangle(img, (block_x-28, block_y+22), (block_x+28, block_y+30), (180, 180, 180), -1)  # Block shadow
    cv2.ellipse(img, (plate_x, plate_y+35), (45, 10), 0, 0, 360, (180, 180, 180), -1)  # Plate shadow
    
    return img

def main():
    """Create test video"""
    output_path = "/Users/tylergee/Documents/TANGRAM/data/raw_videos/test_tabletop.mp4"
    
    # Video parameters
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating test video: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Duration: {duration}s")
    
    for frame_num in range(total_frames):
        # Create frame
        frame = create_tabletop_scene(frame_num, total_frames)
        
        # Write frame
        out.write(frame)
        
        if frame_num % 30 == 0:
            print(f"  Frame {frame_num}/{total_frames}")
    
    # Release video writer
    out.release()
    
    print(f"✅ Test video created: {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()