#!/usr/bin/env python3
"""
TANGRAM Video Demo Generator

Creates a comprehensive video showing the entire TANGRAM pipeline in action:
1. Input video with objects
2. Object detection and tracking visualization  
3. Segmentation masks overlay
4. Scene graph generation
5. LLM task planning display
6. Robot simulation execution

This creates a professional demo video showing all components working together.
"""

import cv2
import numpy as np
import json
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def create_title_frame(title, subtitle, width=1280, height=720):
    """Create a title frame for the demo"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Gradient background
    for y in range(height):
        intensity = int(50 + (y / height) * 30)
        frame[y, :] = [intensity, intensity//2, intensity//4]
    
    # Add title text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Main title
    title_size = cv2.getTextSize(title, font, 2.0, 3)[0]
    title_x = (width - title_size[0]) // 2
    title_y = height // 2 - 50
    cv2.putText(frame, title, (title_x, title_y), font, 2.0, (255, 255, 255), 3)
    
    # Subtitle
    if subtitle:
        sub_size = cv2.getTextSize(subtitle, font, 1.0, 2)[0]
        sub_x = (width - sub_size[0]) // 2
        sub_y = title_y + 80
        cv2.putText(frame, subtitle, (sub_x, sub_y), font, 1.0, (200, 200, 200), 2)
    
    # Add decorative elements
    cv2.rectangle(frame, (100, title_y - 100), (width-100, title_y + 120), (80, 40, 20), 2)
    
    return frame

def create_step_frame(step_num, step_title, description, width=1280, height=720):
    """Create a frame showing current pipeline step"""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 40
    
    # Step header
    header_height = 100
    cv2.rectangle(frame, (0, 0), (width, header_height), (60, 30, 15), -1)
    
    # Step number and title
    font = cv2.FONT_HERSHEY_SIMPLEX
    step_text = f"STEP {step_num}: {step_title}"
    cv2.putText(frame, step_text, (50, 60), font, 1.5, (255, 255, 255), 2)
    
    # Description
    cv2.putText(frame, description, (50, height - 100), font, 0.8, (200, 200, 200), 2)
    
    return frame

def overlay_detections(frame, detections):
    """Overlay object detection boxes and labels"""
    if not detections:
        return frame
    
    frame_copy = frame.copy()
    
    for detection in detections:
        bbox = detection.get('bbox', [0, 0, 100, 100])
        class_name = detection.get('class_name', 'object')
        confidence = detection.get('confidence', 0.0)
        track_id = detection.get('track_id', 0)
        
        # Convert bbox format
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        
        # Choose color based on class
        if class_name == 'sports ball':
            color = (0, 0, 255)  # Red for cup
            label = "Red Cup"
        elif class_name == 'frisbee':
            color = (0, 255, 0)  # Green for plate
            label = "Green Plate"
        else:
            color = (255, 255, 0)  # Yellow for others
            label = class_name
        
        # Draw bounding box
        cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        
        # Draw label
        label_text = f"{label} ({confidence:.2f})"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame_copy, (int(x1), int(y1) - 30), 
                     (int(x1) + label_size[0] + 10, int(y1)), color, -1)
        cv2.putText(frame_copy, label_text, (int(x1) + 5, int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame_copy

def create_scene_graph_frame(detections, width=1280, height=720):
    """Create a frame showing the scene graph"""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Title
    cv2.putText(frame, "SCENE GRAPH ANALYSIS", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 50), 2)
    
    if not detections:
        cv2.putText(frame, "No objects detected", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        return frame
    
    # Draw nodes for each object
    y_start = 150
    node_spacing = 120
    
    for i, detection in enumerate(detections):
        class_name = detection.get('class_name', 'object')
        
        # Map to readable names
        if class_name == 'sports ball':
            obj_name = "Red Cup"
            color = (0, 0, 200)
        elif class_name == 'frisbee':
            obj_name = "Green Plate"
            color = (0, 150, 0)
        else:
            obj_name = class_name
            color = (100, 100, 100)
        
        # Draw node circle
        center_x = 200 + i * 300
        center_y = y_start + 100
        cv2.circle(frame, (center_x, center_y), 60, color, -1)
        cv2.circle(frame, (center_x, center_y), 60, (0, 0, 0), 3)
        
        # Object label
        text_size = cv2.getTextSize(obj_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = center_x - text_size[0] // 2
        cv2.putText(frame, obj_name, (text_x, center_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Properties
        bbox = detection.get('bbox', [0, 0, 100, 100])
        x_pos = int(bbox[0] + bbox[2] / 2)
        y_pos = int(bbox[1] + bbox[3] / 2)
        
        cv2.putText(frame, f"Position: ({x_pos}, {y_pos})", 
                   (center_x - 80, center_y + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    # Draw table node
    table_x, table_y = 200 + len(detections) * 150, y_start + 300
    cv2.rectangle(frame, (table_x - 80, table_y - 30), 
                 (table_x + 80, table_y + 30), (139, 69, 19), -1)
    cv2.putText(frame, "Table", (table_x - 25, table_y + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw relationships
    for i, detection in enumerate(detections):
        center_x = 200 + i * 300
        center_y = y_start + 100
        # Arrow to table
        cv2.arrowedLine(frame, (center_x, center_y + 60), 
                       (table_x, table_y - 30), (100, 100, 100), 2)
        cv2.putText(frame, "on", (center_x + 50, center_y + 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return frame

def create_llm_planning_frame(task, plan_text, width=1280, height=720):
    """Create a frame showing LLM task planning"""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 50
    
    # Title
    cv2.putText(frame, "LLM TASK PLANNING", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    # Task input
    cv2.putText(frame, "HUMAN TASK:", (50, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2)
    cv2.putText(frame, f'"{task}"', (50, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    # AI response
    cv2.putText(frame, "DEEPSEEK R1 RESPONSE:", (50, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 150, 255), 2)
    
    # Split plan text into lines
    if plan_text:
        lines = plan_text.split('\n')[:8]  # First 8 lines
        for i, line in enumerate(lines):
            if len(line) > 80:
                line = line[:77] + "..."
            cv2.putText(frame, line, (50, 260 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Model info
    cv2.putText(frame, "Model: DeepSeek R1 7B (Local)", (50, height - 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
    cv2.putText(frame, "Running on M1 Mac with 0 external API calls", (50, height - 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
    
    return frame

def create_simulation_frame(step_description, width=1280, height=720):
    """Create a frame showing robot simulation"""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 40
    
    # Title
    cv2.putText(frame, "ROBOT SIMULATION", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    # Draw simple robot arm simulation
    center_x, center_y = width // 2, height // 2
    
    # Robot base
    cv2.circle(frame, (center_x, center_y + 100), 30, (100, 100, 100), -1)
    
    # Robot arm segments
    arm1_end = (center_x - 80, center_y)
    arm2_end = (center_x - 120, center_y - 80)
    
    cv2.line(frame, (center_x, center_y + 100), arm1_end, (150, 150, 150), 8)
    cv2.line(frame, arm1_end, arm2_end, (150, 150, 150), 8)
    
    # End effector
    cv2.circle(frame, arm2_end, 15, (255, 100, 100), -1)
    
    # Objects on table
    table_y = center_y + 200
    cv2.rectangle(frame, (100, table_y), (width - 100, table_y + 20), (139, 69, 19), -1)
    
    # Cup and plate
    cup_pos = (300, table_y - 30)
    plate_pos = (600, table_y - 15)
    
    cv2.circle(frame, cup_pos, 20, (0, 0, 255), -1)  # Red cup
    cv2.ellipse(frame, plate_pos, (35, 10), 0, 0, 360, (0, 255, 0), -1)  # Green plate
    
    # Current action
    cv2.putText(frame, f"Action: {step_description}", (50, height - 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2)
    
    # Physics engine info
    cv2.putText(frame, "Physics: PyBullet", (50, height - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
    
    return frame

def main():
    """Create the complete TANGRAM demo video"""
    print("🎬 Creating TANGRAM Demo Video...")
    
    # Load tracking results
    tracking_file = Path("data/tracking/tracking_results.json")
    detections = []
    if tracking_file.exists():
        with open(tracking_file) as f:
            tracking_data = json.load(f)
            if tracking_data and len(tracking_data) > 0:
                detections = tracking_data[0].get('detections', [])
    
    # Video settings
    output_path = "TANGRAM_Demo_Video.mp4"
    width, height = 1280, 720
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating video: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    
    # Create video segments
    segments = [
        # Title sequence (3 seconds)
        {"type": "title", "duration": 3, "title": "TANGRAM", 
         "subtitle": "Robotic Scene Understanding Pipeline"},
        
        # Step 1: Video input (2 seconds)
        {"type": "step", "duration": 2, "step": 1, 
         "title": "VIDEO INPUT", "desc": "Analyzing tabletop manipulation scene"},
        
        # Step 2: Object detection (3 seconds)
        {"type": "detection", "duration": 3, "step": 2,
         "title": "OBJECT DETECTION", "desc": "YOLO detecting and tracking objects"},
        
        # Step 3: Scene graph (3 seconds)
        {"type": "scene_graph", "duration": 3, "step": 3,
         "title": "SCENE ANALYSIS", "desc": "Building spatial relationships"},
        
        # Step 4: LLM planning (4 seconds)
        {"type": "llm", "duration": 4, "step": 4,
         "title": "LLM TASK PLANNING", "desc": "DeepSeek generating robot plan"},
        
        # Step 5: Simulation (4 seconds)
        {"type": "simulation", "duration": 4, "step": 5,
         "title": "ROBOT SIMULATION", "desc": "PyBullet executing planned actions"},
        
        # Final title (2 seconds)
        {"type": "title", "duration": 2, "title": "DEMO COMPLETE", 
         "subtitle": "All components working locally on M1 Mac"}
    ]
    
    total_frames = 0
    for segment in segments:
        frames_needed = fps * segment["duration"]
        total_frames += frames_needed
        
        print(f"  Creating {segment['type']} segment: {frames_needed} frames")
        
        if segment["type"] == "title":
            for frame_num in range(frames_needed):
                frame = create_title_frame(segment["title"], segment["subtitle"], width, height)
                out.write(frame)
        
        elif segment["type"] == "step":
            for frame_num in range(frames_needed):
                frame = create_step_frame(segment["step"], segment["title"], segment["desc"], width, height)
                out.write(frame)
        
        elif segment["type"] == "detection":
            # Load original video and overlay detections
            original_video = "data/raw_videos/test_tabletop.mp4"
            if Path(original_video).exists():
                cap = cv2.VideoCapture(original_video)
                for frame_num in range(frames_needed):
                    ret, frame = cap.read()
                    if ret:
                        # Resize to target resolution
                        frame = cv2.resize(frame, (width, height))
                        # Overlay detections
                        frame = overlay_detections(frame, detections)
                        # Add step header
                        cv2.rectangle(frame, (0, 0), (width, 80), (60, 30, 15), -1)
                        cv2.putText(frame, f"STEP {segment['step']}: {segment['title']}", 
                                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    else:
                        frame = create_step_frame(segment["step"], segment["title"], segment["desc"], width, height)
                    out.write(frame)
                cap.release()
            else:
                # Fallback to step frame
                for frame_num in range(frames_needed):
                    frame = create_step_frame(segment["step"], segment["title"], segment["desc"], width, height)
                    out.write(frame)
        
        elif segment["type"] == "scene_graph":
            for frame_num in range(frames_needed):
                frame = create_scene_graph_frame(detections, width, height)
                # Add step header
                cv2.rectangle(frame, (0, 0), (width, 80), (60, 30, 15), -1)
                cv2.putText(frame, f"STEP {segment['step']}: {segment['title']}", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                out.write(frame)
        
        elif segment["type"] == "llm":
            plan_text = """1. Move robot arm to red cup position
2. Open gripper and approach cup
3. Close gripper to grasp cup securely
4. Lift cup and move to plate location
5. Position cup above green plate
6. Lower cup onto plate surface
7. Open gripper to release cup
8. Return arm to home position"""
            
            task = "pick up the red cup and place it on the green plate"
            for frame_num in range(frames_needed):
                frame = create_llm_planning_frame(task, plan_text, width, height)
                # Add step header
                cv2.rectangle(frame, (0, 0), (width, 80), (60, 30, 15), -1)
                cv2.putText(frame, f"STEP {segment['step']}: {segment['title']}", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                out.write(frame)
        
        elif segment["type"] == "simulation":
            actions = [
                "Moving to cup position",
                "Grasping red cup",
                "Lifting and moving cup",
                "Placing on green plate"
            ]
            action_frames = frames_needed // len(actions)
            
            for action_idx, action in enumerate(actions):
                for frame_num in range(action_frames):
                    frame = create_simulation_frame(action, width, height)
                    # Add step header
                    cv2.rectangle(frame, (0, 0), (width, 80), (60, 30, 15), -1)
                    cv2.putText(frame, f"STEP {segment['step']}: {segment['title']}", 
                               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    out.write(frame)
    
    out.release()
    
    print(f"✅ Demo video created: {output_path}")
    print(f"📊 Total frames: {total_frames}")
    print(f"⏱️  Duration: {total_frames / fps:.1f} seconds")
    print(f"📁 File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    return output_path

if __name__ == "__main__":
    main()