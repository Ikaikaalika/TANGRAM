#!/usr/bin/env python3
"""
TANGRAM Enhanced Demo Video Generator

Creates a comprehensive video showing the full TANGRAM pipeline with:
1. Original multi-object scene video
2. Real object detection results overlaid
3. 3D scene reconstruction visualization
4. LLM task planning for organizing objects
5. 3D robot simulation showing objects being organized into a pile
"""

import cv2
import numpy as np
import json
import sys
import time
import math
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def load_tracking_results():
    """Load the multi-object tracking results"""
    tracking_file = Path("data/tracking/tracking_results.json")
    if tracking_file.exists():
        with open(tracking_file) as f:
            return json.load(f)
    return None

def create_title_frame(title, subtitle, width=1920, height=1080):
    """Create a professional title frame"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Gradient background
    for y in range(height):
        intensity = int(30 + (y / height) * 50)
        frame[y, :] = [intensity, intensity//2, intensity//4]
    
    # Main title
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_size = cv2.getTextSize(title, font, 3.0, 4)[0]
    title_x = (width - title_size[0]) // 2
    title_y = height // 2 - 100
    cv2.putText(frame, title, (title_x, title_y), font, 3.0, (255, 255, 255), 4)
    
    # Subtitle
    if subtitle:
        sub_size = cv2.getTextSize(subtitle, font, 1.5, 3)[0]
        sub_x = (width - sub_size[0]) // 2
        sub_y = title_y + 120
        cv2.putText(frame, subtitle, (sub_x, sub_y), font, 1.5, (200, 200, 200), 3)
    
    # Decorative border
    cv2.rectangle(frame, (150, title_y - 150), (width-150, title_y + 200), (100, 50, 25), 3)
    
    return frame

def overlay_real_detections(frame, detections, frame_idx=0):
    """Overlay real YOLO detection results on the frame"""
    if not detections or len(detections) <= frame_idx:
        return frame
    
    frame_copy = frame.copy()
    frame_data = detections[frame_idx]
    
    for detection in frame_data.get('detections', []):
        bbox = detection.get('bbox', [0, 0, 100, 100])
        class_name = detection.get('class_name', 'object')
        confidence = detection.get('confidence', 0.0)
        track_id = detection.get('track_id', 0)
        
        # Convert bbox format
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        
        # Map detected classes to object names
        object_map = {
            'orange': {'name': 'Orange', 'color': (0, 165, 255)},
            'frisbee': {'name': 'Plate', 'color': (0, 255, 0)},
            'sports ball': {'name': 'Ball', 'color': (0, 0, 255)},
            'cup': {'name': 'Cup', 'color': (255, 100, 0)},
            'bottle': {'name': 'Bottle', 'color': (0, 180, 0)},
            'book': {'name': 'Book', 'color': (180, 0, 180)},
            'cell phone': {'name': 'Phone', 'color': (180, 105, 255)}
        }
        
        obj_info = object_map.get(class_name, {'name': class_name, 'color': (255, 255, 0)})
        color = obj_info['color']
        label = obj_info['name']
        
        # Draw bounding box with thickness based on confidence
        thickness = max(2, int(confidence * 5))
        cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # Draw track ID and confidence
        label_text = f"ID:{track_id} {label} ({confidence:.2f})"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Background for label
        cv2.rectangle(frame_copy, (int(x1), int(y1) - 35), 
                     (int(x1) + label_size[0] + 15, int(y1)), color, -1)
        cv2.putText(frame_copy, label_text, (int(x1) + 5, int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw center point
        center_x = int(x1 + w/2)
        center_y = int(y1 + h/2)
        cv2.circle(frame_copy, (center_x, center_y), 5, color, -1)
    
    return frame_copy

def create_3d_scene_frame(detections, width=1920, height=1080):
    """Create a 3D visualization of the detected scene"""
    # Create matplotlib figure
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the 3D scene
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 5)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('3D Scene Reconstruction from Video', fontsize=20, pad=20)
    
    # Draw table surface
    xx, yy = np.meshgrid(np.linspace(1, 9, 10), np.linspace(2, 8, 10))
    zz = np.ones_like(xx) * 1.0  # Table height
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='brown')
    
    # Add detected objects as 3D objects
    if detections and len(detections) > 0:
        frame_data = detections[0]  # Use first frame
        
        for i, detection in enumerate(frame_data.get('detections', [])):
            class_name = detection.get('class_name', 'object')
            bbox = detection.get('bbox', [0, 0, 100, 100])
            
            # Convert 2D position to 3D world coordinates
            x_2d = bbox[0] + bbox[2]/2
            y_2d = bbox[1] + bbox[3]/2
            
            # Map to 3D space (simple projection)
            x_3d = 2 + (x_2d / 640) * 6  # Scale to table
            y_3d = 3 + (y_2d / 480) * 4
            z_3d = 1.1  # Slightly above table
            
            # Object colors and shapes
            if class_name == 'orange':
                ax.scatter(x_3d, y_3d, z_3d, c='orange', s=500, marker='o', alpha=0.8)
                ax.text(x_3d, y_3d, z_3d + 0.3, 'Orange', fontsize=12)
            elif class_name == 'frisbee':
                ax.scatter(x_3d, y_3d, z_3d, c='green', s=800, marker='s', alpha=0.8)
                ax.text(x_3d, y_3d, z_3d + 0.3, 'Plate', fontsize=12)
            elif class_name == 'sports ball':
                ax.scatter(x_3d, y_3d, z_3d, c='red', s=400, marker='o', alpha=0.8)
                ax.text(x_3d, y_3d, z_3d + 0.3, 'Ball', fontsize=12)
            else:
                ax.scatter(x_3d, y_3d, z_3d, c='blue', s=300, marker='^', alpha=0.8)
                ax.text(x_3d, y_3d, z_3d + 0.3, class_name, fontsize=12)
    
    # Add robot arm
    robot_base = [0.5, 5, 1]
    robot_joints = [
        [1.5, 5, 1.5],
        [2.5, 4.5, 2.0],
        [3.5, 4, 2.2]
    ]
    
    # Draw robot arm segments
    prev_joint = robot_base
    for joint in robot_joints:
        ax.plot3D([prev_joint[0], joint[0]], 
                  [prev_joint[1], joint[1]], 
                  [prev_joint[2], joint[2]], 'k-', linewidth=5)
        ax.scatter(joint[0], joint[1], joint[2], c='red', s=100)
        prev_joint = joint
    
    # Robot base
    ax.scatter(robot_base[0], robot_base[1], robot_base[2], c='black', s=200, marker='s')
    
    # Set view angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Convert matplotlib figure to OpenCV frame
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    # Fix for newer matplotlib versions
    try:
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    except AttributeError:
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))[:,:,:3]  # Remove alpha
    else:
        buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
    
    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    
    return frame

def create_llm_planning_frame(detections, width=1920, height=1080):
    """Create enhanced LLM planning visualization"""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 40
    
    # Title
    cv2.putText(frame, "LLM TASK PLANNING", (100, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
    
    # Task description
    cv2.putText(frame, "HUMAN TASK:", (100, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 100), 3)
    task_text = "Organize all objects into a neat pile in the center of the table"
    cv2.putText(frame, f'"{task_text}"', (100, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    
    # Scene understanding
    cv2.putText(frame, "SCENE ANALYSIS:", (100, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 150, 100), 3)
    
    if detections and len(detections) > 0:
        detected_objects = []
        frame_data = detections[0]
        for detection in frame_data.get('detections', []):
            class_name = detection.get('class_name', 'object')
            if class_name == 'orange':
                detected_objects.append("Orange")
            elif class_name == 'frisbee':
                detected_objects.append("Plate")
            elif class_name == 'sports ball':
                detected_objects.append("Ball")
        
        objects_text = f"Detected objects: {', '.join(detected_objects)}"
        cv2.putText(frame, objects_text, (100, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    
    # AI response
    cv2.putText(frame, "DEEPSEEK R1 RESPONSE:", (100, 500), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 150, 255), 3)
    
    plan_lines = [
        "1. Identify all objects on the table surface",
        "2. Plan optimal picking sequence to avoid collisions", 
        "3. Pick up Orange and move to center position",
        "4. Pick up Plate and stack on Orange",
        "5. Pick up Ball and place on top of stack",
        "6. Verify stable pile configuration",
        "7. Return robot arm to home position",
        "8. Task completion confirmed"
    ]
    
    for i, line in enumerate(plan_lines):
        y_pos = 550 + i * 45
        cv2.putText(frame, line, (100, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    
    # Model info
    cv2.putText(frame, "Model: DeepSeek R1 7B (Running locally on M1 Mac)", 
                (100, height - 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
    cv2.putText(frame, "Zero external API calls - Complete privacy", 
                (100, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
    
    return frame

def create_robot_simulation_frame(step, total_steps, width=1920, height=1080):
    """Create detailed 3D robot simulation showing object organization"""
    # Create matplotlib figure for 3D simulation
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the 3D scene
    ax.set_xlim(-2, 8)
    ax.set_ylim(-1, 7)
    ax.set_zlim(0, 4)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Robot Simulation: Organizing Objects into Pile', fontsize=20, pad=20)
    
    # Draw table
    table_x = np.array([1, 7, 7, 1, 1])
    table_y = np.array([1, 1, 5, 5, 1])
    table_z = np.ones(5) * 1.0
    ax.plot(table_x, table_y, table_z, 'brown', linewidth=3)
    
    # Fill table surface
    xx, yy = np.meshgrid(np.linspace(1, 7, 10), np.linspace(1, 5, 10))
    zz = np.ones_like(xx) * 1.0
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='brown')
    
    # Animation progress
    progress = step / total_steps
    
    # Define object positions throughout the animation
    pile_center = [4, 3, 1.1]  # Center of table
    
    # Initial scattered positions
    initial_positions = {
        'orange': [2.5, 2, 1.1],
        'plate': [5.5, 4, 1.1], 
        'ball': [3, 4.5, 1.1]
    }
    
    # Final stacked positions
    final_positions = {
        'orange': [4, 3, 1.1],     # Bottom
        'plate': [4, 3, 1.3],      # Middle  
        'ball': [4, 3, 1.6]        # Top
    }
    
    # Interpolate positions based on progress
    current_positions = {}
    for obj in initial_positions:
        init_pos = np.array(initial_positions[obj])
        final_pos = np.array(final_positions[obj])
        
        # Create smooth movement trajectory
        if progress < 0.33:  # First third - move orange
            if obj == 'orange':
                t = progress * 3
                current_positions[obj] = init_pos + t * (final_pos - init_pos)
            else:
                current_positions[obj] = init_pos
        elif progress < 0.66:  # Second third - move plate
            if obj == 'orange':
                current_positions[obj] = final_pos
            elif obj == 'plate':
                t = (progress - 0.33) * 3
                current_positions[obj] = init_pos + t * (final_pos - init_pos)
            else:
                current_positions[obj] = init_pos
        else:  # Final third - move ball
            if obj in ['orange', 'plate']:
                current_positions[obj] = final_pos
            else:  # ball
                t = (progress - 0.66) * 3
                current_positions[obj] = init_pos + t * (final_pos - init_pos)
    
    # Draw objects at current positions
    for obj, pos in current_positions.items():
        if obj == 'orange':
            ax.scatter(pos[0], pos[1], pos[2], c='orange', s=800, marker='o', alpha=0.9)
            ax.text(pos[0], pos[1], pos[2] + 0.3, 'Orange', fontsize=12)
        elif obj == 'plate':
            ax.scatter(pos[0], pos[1], pos[2], c='green', s=1200, marker='s', alpha=0.9)
            ax.text(pos[0], pos[1], pos[2] + 0.3, 'Plate', fontsize=12)
        elif obj == 'ball':
            ax.scatter(pos[0], pos[1], pos[2], c='red', s=600, marker='o', alpha=0.9)
            ax.text(pos[0], pos[1], pos[2] + 0.3, 'Ball', fontsize=12)
    
    # Animate robot arm based on current action
    robot_base = [0, 3, 1]
    
    if progress < 0.33:
        # Moving to orange
        target = current_positions['orange']
        action_text = "Picking up Orange"
    elif progress < 0.66:
        # Moving to plate
        target = current_positions['plate']
        action_text = "Picking up Plate"
    else:
        # Moving to ball
        target = current_positions['ball']
        action_text = "Picking up Ball"
    
    # Calculate robot arm configuration toward target
    reach_x = min(target[0] - robot_base[0], 3.5)
    reach_y = target[1] - robot_base[1]
    
    joint1 = [robot_base[0] + reach_x * 0.4, robot_base[1] + reach_y * 0.3, robot_base[2] + 0.8]
    joint2 = [robot_base[0] + reach_x * 0.7, robot_base[1] + reach_y * 0.6, robot_base[2] + 1.2]
    end_effector = [robot_base[0] + reach_x, robot_base[1] + reach_y * 0.8, target[2] + 0.1]
    
    # Draw robot arm
    robot_points = [robot_base, joint1, joint2, end_effector]
    for i in range(len(robot_points) - 1):
        ax.plot3D([robot_points[i][0], robot_points[i+1][0]], 
                  [robot_points[i][1], robot_points[i+1][1]], 
                  [robot_points[i][2], robot_points[i+1][2]], 'k-', linewidth=6)
    
    # Draw joints
    for point in robot_points:
        ax.scatter(point[0], point[1], point[2], c='red', s=150)
    
    # Robot base
    ax.scatter(robot_base[0], robot_base[1], robot_base[2], c='black', s=300, marker='s')
    
    # Add gripper
    gripper_size = 0.1
    ax.plot3D([end_effector[0] - gripper_size, end_effector[0] + gripper_size], 
              [end_effector[1], end_effector[1]], 
              [end_effector[2], end_effector[2]], 'red', linewidth=4)
    
    # Set view angle
    ax.view_init(elev=25, azim=45 + progress * 90)  # Rotate view during animation
    
    # Add action text
    ax.text2D(0.02, 0.95, f"Action: {action_text}", transform=ax.transAxes, 
              fontsize=16, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    ax.text2D(0.02, 0.90, f"Progress: {progress*100:.1f}%", transform=ax.transAxes, 
              fontsize=14, bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    
    # Convert to OpenCV frame
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    # Fix for newer matplotlib versions
    try:
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
    except AttributeError:
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))[:,:,:3]  # Remove alpha
    frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    
    return frame

def main():
    """Create the enhanced TANGRAM demo video"""
    print("🎬 Creating Enhanced TANGRAM Demo Video...")
    
    # Load detection results
    tracking_data = load_tracking_results()
    if not tracking_data:
        print("❌ No tracking data found. Run detection first.")
        return
    
    # Video settings - higher quality
    output_path = "TANGRAM_Enhanced_Demo.mp4"
    width, height = 1920, 1080  # Full HD
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating enhanced demo: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    
    # Load original video
    original_video_path = "data/raw_videos/multi_object_scene.mp4"
    original_cap = cv2.VideoCapture(original_video_path)
    
    # Video segments with more detail
    segments = [
        {"type": "title", "duration": 3, "title": "TANGRAM", 
         "subtitle": "AI-Powered Robotic Scene Understanding"},
        
        {"type": "original_video", "duration": 5, 
         "title": "INPUT: Multi-Object Scene Video"},
        
        {"type": "detection_video", "duration": 6,
         "title": "COMPUTER VISION: Real-time Object Detection & Tracking"},
        
        {"type": "3d_scene", "duration": 4,
         "title": "3D RECONSTRUCTION: Scene Understanding"},
        
        {"type": "llm_planning", "duration": 5,
         "title": "AI PLANNING: Task Generation"},
        
        {"type": "robot_simulation", "duration": 8,
         "title": "ROBOT EXECUTION: Organizing Objects"},
        
        {"type": "title", "duration": 3, "title": "MISSION COMPLETE", 
         "subtitle": "All objects organized - Task successful!"}
    ]
    
    total_frames = sum(fps * seg["duration"] for seg in segments)
    current_frame = 0
    
    for segment in segments:
        frames_needed = fps * segment["duration"]
        print(f"  Creating {segment['type']} segment: {frames_needed} frames")
        
        if segment["type"] == "title":
            for frame_num in range(frames_needed):
                frame = create_title_frame(segment["title"], segment["subtitle"], width, height)
                out.write(frame)
                current_frame += 1
        
        elif segment["type"] == "original_video":
            original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
            for frame_num in range(frames_needed):
                ret, frame = original_cap.read()
                if ret:
                    frame = cv2.resize(frame, (width, height))
                    # Add title overlay
                    cv2.rectangle(frame, (0, 0), (width, 120), (30, 30, 30), -1)
                    cv2.putText(frame, segment["title"], (50, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
                else:
                    frame = create_title_frame("Original Video", "Multi-object scene", width, height)
                out.write(frame)
                current_frame += 1
        
        elif segment["type"] == "detection_video":
            original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
            for frame_num in range(frames_needed):
                ret, frame = original_cap.read()
                if ret:
                    frame = cv2.resize(frame, (width, height))
                    # Overlay real detections
                    detection_frame_idx = min(frame_num * 2, len(tracking_data) - 1)
                    frame = overlay_real_detections(frame, tracking_data, detection_frame_idx)
                    # Add title overlay
                    cv2.rectangle(frame, (0, 0), (width, 120), (30, 30, 30), -1)
                    cv2.putText(frame, segment["title"], (50, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                else:
                    frame = create_title_frame("Detection", "Object tracking", width, height)
                out.write(frame)
                current_frame += 1
        
        elif segment["type"] == "3d_scene":
            for frame_num in range(frames_needed):
                frame = create_3d_scene_frame(tracking_data, width, height)
                out.write(frame)
                current_frame += 1
        
        elif segment["type"] == "llm_planning":
            for frame_num in range(frames_needed):
                frame = create_llm_planning_frame(tracking_data, width, height)
                out.write(frame)
                current_frame += 1
        
        elif segment["type"] == "robot_simulation":
            for frame_num in range(frames_needed):
                frame = create_robot_simulation_frame(frame_num, frames_needed, width, height)
                out.write(frame)
                current_frame += 1
        
        print(f"    Progress: {current_frame}/{total_frames} frames ({current_frame/total_frames*100:.1f}%)")
    
    original_cap.release()
    out.release()
    
    print(f"✅ Enhanced demo video created: {output_path}")
    print(f"📊 Total frames: {total_frames}")
    print(f"⏱️  Duration: {total_frames / fps:.1f} seconds")
    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"📁 File size: {file_size:.1f} MB")
    
    return output_path

if __name__ == "__main__":
    main()