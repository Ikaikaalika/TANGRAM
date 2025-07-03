#!/usr/bin/env python3
"""
Create a realistic video with multiple objects scattered on a table
for TANGRAM pipeline demonstration.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import random
import math

def create_realistic_table_scene(frame_num, total_frames):
    """Create a frame with multiple realistic objects on a table"""
    # Create base image (1280x720 for better quality)
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 235  # Light background
    
    # Create realistic wooden table
    table_y_start = 400
    table_y_end = 680
    table_x_start = 100
    table_x_end = 1180
    
    # Table surface (wood grain effect)
    wood_color = (139, 119, 101)  # Brown wood
    cv2.rectangle(img, (table_x_start, table_y_start), (table_x_end, table_y_end), wood_color, -1)
    
    # Add wood grain texture
    for i in range(table_x_start, table_x_end, 20):
        grain_color = (wood_color[0] + random.randint(-15, 15), 
                      wood_color[1] + random.randint(-15, 15), 
                      wood_color[2] + random.randint(-15, 15))
        cv2.line(img, (i, table_y_start), (i, table_y_end), grain_color, 2)
    
    # Table edge highlighting
    cv2.rectangle(img, (table_x_start, table_y_start), (table_x_end, table_y_start + 15), 
                  (180, 150, 120), -1)
    
    # Define multiple objects with slight animation
    objects = [
        # Red apple
        {
            'type': 'circle',
            'base_pos': (250, 450),
            'radius': 35,
            'color': (0, 0, 220),  # Red
            'shadow_offset': (8, 15),
            'name': 'apple'
        },
        # Orange/sports ball
        {
            'type': 'circle', 
            'base_pos': (400, 480),
            'radius': 30,
            'color': (0, 165, 255),  # Orange
            'shadow_offset': (7, 12),
            'name': 'orange'
        },
        # Blue cup
        {
            'type': 'cup',
            'base_pos': (580, 460),
            'size': (25, 40),
            'color': (255, 100, 0),  # Blue
            'shadow_offset': (6, 18),
            'name': 'cup'
        },
        # Green bottle
        {
            'type': 'bottle',
            'base_pos': (750, 440),
            'size': (15, 55),
            'color': (0, 180, 0),  # Green
            'shadow_offset': (8, 22),
            'name': 'bottle'
        },
        # Yellow banana
        {
            'type': 'ellipse',
            'base_pos': (920, 490),
            'size': (50, 20),
            'color': (0, 255, 255),  # Yellow
            'shadow_offset': (10, 8),
            'name': 'banana'
        },
        # Purple book
        {
            'type': 'rectangle',
            'base_pos': (300, 520),
            'size': (60, 40),
            'color': (180, 0, 180),  # Purple
            'shadow_offset': (12, 15),
            'name': 'book'
        },
        # Pink phone
        {
            'type': 'rectangle',
            'base_pos': (650, 520),
            'size': (25, 45),
            'color': (180, 105, 255),  # Pink
            'shadow_offset': (5, 18),
            'name': 'phone'
        }
    ]
    
    # Add slight movement to simulate camera or lighting changes
    movement_offset = 2 * math.sin(frame_num * 0.05)
    
    # Draw shadows first
    for obj in objects:
        shadow_pos = (obj['base_pos'][0] + obj['shadow_offset'][0] + int(movement_offset),
                     obj['base_pos'][1] + obj['shadow_offset'][1])
        shadow_color = (120, 120, 120)
        
        if obj['type'] == 'circle':
            cv2.circle(img, shadow_pos, obj['radius'] + 2, shadow_color, -1)
        elif obj['type'] == 'cup':
            cv2.ellipse(img, shadow_pos, (obj['size'][0] + 5, obj['size'][1] // 3), 0, 0, 360, shadow_color, -1)
        elif obj['type'] == 'bottle':
            cv2.ellipse(img, shadow_pos, (obj['size'][0] + 3, obj['size'][1] // 4), 0, 0, 360, shadow_color, -1)
        elif obj['type'] == 'ellipse':
            cv2.ellipse(img, shadow_pos, (obj['size'][0] + 5, obj['size'][1] + 3), 0, 0, 360, shadow_color, -1)
        elif obj['type'] == 'rectangle':
            cv2.rectangle(img, 
                         (shadow_pos[0] - obj['size'][0]//2, shadow_pos[1] - obj['size'][1]//2),
                         (shadow_pos[0] + obj['size'][0]//2, shadow_pos[1] + obj['size'][1]//2),
                         shadow_color, -1)
    
    # Draw objects
    for obj in objects:
        pos = (obj['base_pos'][0] + int(movement_offset), obj['base_pos'][1])
        color = obj['color']
        
        if obj['type'] == 'circle':
            # Draw main circle
            cv2.circle(img, pos, obj['radius'], color, -1)
            # Add highlight
            highlight_pos = (pos[0] - obj['radius']//3, pos[1] - obj['radius']//3)
            cv2.circle(img, highlight_pos, obj['radius']//4, 
                      (min(255, color[0] + 80), min(255, color[1] + 80), min(255, color[2] + 80)), -1)
            # Add outline
            cv2.circle(img, pos, obj['radius'], 
                      (max(0, color[0] - 40), max(0, color[1] - 40), max(0, color[2] - 40)), 2)
        
        elif obj['type'] == 'cup':
            # Draw cup body
            cv2.ellipse(img, pos, obj['size'], 0, 0, 360, color, -1)
            # Draw cup rim
            rim_pos = (pos[0], pos[1] - obj['size'][1])
            cv2.ellipse(img, rim_pos, (obj['size'][0], obj['size'][1]//4), 0, 0, 360, color, -1)
            # Add highlight
            cv2.ellipse(img, (pos[0] - 8, pos[1] - 10), (8, 15), 0, 0, 360, 
                       (min(255, color[0] + 60), min(255, color[1] + 60), min(255, color[2] + 60)), -1)
        
        elif obj['type'] == 'bottle':
            # Draw bottle body
            cv2.rectangle(img, 
                         (pos[0] - obj['size'][0]//2, pos[1] - obj['size'][1]//2),
                         (pos[0] + obj['size'][0]//2, pos[1] + obj['size'][1]//2),
                         color, -1)
            # Draw bottle neck
            neck_width = obj['size'][0] // 2
            cv2.rectangle(img, 
                         (pos[0] - neck_width//2, pos[1] - obj['size'][1]//2 - 15),
                         (pos[0] + neck_width//2, pos[1] - obj['size'][1]//2),
                         color, -1)
            # Add highlight
            cv2.rectangle(img, 
                         (pos[0] - obj['size'][0]//4, pos[1] - obj['size'][1]//2),
                         (pos[0] - obj['size'][0]//6, pos[1] + obj['size'][1]//2),
                         (min(255, color[0] + 40), min(255, color[1] + 40), min(255, color[2] + 40)), -1)
        
        elif obj['type'] == 'ellipse':
            # Draw ellipse (banana)
            cv2.ellipse(img, pos, obj['size'], 15, 0, 360, color, -1)
            # Add banana curve highlight
            cv2.ellipse(img, (pos[0] - 10, pos[1] - 5), (obj['size'][0]//3, obj['size'][1]//3), 15, 0, 360,
                       (min(255, color[0] + 50), min(255, color[1] + 50), min(255, color[2] + 50)), -1)
        
        elif obj['type'] == 'rectangle':
            # Draw rectangle
            cv2.rectangle(img, 
                         (pos[0] - obj['size'][0]//2, pos[1] - obj['size'][1]//2),
                         (pos[0] + obj['size'][0]//2, pos[1] + obj['size'][1]//2),
                         color, -1)
            # Add book spine or phone edge
            if obj['name'] == 'book':
                cv2.rectangle(img, 
                             (pos[0] - obj['size'][0]//2, pos[1] - obj['size'][1]//2),
                             (pos[0] - obj['size'][0]//2 + 8, pos[1] + obj['size'][1]//2),
                             (max(0, color[0] - 60), max(0, color[1] - 60), max(0, color[2] - 60)), -1)
            else:  # phone
                cv2.rectangle(img, 
                             (pos[0] - obj['size'][0]//2 + 3, pos[1] - obj['size'][1]//2 + 3),
                             (pos[0] + obj['size'][0]//2 - 3, pos[1] + obj['size'][1]//2 - 3),
                             (20, 20, 20), -1)  # Phone screen
    
    # Add some ambient lighting effects
    lighting_alpha = 0.1
    lighting_overlay = img.copy()
    cv2.circle(lighting_overlay, (640, 200), 400, (255, 255, 255), -1)
    img = cv2.addWeighted(img, 1 - lighting_alpha, lighting_overlay, lighting_alpha, 0)
    
    return img

def main():
    """Create multi-object tabletop video"""
    output_path = "/Users/tylergee/Documents/TANGRAM/data/raw_videos/multi_object_scene.mp4"
    
    # Video parameters - higher quality
    width, height = 1280, 720
    fps = 30
    duration = 8  # seconds - longer for better analysis
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating multi-object scene video: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Duration: {duration}s")
    print("Objects: apple, orange, cup, bottle, banana, book, phone")
    
    for frame_num in range(total_frames):
        # Create frame with multiple objects
        frame = create_realistic_table_scene(frame_num, total_frames)
        
        # Write frame
        out.write(frame)
        
        if frame_num % 60 == 0:
            print(f"  Frame {frame_num}/{total_frames}")
    
    # Release video writer
    out.release()
    
    print(f"✅ Multi-object video created: {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()