#!/usr/bin/env python3
"""
TANGRAM Core Module - Simplified Pipeline

Consolidated core functionality for TANGRAM robotic scene understanding.
This single module contains all essential components previously spread across
multiple files.
"""

import cv2
import numpy as np
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Core ML imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class ObjectDetector:
    """Simplified object detection using YOLO"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available. Install with: pip install ultralytics")
        
        self.model = YOLO(model_path)
        self.device = "mps"  # Apple Silicon
        
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """Detect objects in image"""
        results = self.model(image, device=self.device, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    detection = {
                        "bbox": [int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])],
                        "confidence": float(conf),
                        "class_id": cls,
                        "class_name": self.model.names[cls]
                    }
                    detections.append(detection)
        
        return detections


class ObjectTracker:
    """Simplified object tracking"""
    
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        
    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """Simple tracking based on IoU overlap"""
        tracked_objects = []
        
        for detection in detections:
            track_id = self._assign_track_id(detection)
            detection["track_id"] = track_id
            tracked_objects.append(detection)
        
        return tracked_objects
    
    def _assign_track_id(self, detection: Dict) -> int:
        """Assign track ID based on position similarity"""
        bbox = detection["bbox"]
        center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
        
        # Simple distance-based assignment
        min_dist = float('inf')
        assigned_id = None
        
        for track_id, track_info in self.tracks.items():
            if track_info["class_name"] == detection["class_name"]:
                track_center = track_info["last_center"]
                dist = np.sqrt((center[0] - track_center[0])**2 + (center[1] - track_center[1])**2)
                if dist < min_dist and dist < 100:  # threshold
                    min_dist = dist
                    assigned_id = track_id
        
        if assigned_id is None:
            assigned_id = self.next_id
            self.next_id += 1
        
        self.tracks[assigned_id] = {
            "last_center": center,
            "class_name": detection["class_name"],
            "last_seen": time.time()
        }
        
        return assigned_id


class SceneGraph:
    """Simplified scene graph construction"""
    
    def __init__(self):
        if not NETWORKX_AVAILABLE:
            self.graph = {}  # Fallback to dict
        else:
            self.graph = nx.Graph()
        self.objects = {}
        
    def add_objects(self, detections: List[Dict]):
        """Add objects to scene graph"""
        for detection in detections:
            obj_id = f"{detection['class_name']}_{detection['track_id']}"
            
            # Convert 2D position to estimated 3D
            bbox = detection["bbox"]
            x_2d = bbox[0] + bbox[2]//2
            y_2d = bbox[1] + bbox[3]//2
            
            # Simple 2D to 3D mapping (assumes table surface)
            x_3d = 2 + (x_2d / 800) * 4  # Map to 2-6 meter range
            y_3d = 2 + (y_2d / 600) * 2  # Map to 2-4 meter range
            z_3d = 1.1  # On table surface
            
            self.objects[obj_id] = {
                "position": [x_3d, y_3d, z_3d],
                "class": detection["class_name"],
                "confidence": detection["confidence"],
                "bbox": bbox
            }
            
            if NETWORKX_AVAILABLE:
                self.graph.add_node(obj_id, **self.objects[obj_id])
        
        self._compute_relationships()
    
    def _compute_relationships(self):
        """Compute spatial relationships between objects"""
        objects = list(self.objects.keys())
        
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                pos1 = self.objects[obj1]["position"]
                pos2 = self.objects[obj2]["position"]
                
                # Calculate distance
                dist = np.sqrt(sum((a-b)**2 for a, b in zip(pos1, pos2)))
                
                # Define relationships
                if dist < 0.5:
                    relationship = "near"
                elif dist < 0.1:
                    relationship = "touching"
                elif abs(pos1[2] - pos2[2]) < 0.05:
                    relationship = "on_same_surface"
                else:
                    relationship = "separate"
                
                if NETWORKX_AVAILABLE:
                    self.graph.add_edge(obj1, obj2, relationship=relationship, distance=dist)


class LocalLLM:
    """Simplified local LLM interface"""
    
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.host = host
        self.port = port
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if local LLM server is available"""
        if not REQUESTS_AVAILABLE:
            return False
        
        try:
            response = requests.get(f"http://{self.host}:{self.port}", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from local LLM"""
        if not self.available:
            return "❌ Local LLM not available. Please start Ollama server."
        
        try:
            payload = {
                "model": "deepseek-r1:7b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"http://{self.host}:{self.port}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"❌ LLM Error: {response.status_code}"
                
        except Exception as e:
            return f"❌ LLM Error: {str(e)}"


class RobotSimulator:
    """Simplified robot simulation"""
    
    def __init__(self):
        self.available = PYBULLET_AVAILABLE
        self.physics_client = None
        self.robot_id = None
        self.position = [0, 3, 1.5]  # Default position
        
    def initialize(self, gui: bool = False) -> bool:
        """Initialize robot simulation"""
        if not self.available:
            print("⚠️ PyBullet not available. Robot simulation disabled.")
            return False
        
        try:
            if gui:
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)
            
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1/240)
            
            # Load simple robot (or create basic arm)
            self.robot_id = self._create_simple_robot()
            
            return True
            
        except Exception as e:
            print(f"❌ Robot simulation failed: {e}")
            return False
    
    def _create_simple_robot(self) -> int:
        """Create a simple robot representation"""
        # Create a simple box as robot base
        base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        robot_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=base_shape,
                                   basePosition=self.position)
        return robot_id
    
    def move_to(self, target_position: List[float]) -> bool:
        """Move robot to target position"""
        if not self.available or self.robot_id is None:
            # Simulate movement without physics
            self.position = target_position.copy()
            return True
        
        try:
            # Simple position update for demonstration
            p.resetBasePositionAndOrientation(
                self.robot_id, target_position, [0, 0, 0, 1]
            )
            self.position = target_position.copy()
            
            # Step simulation
            for _ in range(100):
                p.stepSimulation()
                time.sleep(1/240)
            
            return True
            
        except Exception as e:
            print(f"❌ Robot movement failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup simulation resources"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)


class TANGRAMCore:
    """Main TANGRAM pipeline controller - simplified version"""
    
    def __init__(self):
        self.detector = None
        self.tracker = ObjectTracker()
        self.scene_graph = SceneGraph()
        self.llm = LocalLLM()
        self.robot = RobotSimulator()
        
        # Initialize detector if available
        try:
            self.detector = ObjectDetector()
            print("✅ Object detector initialized")
        except Exception as e:
            print(f"⚠️ Object detector failed: {e}")
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Process single image through pipeline"""
        results = {
            "detections": [],
            "tracked_objects": [],
            "scene_graph": {},
            "timestamp": time.time()
        }
        
        if self.detector is None:
            return results
        
        try:
            # 1. Object Detection
            detections = self.detector.detect_objects(image)
            results["detections"] = detections
            
            # 2. Object Tracking
            tracked_objects = self.tracker.update_tracks(detections)
            results["tracked_objects"] = tracked_objects
            
            # 3. Scene Graph
            self.scene_graph.add_objects(tracked_objects)
            results["scene_graph"] = {
                "objects": self.scene_graph.objects,
                "num_objects": len(self.scene_graph.objects)
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return results
    
    def process_video(self, video_path: str) -> List[Dict[str, Any]]:
        """Process video file"""
        results = []
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame for efficiency
            if frame_count % 5 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_results = self.process_image(frame_rgb)
                frame_results["frame_id"] = frame_count
                results.append(frame_results)
            
            frame_count += 1
        
        cap.release()
        return results
    
    def execute_command(self, command: str, scene_data: Dict = None) -> str:
        """Execute natural language command"""
        if scene_data is None:
            scene_data = {"objects": {}}
        
        # Build context for LLM
        context = f"""
Current scene: {len(scene_data.get('objects', {}))} objects detected
Objects: {list(scene_data.get('objects', {}).keys())}
Robot position: {self.robot.position}

Command: {command}

Please provide a brief action plan for the robot:
"""
        
        # Get LLM response
        response = self.llm.generate_response(context)
        
        # Simple command execution
        if "move" in command.lower() or "go" in command.lower():
            if "center" in command.lower():
                success = self.robot.move_to([4, 3, 1.5])
                if success:
                    response += "\n✅ Moved to center position"
            elif "home" in command.lower():
                success = self.robot.move_to([0, 3, 1.5])
                if success:
                    response += "\n✅ Returned to home position"
        
        return response
    
    def visualize_scene(self, scene_data: Dict = None) -> plt.Figure:
        """Create 3D visualization of scene"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw table
        xx, yy = np.meshgrid(np.linspace(1, 7, 10), np.linspace(1, 5, 10))
        zz = np.ones_like(xx) * 1.0
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='brown')
        
        # Draw objects
        if scene_data and "objects" in scene_data:
            for obj_id, obj_data in scene_data["objects"].items():
                pos = obj_data["position"]
                ax.scatter([pos[0]], [pos[1]], [pos[2]], s=100, alpha=0.8)
                ax.text(pos[0], pos[1], pos[2] + 0.1, obj_id, fontsize=8)
        
        # Draw robot
        robot_pos = self.robot.position
        ax.scatter([robot_pos[0]], [robot_pos[1]], [robot_pos[2]], 
                  c='red', s=200, marker='^', label='Robot')
        
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 6)
        ax.set_zlim(0, 4)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('TANGRAM 3D Scene')
        ax.legend()
        
        return fig
    
    def cleanup(self):
        """Cleanup resources"""
        self.robot.cleanup()


# Convenience function for quick usage
def create_tangram_pipeline() -> TANGRAMCore:
    """Create and return TANGRAM pipeline instance"""
    return TANGRAMCore()


if __name__ == "__main__":
    # Simple test
    pipeline = create_tangram_pipeline()
    print("TANGRAM Core Module loaded successfully!")
    print(f"Object Detection: {'✅' if pipeline.detector else '❌'}")
    print(f"Local LLM: {'✅' if pipeline.llm.available else '❌'}")
    print(f"Robot Simulation: {'✅' if pipeline.robot.available else '❌'}")