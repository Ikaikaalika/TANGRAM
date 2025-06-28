#!/usr/bin/env python3
"""
PyBullet Simulation Viewer for TANGRAM GUI

This module provides PyBullet visualization integration for the TANGRAM GUI,
allowing real-time monitoring of robot simulation within the main interface.

Author: TANGRAM Team
License: MIT
"""

import threading
import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import tkinter as tk
from tkinter import ttk

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

class SimulationViewer:
    """
    PyBullet simulation viewer for TANGRAM GUI integration.
    
    Provides embedded PyBullet visualization with real-time monitoring
    and control capabilities.
    """
    
    def __init__(self, parent_frame: tk.Frame, callback_func: Optional[Callable] = None):
        """
        Initialize simulation viewer.
        
        Args:
            parent_frame: Parent tkinter frame
            callback_func: Optional callback for simulation events
        """
        self.parent_frame = parent_frame
        self.callback_func = callback_func
        
        # PyBullet state
        self.physics_client = None
        self.is_connected = False
        self.robot_id = None
        self.object_ids = {}
        
        # Simulation control
        self.is_running = False
        self.simulation_thread = None
        self.stop_event = threading.Event()
        
        # GUI components
        self.setup_viewer_gui()
        
        # Initialize PyBullet if available
        if PYBULLET_AVAILABLE:
            self.initialize_simulation()
        else:
            self.show_unavailable_message()
    
    def setup_viewer_gui(self):
        """Set up the simulation viewer GUI components."""
        # Main container
        self.viewer_frame = ttk.LabelFrame(self.parent_frame, text="PyBullet Simulation", padding=10)
        self.viewer_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.Frame(self.viewer_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Simulation controls
        ttk.Button(control_frame, text="▶️ Start Simulation", 
                  command=self.start_simulation).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="⏸️ Pause", 
                  command=self.pause_simulation).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="🔄 Reset", 
                  command=self.reset_simulation).pack(side=tk.LEFT, padx=2)
        
        # View controls
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Button(control_frame, text="📷 Top View", 
                  command=lambda: self.set_camera_view("top")).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="👁️ Side View", 
                  command=lambda: self.set_camera_view("side")).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="🎯 Follow Robot", 
                  command=self.follow_robot).pack(side=tk.LEFT, padx=2)
        
        # Simulation status display
        self.status_frame = ttk.LabelFrame(self.viewer_frame, text="Simulation Status", padding=5)
        self.status_frame.pack(fill=tk.X, pady=5)
        
        # Status variables
        self.sim_status_var = tk.StringVar(value="Disconnected")
        self.robot_status_var = tk.StringVar(value="No Robot")
        self.objects_count_var = tk.StringVar(value="Objects: 0")
        self.fps_var = tk.StringVar(value="FPS: 0")
        
        ttk.Label(self.status_frame, textvariable=self.sim_status_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.status_frame, textvariable=self.robot_status_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.status_frame, textvariable=self.objects_count_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.status_frame, textvariable=self.fps_var).pack(side=tk.LEFT, padx=5)
        
        # Simulation info text area
        self.info_text = tk.Text(self.viewer_frame, height=8, wrap=tk.WORD)
        info_scroll = ttk.Scrollbar(self.viewer_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add initial message
        self.log_simulation_message("Simulation viewer initialized")
    
    def show_unavailable_message(self):
        """Show message when PyBullet is unavailable."""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, 
            "PyBullet not available locally.\n\n"
            "To enable local simulation:\n"
            "1. Install PyBullet: pip install pybullet\n"
            "2. Or use Thunder Compute for remote simulation\n\n"
            "Current status: Mock simulation mode"
        )
        self.sim_status_var.set("PyBullet Unavailable")
    
    def initialize_simulation(self):
        """Initialize PyBullet simulation."""
        if not PYBULLET_AVAILABLE:
            return False
        
        try:
            # Connect to PyBullet in GUI mode
            self.physics_client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # Set up physics
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1./240.)
            
            # Load ground plane
            p.loadURDF("plane.urdf")
            
            self.is_connected = True
            self.sim_status_var.set("Connected")
            self.log_simulation_message("PyBullet simulation initialized")
            
            return True
            
        except Exception as e:
            self.log_simulation_message(f"Failed to initialize PyBullet: {e}")
            self.sim_status_var.set("Connection Failed")
            return False
    
    def load_robot(self, robot_type: str = "franka_panda") -> bool:
        """
        Load robot into simulation.
        
        Args:
            robot_type: Type of robot to load
            
        Returns:
            True if robot loaded successfully
        """
        if not self.is_connected:
            return False
        
        try:
            robot_urdf_map = {
                "franka_panda": "franka_panda/panda.urdf",
                "kuka_iiwa": "kuka_iiwa/model.urdf", 
                "ur5": "r2d2.urdf"  # Fallback
            }
            
            urdf_path = robot_urdf_map.get(robot_type, "r2d2.urdf")
            
            self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
            
            if self.robot_id is not None:
                self.robot_status_var.set(f"Robot: {robot_type}")
                self.log_simulation_message(f"Loaded robot: {robot_type}")
                return True
            else:
                self.robot_status_var.set("Robot Load Failed")
                return False
                
        except Exception as e:
            self.log_simulation_message(f"Failed to load robot: {e}")
            self.robot_status_var.set("Robot Error")
            return False
    
    def add_scene_objects(self, object_positions: Dict[str, Any]) -> Dict[str, int]:
        """
        Add scene objects to simulation.
        
        Args:
            object_positions: Dictionary of object positions and properties
            
        Returns:
            Mapping of object IDs to PyBullet IDs
        """
        if not self.is_connected:
            return {}
        
        object_mapping = {}
        
        try:
            for obj_id, obj_data in object_positions.items():
                position = obj_data.get("position", [0, 0, 1])
                class_name = obj_data.get("class_name", "cube")
                
                # Select URDF based on class
                urdf_path = self._get_urdf_for_class(class_name)
                
                bullet_id = p.loadURDF(urdf_path, position)
                object_mapping[obj_id] = bullet_id
                self.object_ids[obj_id] = bullet_id
                
                self.log_simulation_message(f"Added {class_name} at {position}")
            
            self.objects_count_var.set(f"Objects: {len(object_mapping)}")
            return object_mapping
            
        except Exception as e:
            self.log_simulation_message(f"Failed to add objects: {e}")
            return {}
    
    def _get_urdf_for_class(self, class_name: str) -> str:
        """Get appropriate URDF file for object class."""
        urdf_map = {
            "cup": "objects/mug.urdf",
            "mug": "objects/mug.urdf",
            "bottle": "sphere2.urdf",
            "book": "cube.urdf",
            "box": "cube.urdf",
            "sphere": "sphere2.urdf",
            "ball": "sphere2.urdf"
        }
        
        return urdf_map.get(class_name.lower(), "cube.urdf")
    
    def execute_robot_task(self, task: Dict[str, Any]) -> bool:
        """
        Execute a robot task in simulation.
        
        Args:
            task: Task description dictionary
            
        Returns:
            True if task executed successfully
        """
        if not self.is_connected or self.robot_id is None:
            return False
        
        try:
            task_type = task.get("type", "unknown")
            self.log_simulation_message(f"Executing task: {task_type}")
            
            # Simple task simulation
            if task_type in ["grasp", "pick"]:
                return self._simulate_grasp()
            elif task_type in ["place", "put"]:
                return self._simulate_place()
            elif task_type in ["push", "slide"]:
                return self._simulate_push()
            else:
                return self._simulate_generic_motion()
                
        except Exception as e:
            self.log_simulation_message(f"Task execution failed: {e}")
            return False
    
    def _simulate_grasp(self) -> bool:
        """Simulate grasping motion."""
        num_joints = p.getNumJoints(self.robot_id)
        
        for step in range(50):
            for joint_idx in range(min(6, num_joints)):
                target_pos = 0.1 * np.sin(step * 0.1) * (joint_idx + 1) / 6
                p.setJointMotorControl2(
                    self.robot_id, joint_idx, 
                    p.POSITION_CONTROL, 
                    targetPosition=target_pos
                )
            
            p.stepSimulation()
            time.sleep(0.01)
        
        return True
    
    def _simulate_place(self) -> bool:
        """Simulate placement motion."""
        num_joints = p.getNumJoints(self.robot_id)
        
        for step in range(30):
            for joint_idx in range(min(6, num_joints)):
                target_pos = 0.05 * np.cos(step * 0.15) * (joint_idx + 1) / 6
                p.setJointMotorControl2(
                    self.robot_id, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos
                )
            
            p.stepSimulation()
            time.sleep(0.01)
        
        return True
    
    def _simulate_push(self) -> bool:
        """Simulate pushing motion."""
        num_joints = p.getNumJoints(self.robot_id)
        
        for step in range(40):
            for joint_idx in range(min(3, num_joints)):
                target_pos = 0.2 * np.sin(step * 0.08) * (joint_idx + 1) / 3
                p.setJointMotorControl2(
                    self.robot_id, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos
                )
            
            p.stepSimulation()
            time.sleep(0.005)
        
        return True
    
    def _simulate_generic_motion(self) -> bool:
        """Simulate generic robot motion."""
        num_joints = p.getNumJoints(self.robot_id)
        
        for step in range(60):
            for joint_idx in range(min(4, num_joints)):
                target_pos = 0.15 * np.sin(step * 0.05 + joint_idx)
                p.setJointMotorControl2(
                    self.robot_id, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos
                )
            
            p.stepSimulation()
            time.sleep(0.008)
        
        return True
    
    def start_simulation(self):
        """Start simulation loop."""
        if not self.is_connected:
            self.log_simulation_message("Cannot start - not connected to PyBullet")
            return
        
        if self.is_running:
            self.log_simulation_message("Simulation already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        self.log_simulation_message("Simulation started")
    
    def _simulation_loop(self):
        """Main simulation loop."""
        fps_counter = 0
        last_fps_time = time.time()
        
        while self.is_running and not self.stop_event.is_set():
            try:
                p.stepSimulation()
                fps_counter += 1
                
                # Update FPS display every second
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.fps_var.set(f"FPS: {fps_counter}")
                    fps_counter = 0
                    last_fps_time = current_time
                
                time.sleep(1./240.)  # 240 Hz
                
            except Exception as e:
                self.log_simulation_message(f"Simulation loop error: {e}")
                break
        
        self.is_running = False
        self.log_simulation_message("Simulation stopped")
    
    def pause_simulation(self):
        """Pause simulation."""
        self.is_running = False
        self.log_simulation_message("Simulation paused")
    
    def reset_simulation(self):
        """Reset simulation to initial state."""
        if not self.is_connected:
            return
        
        try:
            # Stop current simulation
            self.is_running = False
            
            # Reset PyBullet
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            
            # Reload ground plane
            p.loadURDF("plane.urdf")
            
            # Clear object tracking
            self.robot_id = None
            self.object_ids.clear()
            
            # Update status
            self.robot_status_var.set("No Robot")
            self.objects_count_var.set("Objects: 0")
            self.fps_var.set("FPS: 0")
            
            self.log_simulation_message("Simulation reset")
            
        except Exception as e:
            self.log_simulation_message(f"Reset failed: {e}")
    
    def set_camera_view(self, view_type: str):
        """Set camera view in PyBullet."""
        if not self.is_connected:
            return
        
        try:
            if view_type == "top":
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.0,
                    cameraYaw=0,
                    cameraPitch=-89,
                    cameraTargetPosition=[0, 0, 0]
                )
            elif view_type == "side":
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.0,
                    cameraYaw=90,
                    cameraPitch=-30,
                    cameraTargetPosition=[0, 0, 0.5]
                )
            
            self.log_simulation_message(f"Camera set to {view_type} view")
            
        except Exception as e:
            self.log_simulation_message(f"Camera view change failed: {e}")
    
    def follow_robot(self):
        """Set camera to follow robot."""
        if not self.is_connected or self.robot_id is None:
            return
        
        try:
            # Get robot position
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-20,
                cameraTargetPosition=pos
            )
            
            self.log_simulation_message("Camera following robot")
            
        except Exception as e:
            self.log_simulation_message(f"Follow robot failed: {e}")
    
    def log_simulation_message(self, message: str):
        """Add message to simulation log."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        if self.info_text:
            self.info_text.insert(tk.END, log_entry)
            self.info_text.see(tk.END)
        
        # Call callback if provided
        if self.callback_func:
            self.callback_func(f"Simulation: {message}")
    
    def cleanup(self):
        """Clean up simulation resources."""
        self.is_running = False
        self.stop_event.set()
        
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)
        
        if self.is_connected:
            try:
                p.disconnect()
                self.is_connected = False
                self.log_simulation_message("PyBullet disconnected")
            except:
                pass