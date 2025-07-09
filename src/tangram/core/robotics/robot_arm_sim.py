#!/usr/bin/env python3
"""
Robot Arm Simulation for TANGRAM

PyBullet-based robot arm simulation with realistic physics.
Supports object manipulation, task execution, and LLM integration.

Author: TANGRAM Team
License: MIT
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class RobotArmSimulation:
    """PyBullet-based robot arm simulation for TANGRAM"""
    
    def __init__(self, gui: bool = True, debug: bool = False):
        """
        Initialize robot arm simulation
        
        Args:
            gui: Whether to show PyBullet GUI
            debug: Enable debug visualization
        """
        self.gui = gui
        self.debug = debug
        self.connected = False
        
        # Simulation parameters
        self.time_step = 1.0/240.0
        self.gravity = -9.81
        self.max_simulation_time = 300  # 5 minutes
        
        # Robot parameters
        self.robot_id = None
        self.end_effector_link = 7  # End effector link index
        self.gripper_links = [8, 9]  # Gripper finger links
        
        # Object tracking
        self.objects = {}  # Dictionary to track spawned objects
        self.object_counter = 0
        
        # Task execution
        self.current_task = None
        self.task_queue = []
        
        # Connect to PyBullet
        self.connect()
        
    def connect(self):
        """Connect to PyBullet physics engine"""
        if self.connected:
            return
            
        if self.gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
            
        if self.client < 0:
            raise RuntimeError("Failed to connect to PyBullet")
            
        self.connected = True
        logger.info("Connected to PyBullet")
        
        # Configure simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.gravity)
        p.setTimeStep(self.time_step)
        
        # Set up debugging
        if self.debug:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        
    def disconnect(self):
        """Disconnect from PyBullet"""
        if self.connected:
            p.disconnect(self.client)
            self.connected = False
            logger.info("Disconnected from PyBullet")
    
    def reset_simulation(self):
        """Reset the simulation environment"""
        if not self.connected:
            self.connect()
            
        p.resetSimulation()
        p.setGravity(0, 0, self.gravity)
        p.setTimeStep(self.time_step)
        
        # Clear object tracking
        self.objects.clear()
        self.object_counter = 0
        
        # Load environment
        self.setup_environment()
        
        # Load robot
        self.load_robot()
        
        logger.info("Simulation reset complete")
    
    def setup_environment(self):
        """Set up the simulation environment"""
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load table
        table_pos = [0.5, 0, 0]
        table_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Create table using a box
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.6, 0.4])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.6, 0.4], 
                                         rgbaColor=[0.8, 0.6, 0.4, 1])
        
        self.table_id = p.createMultiBody(baseMass=0,  # Static object
                                        baseCollisionShapeIndex=table_collision,
                                        baseVisualShapeIndex=table_visual,
                                        basePosition=table_pos,
                                        baseOrientation=table_orientation)
        
        # Add some lighting
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            
        logger.info("Environment setup complete")
    
    def load_robot(self):
        """Load robot arm into simulation"""
        # Try to load Franka Panda robot (included in pybullet_data)
        try:
            robot_urdf = "franka_panda/panda.urdf"
            robot_pos = [0, 0, 0]
            robot_orientation = p.getQuaternionFromEuler([0, 0, 0])
            
            self.robot_id = p.loadURDF(robot_urdf, robot_pos, robot_orientation, 
                                     useFixedBase=True)
            
            # Get robot joint information
            self.num_joints = p.getNumJoints(self.robot_id)
            self.joint_info = []
            
            for i in range(self.num_joints):
                info = p.getJointInfo(self.robot_id, i)
                self.joint_info.append({
                    'id': i,
                    'name': info[1].decode('utf-8'),
                    'type': info[2],
                    'lower_limit': info[8],
                    'upper_limit': info[9],
                    'max_force': info[10],
                    'max_velocity': info[11]
                })
            
            # Set initial joint positions
            self.reset_robot_pose()
            
            logger.info(f"Robot loaded successfully: {self.num_joints} joints")
            
        except Exception as e:
            logger.error(f"Failed to load robot: {e}")
            # Fallback: create a simple robot
            self.create_simple_robot()
    
    def create_simple_robot(self):
        """Create a simple robot if URDF loading fails"""
        logger.info("Creating simple robot fallback")
        
        # Create a simple 3-DOF robot arm
        base_pos = [0, 0, 0.1]
        
        # Base link
        base_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=0.2)
        base_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.1, height=0.2,
                                        rgbaColor=[0.5, 0.5, 0.5, 1])
        
        # Link 1
        link1_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.3])
        link1_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.3],
                                         rgbaColor=[0.8, 0.2, 0.2, 1])
        
        # Link 2
        link2_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.25])
        link2_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.25],
                                         rgbaColor=[0.2, 0.8, 0.2, 1])
        
        # End effector
        ee_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.1])
        ee_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.1],
                                      rgbaColor=[0.2, 0.2, 0.8, 1])
        
        # Create multi-body robot
        link_masses = [1, 1, 1, 0.5]
        link_collision_shapes = [base_collision, link1_collision, link2_collision, ee_collision]
        link_visual_shapes = [base_visual, link1_visual, link2_visual, ee_visual]
        link_positions = [[0, 0, 0], [0, 0, 0.3], [0, 0, 0.25], [0, 0, 0.1]]
        link_orientations = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
        link_inertial_frame_positions = [[0, 0, 0], [0, 0, 0.15], [0, 0, 0.125], [0, 0, 0.05]]
        link_inertial_frame_orientations = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
        link_parent_indices = [0, 0, 1, 2]
        link_joint_types = [p.JOINT_FIXED, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
        link_joint_axis = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]]
        
        self.robot_id = p.createMultiBody(
            baseMass=2,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=base_pos,
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shapes,
            linkVisualShapeIndices=link_visual_shapes,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_frame_positions,
            linkInertialFrameOrientations=link_inertial_frame_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axis
        )
        
        self.num_joints = 3  # 3 revolute joints
        self.end_effector_link = 3
        
        # Simple joint info
        self.joint_info = [
            {'id': 0, 'name': 'base_joint', 'type': p.JOINT_REVOLUTE, 'lower_limit': -np.pi, 'upper_limit': np.pi},
            {'id': 1, 'name': 'shoulder_joint', 'type': p.JOINT_REVOLUTE, 'lower_limit': -np.pi/2, 'upper_limit': np.pi/2},
            {'id': 2, 'name': 'elbow_joint', 'type': p.JOINT_REVOLUTE, 'lower_limit': -np.pi/2, 'upper_limit': np.pi/2}
        ]
        
        logger.info("Simple robot created successfully")
    
    def reset_robot_pose(self):
        """Reset robot to initial pose"""
        if self.robot_id is None:
            return
            
        # Set joints to neutral position
        neutral_positions = [0, 0, 0, -np.pi/2, 0, np.pi/2, 0]  # Typical neutral for 7-DOF
        
        for i in range(min(len(neutral_positions), self.num_joints)):
            p.resetJointState(self.robot_id, i, neutral_positions[i])
        
        logger.info("Robot pose reset to neutral")
    
    def spawn_object(self, object_type: str, position: List[float], 
                    orientation: List[float] = None, 
                    scale: float = 1.0,
                    color: List[float] = None) -> int:
        """
        Spawn an object in the simulation
        
        Args:
            object_type: Type of object (cube, sphere, cylinder, etc.)
            position: 3D position [x, y, z]
            orientation: Orientation as euler angles [rx, ry, rz]
            scale: Scale factor
            color: RGBA color [r, g, b, a]
            
        Returns:
            Object ID
        """
        if orientation is None:
            orientation = [0, 0, 0]
        if color is None:
            color = [0.8, 0.4, 0.2, 1.0]
        
        quat_orientation = p.getQuaternionFromEuler(orientation)
        
        # Create different object types
        if object_type == "cube":
            collision_shape = p.createCollisionShape(p.GEOM_BOX, 
                                                   halfExtents=[0.05*scale, 0.05*scale, 0.05*scale])
            visual_shape = p.createVisualShape(p.GEOM_BOX, 
                                             halfExtents=[0.05*scale, 0.05*scale, 0.05*scale],
                                             rgbaColor=color)
        elif object_type == "sphere":
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05*scale)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.05*scale,
                                             rgbaColor=color)
        elif object_type == "cylinder":
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, 
                                                   radius=0.05*scale, height=0.1*scale)
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, 
                                             radius=0.05*scale, length=0.1*scale,
                                             rgbaColor=color)
        else:
            # Default to cube
            collision_shape = p.createCollisionShape(p.GEOM_BOX, 
                                                   halfExtents=[0.05*scale, 0.05*scale, 0.05*scale])
            visual_shape = p.createVisualShape(p.GEOM_BOX, 
                                             halfExtents=[0.05*scale, 0.05*scale, 0.05*scale],
                                             rgbaColor=color)
        
        # Create multi-body object
        object_id = p.createMultiBody(baseMass=0.1,
                                    baseCollisionShapeIndex=collision_shape,
                                    baseVisualShapeIndex=visual_shape,
                                    basePosition=position,
                                    baseOrientation=quat_orientation)
        
        # Track object
        self.object_counter += 1
        self.objects[self.object_counter] = {
            'id': object_id,
            'type': object_type,
            'position': position,
            'orientation': orientation,
            'scale': scale,
            'color': color,
            'name': f"{object_type}_{self.object_counter}"
        }
        
        logger.info(f"Spawned {object_type} at {position}")
        return object_id
    
    def move_to_position(self, target_position: List[float], 
                        target_orientation: List[float] = None,
                        speed: float = 0.1) -> bool:
        """
        Move robot end effector to target position
        
        Args:
            target_position: Target 3D position [x, y, z]
            target_orientation: Target orientation as euler angles
            speed: Movement speed
            
        Returns:
            True if movement successful
        """
        if self.robot_id is None:
            logger.error("Robot not loaded")
            return False
        
        if target_orientation is None:
            target_orientation = [0, np.pi, 0]  # Default downward orientation
        
        target_quat = p.getQuaternionFromEuler(target_orientation)
        
        # Inverse kinematics to find joint positions
        joint_positions = p.calculateInverseKinematics(
            self.robot_id, 
            self.end_effector_link, 
            target_position, 
            target_quat
        )
        
        # Move joints to target positions
        for i, pos in enumerate(joint_positions[:self.num_joints]):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, 
                                  targetPosition=pos, maxVelocity=speed)
        
        # Wait for movement to complete
        for _ in range(100):  # Max 100 simulation steps
            p.stepSimulation()
            time.sleep(self.time_step)
            
            # Check if close to target
            current_pos = p.getLinkState(self.robot_id, self.end_effector_link)[0]
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_position))
            
            if distance < 0.01:  # 1cm tolerance
                logger.info(f"Reached target position: {target_position}")
                return True
        
        logger.warning(f"Failed to reach target position: {target_position}")
        return False
    
    def pick_object(self, object_id: int) -> bool:
        """
        Pick up an object
        
        Args:
            object_id: ID of object to pick up
            
        Returns:
            True if successful
        """
        if self.robot_id is None:
            logger.error("Robot not loaded")
            return False
        
        # Get object position
        object_pos, object_orn = p.getBasePositionAndOrientation(object_id)
        
        # Move above object
        pre_grasp_pos = [object_pos[0], object_pos[1], object_pos[2] + 0.1]
        if not self.move_to_position(pre_grasp_pos):
            return False
        
        # Move down to object
        grasp_pos = [object_pos[0], object_pos[1], object_pos[2] + 0.05]
        if not self.move_to_position(grasp_pos):
            return False
        
        # Close gripper (simulate)
        if hasattr(self, 'gripper_links'):
            for link in self.gripper_links:
                p.setJointMotorControl2(self.robot_id, link, p.POSITION_CONTROL, 
                                      targetPosition=0.02)  # Close gripper
        
        # Create constraint to attach object to gripper
        constraint_id = p.createConstraint(
            self.robot_id, self.end_effector_link,
            object_id, -1,
            p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0.05]
        )
        
        # Lift object
        lift_pos = [object_pos[0], object_pos[1], object_pos[2] + 0.2]
        success = self.move_to_position(lift_pos)
        
        if success:
            logger.info(f"Successfully picked up object {object_id}")
            return constraint_id
        else:
            p.removeConstraint(constraint_id)
            return False
    
    def place_object(self, constraint_id: int, target_position: List[float]) -> bool:
        """
        Place an object at target position
        
        Args:
            constraint_id: Constraint ID from pick_object
            target_position: Target position to place object
            
        Returns:
            True if successful
        """
        if self.robot_id is None:
            logger.error("Robot not loaded")
            return False
        
        # Move to above target position
        pre_place_pos = [target_position[0], target_position[1], target_position[2] + 0.1]
        if not self.move_to_position(pre_place_pos):
            return False
        
        # Move down to place
        place_pos = [target_position[0], target_position[1], target_position[2] + 0.05]
        if not self.move_to_position(place_pos):
            return False
        
        # Release object
        p.removeConstraint(constraint_id)
        
        # Open gripper
        if hasattr(self, 'gripper_links'):
            for link in self.gripper_links:
                p.setJointMotorControl2(self.robot_id, link, p.POSITION_CONTROL, 
                                      targetPosition=0.08)  # Open gripper
        
        # Move away
        retreat_pos = [target_position[0], target_position[1], target_position[2] + 0.2]
        self.move_to_position(retreat_pos)
        
        logger.info(f"Placed object at {target_position}")
        return True
    
    def execute_task(self, task: Dict[str, Any]) -> bool:
        """
        Execute a task command
        
        Args:
            task: Task dictionary with action and parameters
            
        Returns:
            True if successful
        """
        action = task.get("action", "")
        params = task.get("parameters", {})
        
        logger.info(f"Executing task: {action}")
        
        if action == "pick":
            object_id = params.get("object_id")
            if object_id:
                return self.pick_object(object_id)
                
        elif action == "place":
            constraint_id = params.get("constraint_id")
            position = params.get("position", [0.5, 0, 0.5])
            if constraint_id:
                return self.place_object(constraint_id, position)
                
        elif action == "move":
            position = params.get("position", [0.5, 0, 0.5])
            return self.move_to_position(position)
            
        elif action == "spawn":
            object_type = params.get("type", "cube")
            position = params.get("position", [0.5, 0, 0.5])
            return self.spawn_object(object_type, position)
            
        else:
            logger.error(f"Unknown action: {action}")
            return False
    
    def get_scene_state(self) -> Dict[str, Any]:
        """Get current state of the simulation scene"""
        scene_state = {
            "objects": [],
            "robot_pose": None,
            "timestamp": time.time()
        }
        
        # Get object states
        for obj_id, obj_info in self.objects.items():
            pos, orn = p.getBasePositionAndOrientation(obj_info["id"])
            scene_state["objects"].append({
                "id": obj_id,
                "name": obj_info["name"],
                "type": obj_info["type"],
                "position": list(pos),
                "orientation": list(orn),
                "scale": obj_info["scale"]
            })
        
        # Get robot pose
        if self.robot_id is not None:
            ee_state = p.getLinkState(self.robot_id, self.end_effector_link)
            scene_state["robot_pose"] = {
                "position": list(ee_state[0]),
                "orientation": list(ee_state[1])
            }
        
        return scene_state
    
    def create_demo_scene(self):
        """Create a demo scene with some objects"""
        logger.info("Creating demo scene...")
        
        # Spawn some objects on the table
        self.spawn_object("cube", [0.4, 0.2, 0.5], color=[1, 0, 0, 1])
        self.spawn_object("sphere", [0.6, 0.2, 0.5], color=[0, 1, 0, 1])
        self.spawn_object("cylinder", [0.5, 0.4, 0.5], color=[0, 0, 1, 1])
        
        logger.info("Demo scene created")
    
    def run_simulation(self, duration: float = 5.0):
        """Run simulation for specified duration"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            p.stepSimulation()
            time.sleep(self.time_step)
        
        logger.info(f"Simulation ran for {duration} seconds")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.disconnect()

def main():
    """Test robot arm simulation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test robot arm simulation")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration")
    
    args = parser.parse_args()
    
    # Create simulation
    sim = RobotArmSimulation(gui=not args.no_gui, debug=args.debug)
    
    try:
        # Reset and setup
        sim.reset_simulation()
        
        # Create demo scene
        sim.create_demo_scene()
        
        # Run some demo tasks
        print("Running demo tasks...")
        
        # Move to different positions
        sim.move_to_position([0.5, 0.3, 0.6])
        sim.move_to_position([0.4, 0.2, 0.6])
        
        # Try to pick up an object
        cube_id = list(sim.objects.values())[0]["id"]
        constraint = sim.pick_object(cube_id)
        
        if constraint:
            # Place it somewhere else
            sim.place_object(constraint, [0.3, 0.4, 0.5])
        
        # Run simulation
        sim.run_simulation(args.duration)
        
        print("Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
        
    finally:
        sim.disconnect()

if __name__ == "__main__":
    main()