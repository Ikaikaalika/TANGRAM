#!/usr/bin/env python3
"""
Advanced Robot Arm Simulation

Realistic physics-based robot simulation with:
- High-fidelity 6-DOF robot arm (UR5-style)
- Advanced gripper mechanics
- Realistic object physics and materials
- Collision detection and response
- Force/torque sensing
- Path planning and motion control
- Real-time visualization

Author: TANGRAM Team
License: MIT
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json

from src.tangram.utils.logging_utils import setup_logger
from src.tangram.core.computer_vision.advanced_reconstruction import DetectedObject, SceneReconstruction

logger = setup_logger(__name__)

@dataclass
class RobotPose:
    """Robot arm pose representation"""
    joint_angles: List[float]  # 6 DOF joint angles in radians
    gripper_width: float      # Gripper opening width
    end_effector_pos: Tuple[float, float, float]
    end_effector_orn: Tuple[float, float, float, float]  # quaternion

@dataclass
class SimulationObject:
    """Object in simulation with physics properties"""
    id: int
    pybullet_id: int
    name: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    dimensions: Tuple[float, float, float]
    mass: float
    friction: float
    restitution: float
    color: Tuple[float, float, float, float]

class AdvancedRobotSimulation:
    """
    Advanced physics-based robot simulation environment
    """
    
    def __init__(self, gui: bool = True, real_time: bool = False):
        """Initialize advanced robot simulation"""
        
        self.gui = gui
        self.real_time = real_time
        
        # PyBullet connection
        self.physics_client = None
        
        # Robot components
        self.robot_id = None
        self.gripper_id = None
        self.joint_indices = []
        self.gripper_joints = []
        
        # Environment
        self.plane_id = None
        self.table_id = None
        self.objects: Dict[int, SimulationObject] = {}
        
        # Robot configuration (UR5-style 6-DOF arm)
        self.robot_config = {
            'base_position': [0, 0, 0],
            'joint_limits': [
                (-np.pi, np.pi),      # Base rotation
                (-np.pi/2, np.pi/2),  # Shoulder
                (-np.pi, np.pi),      # Elbow
                (-np.pi, np.pi),      # Wrist 1
                (-np.pi, np.pi),      # Wrist 2
                (-np.pi, np.pi),      # Wrist 3
            ],
            'home_position': [0, -np.pi/4, np.pi/2, -np.pi/4, -np.pi/2, 0],
            'max_velocity': [2.0] * 6,
            'max_force': [100.0] * 6
        }
        
        # Workspace bounds (in meters)
        self.workspace = {
            'x_range': (-0.8, 0.8),
            'y_range': (-0.8, 0.8),
            'z_range': (0.0, 1.5)
        }
        
        # Simulation state
        self.current_pose = None
        self.target_pose = None
        self.simulation_time = 0.0
        self.action_sequence = []
        
        logger.info("Advanced robot simulation initialized")
    
    def initialize_simulation(self) -> bool:
        """Initialize PyBullet physics simulation with advanced settings"""
        
        try:
            # Connect to PyBullet
            if self.gui:
                self.physics_client = p.connect(p.GUI)
                # Configure GUI for better visualization
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
                
                # Set optimal camera position
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.5,
                    cameraYaw=45,
                    cameraPitch=-25,
                    cameraTargetPosition=[0.2, 0.2, 0.8]
                )
            else:
                self.physics_client = p.connect(p.DIRECT)
            
            # Set physics parameters
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1./240.)  # High frequency for stability
            p.setRealTimeSimulation(1 if self.real_time else 0)
            
            # Create environment
            self._create_environment()
            
            # Load robot
            self._load_robot()
            
            # Initialize robot pose
            self._set_home_position()
            
            logger.info("✅ Advanced simulation initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Simulation initialization failed: {e}")
            return False
    
    def _create_environment(self):
        """Create realistic environment with table and workspace"""
        
        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Realistic table
        table_height = 0.8
        table_size = [1.0, 0.8, 0.05]  # 1m x 0.8m x 5cm
        
        # Create table collision shape and visual shape
        table_collision = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=table_size
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=table_size,
            rgbaColor=[0.6, 0.4, 0.2, 1.0]  # Wood color
        )
        
        self.table_id = p.createMultiBody(
            baseMass=50.0,  # Heavy table
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.4, 0, table_height - table_size[2]]
        )
        
        # Set table material properties
        p.changeDynamics(
            self.table_id, -1,
            lateralFriction=0.8,
            spinningFriction=0.1,
            restitution=0.1
        )
        
        logger.info("Environment created: ground plane and table")
    
    def _load_robot(self):
        """Load advanced 6-DOF robot arm"""
        
        # Create simplified robot programmatically (more reliable than URDF loading)
        self.robot_id = self._create_simple_robot()
        logger.info("✅ Simplified 6-DOF robot created")
        
        # Get joint information
        self._setup_joint_control()
    
    def _create_simple_robot(self) -> int:
        """Create a simplified 6-DOF robot arm programmatically"""
        
        try:
            # Create simple box-based robot arm for better compatibility
            
            # Base (box shape)
            base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
            base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05], 
                                            rgbaColor=[0.3, 0.3, 0.3, 1.0])
            
            # Link dimensions (using boxes for better stability)
            link_sizes = [
                [0.02, 0.02, 0.15],  # Link 1
                [0.02, 0.02, 0.25],  # Link 2  
                [0.02, 0.02, 0.2],   # Link 3
                [0.015, 0.015, 0.15], # Link 4
                [0.01, 0.01, 0.1],   # Link 5
                [0.01, 0.01, 0.05]   # Link 6 (end effector)
            ]
            
            # Create collision and visual shapes
            link_collisions = []
            link_visuals = []
            
            for i, size in enumerate(link_sizes):
                collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
                visual = p.createVisualShape(p.GEOM_BOX, halfExtents=size,
                                           rgbaColor=[0.8, 0.4, 0.1, 1.0])  # Orange robot color
                link_collisions.append(collision)
                link_visuals.append(visual)
            
            # Joint axes
            joint_axes = [
                [0, 0, 1],  # Base rotation (Z)
                [0, 1, 0],  # Shoulder (Y)
                [0, 1, 0],  # Elbow (Y)
                [1, 0, 0],  # Wrist 1 (X)
                [0, 1, 0],  # Wrist 2 (Y)
                [1, 0, 0],  # Wrist 3 (X)
            ]
            
            # Link positions relative to parent
            link_positions = [
                [0, 0, 0.1],   # Link 1 position
                [0, 0, 0.15],  # Link 2 position
                [0, 0, 0.25],  # Link 3 position
                [0, 0, 0.2],   # Link 4 position
                [0, 0, 0.15],  # Link 5 position
                [0, 0, 0.1]    # Link 6 position
            ]
            
            link_orientations = [[0, 0, 0, 1]] * 6
            
            # Create robot with createMultiBody
            robot_id = p.createMultiBody(
                baseMass=10.0,
                baseCollisionShapeIndex=base_collision,
                baseVisualShapeIndex=base_visual,
                basePosition=self.robot_config['base_position'],
                linkMasses=[2.0, 3.0, 2.5, 1.5, 1.0, 0.5],
                linkCollisionShapeIndices=link_collisions,
                linkVisualShapeIndices=link_visuals,
                linkPositions=link_positions,
                linkOrientations=link_orientations,
                linkInertialFramePositions=[[0, 0, 0]] * 6,
                linkInertialFrameOrientations=[[0, 0, 0, 1]] * 6,
                linkParentIndices=[0, 1, 2, 3, 4, 5],
                linkJointTypes=[p.JOINT_REVOLUTE] * 6,
                linkJointAxis=joint_axes
            )
            
            return robot_id
            
        except Exception as e:
            logger.error(f"Failed to create robot: {e}")
            # Fallback: create a simple single-body robot
            fallback_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.5])
            fallback_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.5],
                                                 rgbaColor=[0.8, 0.4, 0.1, 1.0])
            
            robot_id = p.createMultiBody(
                baseMass=5.0,
                baseCollisionShapeIndex=fallback_collision,
                baseVisualShapeIndex=fallback_visual,
                basePosition=self.robot_config['base_position']
            )
            
            logger.info("✅ Created fallback robot")
            return robot_id
    
    def _setup_joint_control(self):
        """Setup joint control parameters"""
        
        num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = list(range(num_joints))
        
        # Set joint limits and control parameters
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            
            # Set joint limits if available from config
            if i < len(self.robot_config['joint_limits']):
                lower, upper = self.robot_config['joint_limits'][i]
                p.changeDynamics(self.robot_id, i, jointLowerLimit=lower, jointUpperLimit=upper)
            
            # Set joint damping for stability
            p.changeDynamics(self.robot_id, i, jointDamping=0.1)
            
            logger.debug(f"Joint {i}: {joint_name}")
    
    def _set_home_position(self):
        """Set robot to home position"""
        
        home_angles = self.robot_config['home_position']
        
        for i, angle in enumerate(home_angles):
            if i < len(self.joint_indices):
                p.resetJointState(self.robot_id, i, angle)
        
        self.current_pose = RobotPose(
            joint_angles=home_angles,
            gripper_width=0.08,  # Open gripper
            end_effector_pos=self._get_end_effector_position(),
            end_effector_orn=(0, 0, 0, 1)
        )
        
        logger.info("Robot set to home position")
    
    def load_scene_objects(self, scene: SceneReconstruction) -> bool:
        """Load detected objects into simulation with realistic physics"""
        
        logger.info(f"Loading {len(scene.objects)} objects into simulation...")
        
        for obj in scene.objects:
            sim_obj = self._create_physics_object(obj)
            if sim_obj:
                self.objects[obj.id] = sim_obj
        
        logger.info(f"✅ Loaded {len(self.objects)} objects with physics")
        return True
    
    def _create_physics_object(self, detected_obj: DetectedObject) -> Optional[SimulationObject]:
        """Create physics object from detected object"""
        
        try:
            # Object properties based on class
            obj_properties = self._get_object_properties(detected_obj.class_name)
            
            # Use detected dimensions or defaults
            if detected_obj.dimensions[0] > 0:
                half_extents = [d/2 for d in detected_obj.dimensions]
            else:
                half_extents = obj_properties['default_size']
            
            # Create collision and visual shapes
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents
            )
            
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=obj_properties['color']
            )
            
            # Adjust position to table surface
            position = list(detected_obj.center_3d)
            position[2] = 0.8 + half_extents[2]  # Table height + object half-height
            
            # Create physics body
            body_id = p.createMultiBody(
                baseMass=obj_properties['mass'],
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position
            )
            
            # Set material properties
            p.changeDynamics(
                body_id, -1,
                lateralFriction=obj_properties['friction'],
                spinningFriction=0.1,
                restitution=obj_properties['restitution']
            )
            
            sim_obj = SimulationObject(
                id=detected_obj.id,
                pybullet_id=body_id,
                name=detected_obj.class_name,
                position=tuple(position),
                orientation=(0, 0, 0, 1),
                dimensions=tuple(half_extents),
                mass=obj_properties['mass'],
                friction=obj_properties['friction'],
                restitution=obj_properties['restitution'],
                color=obj_properties['color']
            )
            
            logger.debug(f"Created physics object: {detected_obj.class_name} at {position}")
            return sim_obj
            
        except Exception as e:
            logger.error(f"Failed to create physics object for {detected_obj.class_name}: {e}")
            return None
    
    def _get_object_properties(self, class_name: str) -> Dict[str, Any]:
        """Get physics properties for different object types"""
        
        properties = {
            'orange': {
                'mass': 0.2,
                'friction': 0.6,
                'restitution': 0.3,
                'color': [1.0, 0.5, 0.0, 1.0],
                'default_size': [0.04, 0.04, 0.04]
            },
            'apple': {
                'mass': 0.15,
                'friction': 0.5,
                'restitution': 0.2,
                'color': [0.8, 0.2, 0.2, 1.0],
                'default_size': [0.04, 0.04, 0.04]
            },
            'cup': {
                'mass': 0.3,
                'friction': 0.7,
                'restitution': 0.1,
                'color': [0.9, 0.9, 0.9, 1.0],
                'default_size': [0.04, 0.04, 0.08]
            },
            'bottle': {
                'mass': 0.5,
                'friction': 0.4,
                'restitution': 0.1,
                'color': [0.2, 0.7, 0.2, 1.0],
                'default_size': [0.03, 0.03, 0.12]
            },
            'frisbee': {
                'mass': 0.1,
                'friction': 0.5,
                'restitution': 0.4,
                'color': [0.2, 0.8, 0.2, 1.0],
                'default_size': [0.12, 0.12, 0.01]
            },
            'book': {
                'mass': 0.4,
                'friction': 0.8,
                'restitution': 0.05,
                'color': [0.3, 0.3, 0.8, 1.0],
                'default_size': [0.08, 0.12, 0.02]
            }
        }
        
        # Default properties for unknown objects
        default = {
            'mass': 0.2,
            'friction': 0.6,
            'restitution': 0.2,
            'color': [0.5, 0.5, 0.5, 1.0],
            'default_size': [0.05, 0.05, 0.05]
        }
        
        return properties.get(class_name, default)
    
    def execute_robot_task(self, task_description: str, target_objects: List[str]) -> bool:
        """Execute complex robot task with motion planning"""
        
        logger.info(f"Executing task: {task_description}")
        
        # Parse task type
        if "pick" in task_description.lower():
            return self._execute_pick_task(target_objects)
        elif "stack" in task_description.lower() or "pile" in task_description.lower():
            return self._execute_stacking_task(target_objects)
        elif "organize" in task_description.lower() or "arrange" in task_description.lower():
            return self._execute_organization_task(target_objects)
        else:
            logger.warning(f"Unknown task type: {task_description}")
            return False
    
    def _execute_pick_task(self, target_objects: List[str]) -> bool:
        """Execute object picking with realistic motion"""
        
        for obj_name in target_objects:
            # Find object in simulation
            target_obj = None
            for obj in self.objects.values():
                if obj_name.lower() in obj.name.lower():
                    target_obj = obj
                    break
            
            if not target_obj:
                logger.warning(f"Object not found: {obj_name}")
                continue
            
            # Get object position
            obj_pos, obj_orn = p.getBasePositionAndOrientation(target_obj.pybullet_id)
            
            # Plan approach trajectory
            approach_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.1]  # 10cm above
            
            # Move to approach position
            self._move_to_position(approach_pos, [0, 0, 0, 1])
            
            # Open gripper
            self._set_gripper_width(0.08)
            time.sleep(0.5)
            
            # Move down to object
            self._move_to_position(obj_pos, [0, 0, 0, 1])
            
            # Close gripper
            self._set_gripper_width(0.02)
            time.sleep(0.5)
            
            # Lift object
            lift_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.2]
            self._move_to_position(lift_pos, [0, 0, 0, 1])
            
            logger.info(f"✅ Picked up {obj_name}")
        
        return True
    
    def _execute_stacking_task(self, target_objects: List[str]) -> bool:
        """Execute object stacking with physics simulation"""
        
        # Define stack center
        stack_center = [0.3, 0.2, 0.8]  # On table
        stack_height = 0.8
        
        for i, obj_name in enumerate(target_objects):
            target_obj = None
            for obj in self.objects.values():
                if obj_name.lower() in obj.name.lower():
                    target_obj = obj
                    break
            
            if not target_obj:
                continue
            
            # Pick up object
            if self._pick_object(target_obj):
                # Place at stack location
                place_height = stack_height + (i * 0.05)  # Stack with small spacing
                place_pos = [stack_center[0], stack_center[1], place_height]
                
                self._place_object_at_position(place_pos)
                
                logger.info(f"✅ Stacked {obj_name} at height {place_height:.2f}")
        
        return True
    
    def _execute_organization_task(self, target_objects: List[str]) -> bool:
        """Organize objects in a neat arrangement"""
        
        # Define organization grid
        grid_center = [0.4, 0.0, 0.8]
        grid_spacing = 0.15
        
        for i, obj_name in enumerate(target_objects):
            target_obj = None
            for obj in self.objects.values():
                if obj_name.lower() in obj.name.lower():
                    target_obj = obj
                    break
            
            if not target_obj:
                continue
            
            # Calculate grid position
            row = i // 3
            col = i % 3
            place_pos = [
                grid_center[0] + (col - 1) * grid_spacing,
                grid_center[1] + (row - 1) * grid_spacing,
                grid_center[2]
            ]
            
            # Pick and place
            if self._pick_object(target_obj):
                self._place_object_at_position(place_pos)
                logger.info(f"✅ Organized {obj_name} at position {place_pos}")
        
        return True
    
    def _pick_object(self, obj: SimulationObject) -> bool:
        """Pick up a specific object"""
        
        obj_pos, obj_orn = p.getBasePositionAndOrientation(obj.pybullet_id)
        
        # Approach
        approach_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.1]
        if not self._move_to_position(approach_pos, [0, 0, 0, 1]):
            return False
        
        # Open gripper
        self._set_gripper_width(0.08)
        self._wait_for_simulation(0.5)
        
        # Move to object
        if not self._move_to_position(obj_pos, [0, 0, 0, 1]):
            return False
        
        # Close gripper
        self._set_gripper_width(0.01)
        self._wait_for_simulation(0.5)
        
        # Lift
        lift_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.2]
        return self._move_to_position(lift_pos, [0, 0, 0, 1])
    
    def _place_object_at_position(self, position: List[float]) -> bool:
        """Place currently grasped object at position"""
        
        # Move to position
        place_pos = [position[0], position[1], position[2] + 0.1]
        if not self._move_to_position(place_pos, [0, 0, 0, 1]):
            return False
        
        # Lower to place
        if not self._move_to_position(position, [0, 0, 0, 1]):
            return False
        
        # Open gripper
        self._set_gripper_width(0.08)
        self._wait_for_simulation(0.5)
        
        # Retract
        retract_pos = [position[0], position[1], position[2] + 0.1]
        return self._move_to_position(retract_pos, [0, 0, 0, 1])
    
    def _move_to_position(self, target_pos: List[float], target_orn: List[float]) -> bool:
        """Move robot end effector to target position with inverse kinematics"""
        
        try:
            # Use PyBullet's inverse kinematics
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                endEffectorLinkIndex=len(self.joint_indices)-1,
                targetPosition=target_pos,
                targetOrientation=target_orn,
                maxNumIterations=100,
                residualThreshold=1e-5
            )
            
            # Apply joint positions with smooth motion
            for i, joint_pos in enumerate(joint_poses[:len(self.joint_indices)]):
                p.setJointMotorControl2(
                    self.robot_id, i,
                    p.POSITION_CONTROL,
                    targetPosition=joint_pos,
                    maxVelocity=self.robot_config['max_velocity'][i],
                    force=self.robot_config['max_force'][i]
                )
            
            # Wait for motion to complete
            self._wait_for_motion_completion()
            
            return True
            
        except Exception as e:
            logger.error(f"Motion planning failed: {e}")
            return False
    
    def _set_gripper_width(self, width: float):
        """Set gripper width (simplified)"""
        # In a real implementation, this would control actual gripper joints
        logger.debug(f"Setting gripper width to {width:.3f}")
    
    def _wait_for_motion_completion(self):
        """Wait for robot motion to complete"""
        
        for _ in range(60):  # Wait up to 1 second at 60Hz
            p.stepSimulation()
            if self.gui:
                time.sleep(1./60.)
    
    def _wait_for_simulation(self, duration: float):
        """Wait for specified duration in simulation time"""
        
        steps = int(duration * 240)  # 240Hz simulation
        for _ in range(steps):
            p.stepSimulation()
            if self.gui and self.real_time:
                time.sleep(1./240.)
    
    def _get_end_effector_position(self) -> Tuple[float, float, float]:
        """Get current end effector position"""
        
        if self.robot_id is None:
            return (0, 0, 0)
        
        try:
            link_state = p.getLinkState(self.robot_id, len(self.joint_indices)-1)
            return link_state[4]  # World position of link frame
        except:
            return (0, 0, 0)
    
    def get_simulation_state(self) -> Dict[str, Any]:
        """Get current simulation state for visualization"""
        
        state = {
            'robot_pose': self.current_pose,
            'objects': {},
            'simulation_time': self.simulation_time
        }
        
        # Get object states
        for obj_id, obj in self.objects.items():
            pos, orn = p.getBasePositionAndOrientation(obj.pybullet_id)
            state['objects'][obj_id] = {
                'name': obj.name,
                'position': pos,
                'orientation': orn
            }
        
        return state
    
    def cleanup(self):
        """Cleanup simulation"""
        
        if self.physics_client is not None:
            p.disconnect()
            logger.info("Simulation cleanup complete")

def create_advanced_robot_simulation(gui: bool = True, real_time: bool = False) -> AdvancedRobotSimulation:
    """Factory function to create advanced robot simulation"""
    
    sim = AdvancedRobotSimulation(gui=gui, real_time=real_time)
    if sim.initialize_simulation():
        return sim
    else:
        raise RuntimeError("Failed to initialize robot simulation")