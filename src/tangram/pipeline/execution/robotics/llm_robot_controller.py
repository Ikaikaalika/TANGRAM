#!/usr/bin/env python3
"""
LLM Robot Controller for TANGRAM

Bridges natural language instructions from LLM to robot actions.
Converts high-level commands to executable robot tasks.

Author: TANGRAM Team
License: MIT
"""

import json
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from .robot_arm_sim import RobotArmSimulation
from ..reconstruction.object_3d_mapper import Object3DMapper, Object3D
from ..llm.local_llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of robot actions"""
    PICK = "pick"
    PLACE = "place"
    MOVE = "move"
    OBSERVE = "observe"
    WAIT = "wait"
    ORGANIZE = "organize"

@dataclass
class RobotAction:
    """Single robot action"""
    action_type: ActionType
    target_object: Optional[str] = None
    target_position: Optional[np.ndarray] = None
    parameters: Optional[Dict[str, Any]] = None
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "action_type": self.action_type.value,
            "target_object": self.target_object,
            "target_position": self.target_position.tolist() if self.target_position is not None else None,
            "parameters": self.parameters,
            "description": self.description
        }

@dataclass
class TaskPlan:
    """Complete task plan with multiple actions"""
    actions: List[RobotAction]
    description: str
    safety_notes: List[str]
    estimated_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "actions": [action.to_dict() for action in self.actions],
            "description": self.description,
            "safety_notes": self.safety_notes,
            "estimated_duration": self.estimated_duration,
            "num_actions": len(self.actions)
        }

class LLMRobotController:
    """LLM-powered robot controller"""
    
    def __init__(self, 
                 robot_sim: RobotArmSimulation,
                 object_mapper: Object3DMapper,
                 llm_client: UnifiedLLMClient):
        """
        Initialize LLM robot controller
        
        Args:
            robot_sim: Robot simulation instance
            object_mapper: 3D object mapper
            llm_client: LLM client for natural language processing
        """
        self.robot_sim = robot_sim
        self.object_mapper = object_mapper
        self.llm_client = llm_client
        
        # Task execution state
        self.current_plan = None
        self.execution_history = []
        self.active_constraints = []  # For object grasping
        
        # Safety parameters
        self.workspace_bounds = {
            "x": [-0.8, 0.8],
            "y": [-0.8, 0.8],
            "z": [0.0, 1.0]
        }
        
        # Common positions
        self.common_positions = {
            "home": np.array([0.0, 0.0, 0.5]),
            "table_center": np.array([0.5, 0.0, 0.45]),
            "safe_position": np.array([0.0, 0.0, 0.7])
        }
    
    def process_natural_language_command(self, command: str) -> TaskPlan:
        """
        Process natural language command and generate task plan
        
        Args:
            command: Natural language command
            
        Returns:
            Generated task plan
        """
        logger.info(f"Processing command: {command}")
        
        # Get current scene state
        scene_description = self.object_mapper.create_scene_description()
        
        # Create LLM prompt
        prompt = self._create_task_planning_prompt(command, scene_description)
        
        # Get LLM response
        try:
            response = self.llm_client.generate_response(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse response into task plan
            task_plan = self._parse_llm_response(response["content"])
            
            logger.info(f"Generated task plan with {len(task_plan.actions)} actions")
            return task_plan
            
        except Exception as e:
            logger.error(f"Failed to process command: {e}")
            return self._create_fallback_plan(command)
    
    def _create_task_planning_prompt(self, command: str, scene_description: Dict[str, Any]) -> str:
        """Create prompt for task planning"""
        
        objects_str = "\\n".join([
            f"- {obj['name']} at position {obj['position']} (ID: {obj['id']})"
            for obj in scene_description["objects"]
        ])
        
        relationships_str = "\\n".join([
            f"- Object {rel['object1']} is {rel['relation']} object {rel['object2']}"
            for rel in scene_description["spatial_relationships"][:10]  # Limit to avoid token overflow
        ])
        
        prompt = f"""
ROBOT TASK PLANNING REQUEST

USER COMMAND: {command}

CURRENT SCENE:
{scene_description['summary']}

OBJECTS IN SCENE:
{objects_str}

SPATIAL RELATIONSHIPS:
{relationships_str}

WORKSPACE BOUNDS:
- X: {self.workspace_bounds['x'][0]} to {self.workspace_bounds['x'][1]}
- Y: {self.workspace_bounds['y'][0]} to {self.workspace_bounds['y'][1]}
- Z: {self.workspace_bounds['z'][0]} to {self.workspace_bounds['z'][1]}

Please generate a detailed task plan as a JSON object with the following structure:
{{
    "description": "Brief description of the overall task",
    "actions": [
        {{
            "action_type": "pick|place|move|observe|wait|organize",
            "target_object": "object_name or null",
            "target_position": [x, y, z] or null,
            "description": "Description of this action"
        }}
    ],
    "safety_notes": ["Safety consideration 1", "Safety consideration 2"],
    "estimated_duration": 30.0
}}

IMPORTANT:
- Use only objects that exist in the scene
- Ensure all positions are within workspace bounds
- Consider object positions and relationships
- Include safety considerations
- Break complex tasks into simple actions
- Use precise 3D coordinates
"""
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """You are a robot task planner for TANGRAM. Your role is to convert natural language commands into executable robot actions.

You have access to a robot arm that can:
- Pick up objects
- Place objects at specific locations
- Move to positions
- Observe the scene
- Wait for specified duration
- Organize objects

Always prioritize safety and feasibility. Generate realistic, executable plans with precise coordinates. Consider object positions, workspace constraints, and physics limitations.

Respond ONLY with valid JSON as specified in the prompt."""
    
    def _parse_llm_response(self, response: str) -> TaskPlan:
        """Parse LLM response into TaskPlan"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\\{.*\\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                plan_data = json.loads(json_str)
            else:
                # Try to parse the whole response as JSON
                plan_data = json.loads(response)
            
            # Convert to TaskPlan
            actions = []
            for action_data in plan_data.get("actions", []):
                action_type = ActionType(action_data["action_type"])
                
                target_position = None
                if action_data.get("target_position"):
                    target_position = np.array(action_data["target_position"])
                
                action = RobotAction(
                    action_type=action_type,
                    target_object=action_data.get("target_object"),
                    target_position=target_position,
                    parameters=action_data.get("parameters"),
                    description=action_data.get("description", "")
                )
                actions.append(action)
            
            task_plan = TaskPlan(
                actions=actions,
                description=plan_data.get("description", ""),
                safety_notes=plan_data.get("safety_notes", []),
                estimated_duration=plan_data.get("estimated_duration", 0.0)
            )
            
            return task_plan
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._create_fallback_plan(f"Parse error: {str(e)}")
    
    def _create_fallback_plan(self, command: str) -> TaskPlan:
        """Create fallback plan when LLM fails"""
        # Simple fallback based on command keywords
        actions = []
        
        if "pick" in command.lower():
            # Try to pick up first available object
            objects = list(self.object_mapper.objects_3d.values())
            if objects:
                obj = objects[0]
                actions.append(RobotAction(
                    action_type=ActionType.PICK,
                    target_object=obj.class_name,
                    target_position=obj.position_3d,
                    description=f"Pick up {obj.class_name}"
                ))
        
        elif "move" in command.lower():
            # Move to table center
            actions.append(RobotAction(
                action_type=ActionType.MOVE,
                target_position=self.common_positions["table_center"],
                description="Move to table center"
            ))
        
        else:
            # Default: observe scene
            actions.append(RobotAction(
                action_type=ActionType.OBSERVE,
                description="Observe the current scene"
            ))
        
        return TaskPlan(
            actions=actions,
            description=f"Fallback plan for: {command}",
            safety_notes=["This is a fallback plan with limited capabilities"],
            estimated_duration=10.0
        )
    
    def execute_task_plan(self, task_plan: TaskPlan) -> Dict[str, Any]:
        """
        Execute a complete task plan
        
        Args:
            task_plan: Task plan to execute
            
        Returns:
            Execution results
        """
        logger.info(f"Executing task plan: {task_plan.description}")
        
        self.current_plan = task_plan
        execution_results = {
            "success": True,
            "completed_actions": 0,
            "failed_actions": 0,
            "execution_log": [],
            "final_state": None
        }
        
        for i, action in enumerate(task_plan.actions):
            logger.info(f"Executing action {i+1}/{len(task_plan.actions)}: {action.description}")
            
            try:
                # Execute action
                action_result = self._execute_action(action)
                
                execution_results["execution_log"].append({
                    "action_index": i,
                    "action": action.to_dict(),
                    "success": action_result["success"],
                    "details": action_result.get("details", ""),
                    "duration": action_result.get("duration", 0.0)
                })
                
                if action_result["success"]:
                    execution_results["completed_actions"] += 1
                else:
                    execution_results["failed_actions"] += 1
                    logger.warning(f"Action {i+1} failed: {action_result.get('error', 'Unknown error')}")
                    
                    # Decide whether to continue or abort
                    if action.action_type in [ActionType.PICK, ActionType.PLACE]:
                        # Critical actions - abort on failure
                        execution_results["success"] = False
                        break
                
            except Exception as e:
                logger.error(f"Action {i+1} execution error: {e}")
                execution_results["failed_actions"] += 1
                execution_results["execution_log"].append({
                    "action_index": i,
                    "action": action.to_dict(),
                    "success": False,
                    "error": str(e)
                })
                
                # Abort on exception
                execution_results["success"] = False
                break
        
        # Get final scene state
        execution_results["final_state"] = self.robot_sim.get_scene_state()
        
        # Add to execution history
        self.execution_history.append({
            "task_plan": task_plan.to_dict(),
            "results": execution_results,
            "timestamp": __import__('time').time()
        })
        
        logger.info(f"Task execution completed. Success: {execution_results['success']}")
        return execution_results
    
    def _execute_action(self, action: RobotAction) -> Dict[str, Any]:
        """Execute a single robot action"""
        import time
        start_time = time.time()
        
        try:
            if action.action_type == ActionType.PICK:
                return self._execute_pick_action(action)
            elif action.action_type == ActionType.PLACE:
                return self._execute_place_action(action)
            elif action.action_type == ActionType.MOVE:
                return self._execute_move_action(action)
            elif action.action_type == ActionType.OBSERVE:
                return self._execute_observe_action(action)
            elif action.action_type == ActionType.WAIT:
                return self._execute_wait_action(action)
            elif action.action_type == ActionType.ORGANIZE:
                return self._execute_organize_action(action)
            else:
                return {"success": False, "error": f"Unknown action type: {action.action_type}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
        
        finally:
            duration = time.time() - start_time
            return {"success": True, "duration": duration}
    
    def _execute_pick_action(self, action: RobotAction) -> Dict[str, Any]:
        """Execute pick action"""
        if action.target_object is None:
            return {"success": False, "error": "No target object specified"}
        
        # Find object by name
        target_obj = None
        for obj in self.object_mapper.objects_3d.values():
            if obj.class_name == action.target_object:
                target_obj = obj
                break
        
        if target_obj is None:
            return {"success": False, "error": f"Object '{action.target_object}' not found in scene"}
        
        # Find corresponding simulation object
        sim_object_id = None
        for obj_id, obj_info in self.robot_sim.objects.items():
            if obj_info["type"] == target_obj.class_name:
                sim_object_id = obj_info["id"]
                break
        
        if sim_object_id is None:
            return {"success": False, "error": f"Simulation object for '{action.target_object}' not found"}
        
        # Execute pick
        constraint_id = self.robot_sim.pick_object(sim_object_id)
        
        if constraint_id:
            self.active_constraints.append(constraint_id)
            return {"success": True, "details": f"Picked up {action.target_object}", "constraint_id": constraint_id}
        else:
            return {"success": False, "error": f"Failed to pick up {action.target_object}"}
    
    def _execute_place_action(self, action: RobotAction) -> Dict[str, Any]:
        """Execute place action"""
        if action.target_position is None:
            return {"success": False, "error": "No target position specified"}
        
        if not self.active_constraints:
            return {"success": False, "error": "No object currently grasped"}
        
        # Use the most recent constraint
        constraint_id = self.active_constraints[-1]
        
        # Execute place
        success = self.robot_sim.place_object(constraint_id, action.target_position.tolist())
        
        if success:
            self.active_constraints.remove(constraint_id)
            return {"success": True, "details": f"Placed object at {action.target_position}"}
        else:
            return {"success": False, "error": f"Failed to place object at {action.target_position}"}
    
    def _execute_move_action(self, action: RobotAction) -> Dict[str, Any]:
        """Execute move action"""
        if action.target_position is None:
            return {"success": False, "error": "No target position specified"}
        
        success = self.robot_sim.move_to_position(action.target_position.tolist())
        
        if success:
            return {"success": True, "details": f"Moved to {action.target_position}"}
        else:
            return {"success": False, "error": f"Failed to move to {action.target_position}"}
    
    def _execute_observe_action(self, action: RobotAction) -> Dict[str, Any]:
        """Execute observe action"""
        scene_state = self.robot_sim.get_scene_state()
        return {"success": True, "details": f"Observed {len(scene_state['objects'])} objects", "scene_state": scene_state}
    
    def _execute_wait_action(self, action: RobotAction) -> Dict[str, Any]:
        """Execute wait action"""
        import time
        duration = action.parameters.get("duration", 1.0) if action.parameters else 1.0
        time.sleep(duration)
        return {"success": True, "details": f"Waited {duration} seconds"}
    
    def _execute_organize_action(self, action: RobotAction) -> Dict[str, Any]:
        """Execute organize action (complex multi-step)"""
        # This is a meta-action that generates a sequence of pick/place actions
        # For now, implement a simple version that moves objects to a grid
        
        objects = list(self.object_mapper.objects_3d.values())
        if not objects:
            return {"success": False, "error": "No objects to organize"}
        
        # Create grid positions
        grid_positions = []
        for i, obj in enumerate(objects):
            x = 0.4 + (i % 3) * 0.1  # 3 columns, 10cm apart
            y = 0.0 + (i // 3) * 0.1  # Rows 10cm apart
            z = 0.45  # Table height
            grid_positions.append([x, y, z])
        
        # Execute pick and place for each object
        organized_count = 0
        for i, obj in enumerate(objects):
            if i >= len(grid_positions):
                break
                
            # Pick
            pick_action = RobotAction(ActionType.PICK, target_object=obj.class_name)
            pick_result = self._execute_pick_action(pick_action)
            
            if pick_result["success"]:
                # Place
                place_action = RobotAction(ActionType.PLACE, target_position=np.array(grid_positions[i]))
                place_result = self._execute_place_action(place_action)
                
                if place_result["success"]:
                    organized_count += 1
        
        return {"success": True, "details": f"Organized {organized_count} objects into grid"}
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        status = {
            "current_plan": self.current_plan.to_dict() if self.current_plan else None,
            "active_constraints": len(self.active_constraints),
            "execution_history_count": len(self.execution_history),
            "robot_state": self.robot_sim.get_scene_state()
        }
        return status
    
    def emergency_stop(self):
        """Emergency stop - release all constraints and move to safe position"""
        logger.warning("Emergency stop activated")
        
        # Release all constraints
        for constraint_id in self.active_constraints:
            try:
                self.robot_sim.place_object(constraint_id, [0.5, 0.5, 0.45])
            except:
                pass
        
        self.active_constraints.clear()
        
        # Move to safe position
        self.robot_sim.move_to_position(self.common_positions["safe_position"].tolist())
        
        logger.info("Emergency stop completed")

def main():
    """Test LLM robot controller"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM Robot Controller")
    parser.add_argument("--colmap-dir", required=True, help="COLMAP reconstruction directory")
    parser.add_argument("--command", required=True, help="Natural language command")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    
    args = parser.parse_args()
    
    # Initialize components
    robot_sim = RobotArmSimulation(gui=not args.no_gui)
    object_mapper = Object3DMapper(args.colmap_dir)
    llm_client = UnifiedLLMClient(prefer_local=True)
    
    # Initialize controller
    controller = LLMRobotController(robot_sim, object_mapper, llm_client)
    
    try:
        # Setup simulation
        robot_sim.reset_simulation()
        robot_sim.create_demo_scene()
        
        # Create some fake 3D objects for testing
        # (In real usage, these would come from object_mapper.triangulate_objects())
        from ..reconstruction.object_3d_mapper import Object3D
        
        fake_objects = {
            1: Object3D(
                id=1, track_id=1, class_name="cube", confidence=0.9,
                position_3d=np.array([0.4, 0.2, 0.5]), bbox_2d=[100, 100, 150, 150]
            ),
            2: Object3D(
                id=2, track_id=2, class_name="sphere", confidence=0.8,
                position_3d=np.array([0.6, 0.2, 0.5]), bbox_2d=[200, 100, 250, 150]
            )
        }
        
        object_mapper.objects_3d = fake_objects
        
        # Process command
        print(f"Processing command: {args.command}")
        task_plan = controller.process_natural_language_command(args.command)
        
        print(f"Generated plan: {task_plan.description}")
        print(f"Actions: {len(task_plan.actions)}")
        
        # Execute plan
        results = controller.execute_task_plan(task_plan)
        
        print(f"Execution results: {results['success']}")
        print(f"Completed actions: {results['completed_actions']}")
        print(f"Failed actions: {results['failed_actions']}")
        
        # Run simulation
        robot_sim.run_simulation(10.0)
        
    except KeyboardInterrupt:
        controller.emergency_stop()
        print("Interrupted by user")
    
    finally:
        robot_sim.disconnect()

if __name__ == "__main__":
    main()