#!/usr/bin/env python3
"""
Advanced LLM Task Planner

Intelligent task planning using local DeepSeek R1 7B model with:
- Complex scene understanding
- Multi-step task decomposition
- Constraint-aware planning
- Safety validation
- Execution monitoring

Author: TANGRAM Team
License: MIT
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.tangram.core.llm.local_llm_client import LocalLLMClient
from src.tangram.core.computer_vision.advanced_reconstruction import DetectedObject, SceneReconstruction
from src.tangram.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class TaskType(Enum):
    """Types of robotic tasks"""
    PICK = "pick"
    PLACE = "place"
    STACK = "stack"
    ORGANIZE = "organize"
    MOVE = "move"
    SORT = "sort"
    CLEAN = "clean"

@dataclass
class TaskStep:
    """Individual task step"""
    step_id: int
    action: TaskType
    target_object: str
    destination: Optional[Tuple[float, float, float]] = None
    constraints: List[str] = None
    estimated_duration: float = 0.0
    priority: int = 1
    safety_checks: List[str] = None

@dataclass
class TaskPlan:
    """Complete task plan"""
    task_id: str
    description: str
    steps: List[TaskStep]
    estimated_total_time: float
    complexity_score: float
    safety_score: float
    success_probability: float
    alternative_plans: List['TaskPlan'] = None

class AdvancedTaskPlanner:
    """
    Advanced LLM-powered task planner for robotic manipulation
    """
    
    def __init__(self):
        """Initialize the advanced task planner"""
        
        self.llm_client = None
        self.scene_context = None
        self.workspace_constraints = {
            'max_reach': 0.8,  # meters
            'max_lift_height': 1.5,  # meters
            'max_object_weight': 2.0,  # kg
            'collision_margin': 0.05,  # meters
        }
        
        # Task planning templates
        self.planning_templates = {
            'scene_analysis': self._get_scene_analysis_prompt(),
            'task_decomposition': self._get_task_decomposition_prompt(),
            'safety_validation': self._get_safety_validation_prompt(),
            'execution_monitoring': self._get_execution_monitoring_prompt()
        }
        
    def initialize_llm(self) -> bool:
        """Initialize local LLM client"""
        
        try:
            self.llm_client = LocalLLMClient()
            logger.info("✅ Advanced task planner LLM initialized")
            return True
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            return False
    
    def analyze_scene(self, scene: SceneReconstruction) -> Dict[str, Any]:
        """Analyze scene for task planning context"""
        
        logger.info("Analyzing scene for task planning...")
        
        # Extract scene information
        objects_info = []
        for obj in scene.objects:
            obj_info = {
                'name': obj.class_name,
                'id': obj.id,
                'position': obj.center_3d,
                'dimensions': obj.dimensions,
                'confidence': obj.confidence
            }
            objects_info.append(obj_info)
        
        # Create scene context
        self.scene_context = {
            'objects': objects_info,
            'object_count': len(objects_info),
            'workspace_bounds': scene.workspace_bounds,
            'object_types': list(set(obj['name'] for obj in objects_info)),
            'spatial_relationships': self._analyze_spatial_relationships(objects_info)
        }
        
        # Get LLM scene analysis
        if self.llm_client:
            scene_analysis = self._get_llm_scene_analysis()
            self.scene_context['llm_analysis'] = scene_analysis
        
        logger.info(f"Scene analysis complete: {len(objects_info)} objects, "
                   f"{len(self.scene_context['object_types'])} types")
        
        return self.scene_context
    
    def create_task_plan(self, task_description: str, constraints: Dict[str, Any] = None) -> TaskPlan:
        """Create comprehensive task plan from natural language description"""
        
        logger.info(f"Creating task plan for: {task_description}")
        
        if not self.scene_context:
            raise ValueError("Scene must be analyzed before task planning")
        
        # Parse task description with LLM
        task_analysis = self._analyze_task_with_llm(task_description)
        
        # Decompose into steps
        task_steps = self._decompose_task_into_steps(task_analysis)
        
        # Validate feasibility
        validated_steps = self._validate_task_feasibility(task_steps)
        
        # Estimate timing and complexity
        timing_info = self._estimate_task_timing(validated_steps)
        
        # Safety analysis
        safety_analysis = self._perform_safety_analysis(validated_steps)
        
        # Create final plan
        task_plan = TaskPlan(
            task_id=f"task_{int(time.time())}",
            description=task_description,
            steps=validated_steps,
            estimated_total_time=timing_info['total_time'],
            complexity_score=timing_info['complexity'],
            safety_score=safety_analysis['safety_score'],
            success_probability=self._estimate_success_probability(validated_steps)
        )
        
        logger.info(f"✅ Task plan created: {len(validated_steps)} steps, "
                   f"{timing_info['total_time']:.1f}s estimated")
        
        return task_plan
    
    def _analyze_task_with_llm(self, task_description: str) -> Dict[str, Any]:
        """Use LLM to analyze and understand the task"""
        
        if not self.llm_client:
            return self._fallback_task_analysis(task_description)
        
        # Build comprehensive prompt
        prompt = self.planning_templates['task_decomposition'].format(
            task_description=task_description,
            scene_objects=json.dumps([obj['name'] for obj in self.scene_context['objects']], indent=2),
            workspace_constraints=json.dumps(self.workspace_constraints, indent=2),
            spatial_relationships=json.dumps(self.scene_context['spatial_relationships'], indent=2)
        )
        
        try:
            response = self.llm_client.generate_response(prompt)
            
            # Parse LLM response
            task_analysis = self._parse_llm_task_response(response)
            
            logger.debug(f"LLM task analysis: {task_analysis}")
            return task_analysis
            
        except Exception as e:
            logger.warning(f"LLM task analysis failed: {e}, using fallback")
            return self._fallback_task_analysis(task_description)
    
    def _decompose_task_into_steps(self, task_analysis: Dict[str, Any]) -> List[TaskStep]:
        """Decompose high-level task into executable steps"""
        
        steps = []
        step_id = 1
        
        # Extract main action type
        main_action = task_analysis.get('main_action', 'organize')
        target_objects = task_analysis.get('target_objects', [])
        
        if main_action in ['pick', 'pick_up']:
            # Simple pick task
            for obj_name in target_objects:
                step = TaskStep(
                    step_id=step_id,
                    action=TaskType.PICK,
                    target_object=obj_name,
                    estimated_duration=5.0,
                    safety_checks=['collision_check', 'reachability_check']
                )
                steps.append(step)
                step_id += 1
        
        elif main_action in ['stack', 'pile']:
            # Stacking task
            stack_center = task_analysis.get('stack_location', (0.3, 0.2, 0.8))
            
            for i, obj_name in enumerate(target_objects):
                # Pick step
                pick_step = TaskStep(
                    step_id=step_id,
                    action=TaskType.PICK,
                    target_object=obj_name,
                    estimated_duration=4.0,
                    safety_checks=['collision_check', 'grip_check']
                )
                steps.append(pick_step)
                step_id += 1
                
                # Place step
                place_height = stack_center[2] + (i * 0.05)
                place_step = TaskStep(
                    step_id=step_id,
                    action=TaskType.PLACE,
                    target_object=obj_name,
                    destination=(stack_center[0], stack_center[1], place_height),
                    estimated_duration=3.0,
                    safety_checks=['stability_check', 'placement_check']
                )
                steps.append(place_step)
                step_id += 1
        
        elif main_action in ['organize', 'arrange', 'sort']:
            # Organization task
            positions = self._generate_organization_positions(len(target_objects))
            
            for obj_name, position in zip(target_objects, positions):
                # Pick step
                pick_step = TaskStep(
                    step_id=step_id,
                    action=TaskType.PICK,
                    target_object=obj_name,
                    estimated_duration=4.0
                )
                steps.append(pick_step)
                step_id += 1
                
                # Place step
                place_step = TaskStep(
                    step_id=step_id,
                    action=TaskType.PLACE,
                    target_object=obj_name,
                    destination=position,
                    estimated_duration=3.0
                )
                steps.append(place_step)
                step_id += 1
        
        return steps
    
    def _validate_task_feasibility(self, steps: List[TaskStep]) -> List[TaskStep]:
        """Validate that task steps are physically feasible"""
        
        validated_steps = []
        
        for step in steps:
            # Check reachability
            if step.destination:
                if self._is_position_reachable(step.destination):
                    validated_steps.append(step)
                else:
                    logger.warning(f"Step {step.step_id} destination unreachable: {step.destination}")
                    # Modify to reachable position
                    modified_step = self._modify_to_reachable_position(step)
                    validated_steps.append(modified_step)
            else:
                validated_steps.append(step)
        
        return validated_steps
    
    def _estimate_task_timing(self, steps: List[TaskStep]) -> Dict[str, float]:
        """Estimate task execution timing"""
        
        total_time = 0.0
        complexity_factors = []
        
        for step in steps:
            # Base time from step
            step_time = step.estimated_duration
            
            # Add complexity factors
            if step.action == TaskType.PICK:
                step_time += 2.0  # Approach and grasp time
            elif step.action == TaskType.PLACE:
                step_time += 1.5  # Placement precision time
            
            # Safety check overhead
            if step.safety_checks:
                step_time += len(step.safety_checks) * 0.5
            
            total_time += step_time
            complexity_factors.append(step_time / 3.0)  # Normalize complexity
        
        complexity_score = sum(complexity_factors) / len(complexity_factors) if complexity_factors else 1.0
        
        return {
            'total_time': total_time,
            'complexity': min(complexity_score, 10.0)  # Cap at 10
        }
    
    def _perform_safety_analysis(self, steps: List[TaskStep]) -> Dict[str, Any]:
        """Perform comprehensive safety analysis"""
        
        safety_issues = []
        safety_score = 10.0  # Start with perfect score
        
        for step in steps:
            # Check for collision risks
            if step.destination:
                if self._check_collision_risk(step.destination):
                    safety_issues.append(f"Collision risk at step {step.step_id}")
                    safety_score -= 1.0
            
            # Check object weight limits
            obj_info = self._get_object_info(step.target_object)
            if obj_info and obj_info.get('estimated_weight', 0) > self.workspace_constraints['max_object_weight']:
                safety_issues.append(f"Object too heavy: {step.target_object}")
                safety_score -= 2.0
            
            # Check workspace boundaries
            if step.destination and not self._is_within_workspace(step.destination):
                safety_issues.append(f"Destination outside workspace: {step.destination}")
                safety_score -= 1.5
        
        return {
            'safety_score': max(safety_score, 0.0),
            'safety_issues': safety_issues,
            'is_safe': safety_score > 7.0
        }
    
    def _estimate_success_probability(self, steps: List[TaskStep]) -> float:
        """Estimate probability of successful task execution"""
        
        base_probability = 0.95  # High base success rate
        
        # Reduce probability based on complexity
        complexity_penalty = len(steps) * 0.02  # 2% per step
        
        # Reduce based on precision requirements
        precision_penalty = 0.0
        for step in steps:
            if step.action == TaskType.PLACE:
                precision_penalty += 0.05  # 5% per placement
        
        final_probability = base_probability - complexity_penalty - precision_penalty
        return max(final_probability, 0.5)  # Minimum 50% probability
    
    def monitor_execution(self, plan: TaskPlan, current_step: int, 
                         execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor task execution and provide guidance"""
        
        if current_step >= len(plan.steps):
            return {'status': 'completed', 'message': 'Task completed successfully'}
        
        current_task_step = plan.steps[current_step]
        
        # Check execution state
        monitoring_result = {
            'status': 'executing',
            'current_step': current_step,
            'step_description': f"{current_task_step.action.value} {current_task_step.target_object}",
            'progress': current_step / len(plan.steps),
            'estimated_remaining': sum(step.estimated_duration for step in plan.steps[current_step:])
        }
        
        # Check for execution problems
        if execution_state.get('error'):
            monitoring_result['status'] = 'error'
            monitoring_result['error'] = execution_state['error']
            monitoring_result['recovery_suggestions'] = self._suggest_recovery_actions(
                current_task_step, execution_state['error']
            )
        
        return monitoring_result
    
    def _get_scene_analysis_prompt(self) -> str:
        return """
        Analyze this robotic manipulation scene:

        Objects detected: {objects}
        Workspace bounds: {workspace_bounds}
        
        Provide analysis of:
        1. Object arrangement and spatial relationships
        2. Manipulation opportunities and constraints
        3. Recommended approach strategies
        4. Potential challenges or risks
        
        Focus on practical robotic manipulation considerations.
        """
    
    def _get_task_decomposition_prompt(self) -> str:
        return """
        You are an expert robotic task planner. Analyze this manipulation task:

        Task: "{task_description}"
        
        Available objects: {scene_objects}
        Workspace constraints: {workspace_constraints}
        Spatial relationships: {spatial_relationships}
        
        Decompose this task into specific, executable steps. Consider:
        1. Object accessibility and reachability
        2. Required precision and safety
        3. Optimal sequence to avoid collisions
        4. Physical constraints and limitations
        
        Respond with JSON format:
        {{
            "main_action": "pick|place|stack|organize",
            "target_objects": ["object1", "object2"],
            "strategy": "brief strategy description",
            "estimated_difficulty": "low|medium|high",
            "key_constraints": ["constraint1", "constraint2"]
        }}
        """
    
    def _get_safety_validation_prompt(self) -> str:
        return """
        Validate the safety of this robotic manipulation plan:
        
        Steps: {steps}
        Workspace: {workspace}
        
        Check for:
        1. Collision risks
        2. Stability issues
        3. Reachability problems
        4. Object handling safety
        
        Rate safety from 1-10 and suggest improvements.
        """
    
    def _get_execution_monitoring_prompt(self) -> str:
        return """
        Monitor execution of robotic task:
        
        Current step: {current_step}
        Execution state: {execution_state}
        
        Assess progress and provide:
        1. Status evaluation
        2. Problem identification
        3. Recovery recommendations
        4. Next step guidance
        """
    
    def _fallback_task_analysis(self, task_description: str) -> Dict[str, Any]:
        """Fallback task analysis without LLM"""
        
        # Simple keyword-based analysis
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ['pick', 'grab', 'take']):
            main_action = 'pick'
        elif any(word in task_lower for word in ['stack', 'pile']):
            main_action = 'stack'
        elif any(word in task_lower for word in ['organize', 'arrange', 'sort']):
            main_action = 'organize'
        else:
            main_action = 'organize'
        
        # Extract object names from scene
        target_objects = [obj['name'] for obj in self.scene_context['objects']]
        
        return {
            'main_action': main_action,
            'target_objects': target_objects[:3],  # Limit to 3 objects
            'strategy': 'basic manipulation',
            'estimated_difficulty': 'medium',
            'key_constraints': ['reachability', 'collision_avoidance']
        }
    
    def _parse_llm_task_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured task analysis"""
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback parsing
        return self._fallback_task_analysis(response)
    
    def _analyze_spatial_relationships(self, objects_info: List[Dict]) -> Dict[str, Any]:
        """Analyze spatial relationships between objects"""
        
        relationships = {}
        
        for i, obj1 in enumerate(objects_info):
            for j, obj2 in enumerate(objects_info[i+1:], i+1):
                distance = self._calculate_distance(obj1['position'], obj2['position'])
                
                if distance < 0.1:  # Very close
                    rel_type = 'adjacent'
                elif distance < 0.2:  # Close
                    rel_type = 'near'
                else:
                    rel_type = 'distant'
                
                relationships[f"{obj1['name']}_{obj2['name']}"] = {
                    'type': rel_type,
                    'distance': distance
                }
        
        return relationships
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate 3D distance between positions"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)**0.5
    
    def _generate_organization_positions(self, num_objects: int) -> List[Tuple[float, float, float]]:
        """Generate organized positions for objects"""
        
        positions = []
        grid_center = (0.4, 0.0, 0.8)
        spacing = 0.15
        
        for i in range(num_objects):
            row = i // 3
            col = i % 3
            
            x = grid_center[0] + (col - 1) * spacing
            y = grid_center[1] + (row - 1) * spacing
            z = grid_center[2]
            
            positions.append((x, y, z))
        
        return positions
    
    def _is_position_reachable(self, position: Tuple[float, float, float]) -> bool:
        """Check if position is within robot reach"""
        
        # Simple reach check based on distance from robot base
        robot_base = (0, 0, 0)
        distance = self._calculate_distance(robot_base, position)
        
        return distance <= self.workspace_constraints['max_reach']
    
    def _modify_to_reachable_position(self, step: TaskStep) -> TaskStep:
        """Modify step destination to be reachable"""
        
        if not step.destination:
            return step
        
        # Move position closer to robot base
        robot_base = (0, 0, 0)
        direction = [
            step.destination[0] - robot_base[0],
            step.destination[1] - robot_base[1],
            step.destination[2] - robot_base[2]
        ]
        
        # Normalize and scale to max reach
        distance = (direction[0]**2 + direction[1]**2 + direction[2]**2)**0.5
        if distance > 0:
            scale = self.workspace_constraints['max_reach'] * 0.9 / distance
            new_destination = (
                robot_base[0] + direction[0] * scale,
                robot_base[1] + direction[1] * scale,
                robot_base[2] + direction[2] * scale
            )
            step.destination = new_destination
        
        return step
    
    def _check_collision_risk(self, position: Tuple[float, float, float]) -> bool:
        """Check for collision risk at position"""
        
        # Check against other object positions
        for obj in self.scene_context['objects']:
            obj_pos = obj['position']
            distance = self._calculate_distance(position, obj_pos)
            
            if distance < self.workspace_constraints['collision_margin']:
                return True
        
        return False
    
    def _is_within_workspace(self, position: Tuple[float, float, float]) -> bool:
        """Check if position is within workspace bounds"""
        
        if not self.scene_context['workspace_bounds']:
            return True
        
        bounds = self.scene_context['workspace_bounds']
        x, y, z = position
        
        return (bounds[0] <= x <= bounds[1] and 
                bounds[2] <= y <= bounds[3] and 
                bounds[4] <= z <= bounds[5])
    
    def _get_object_info(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Get information about specific object"""
        
        for obj in self.scene_context['objects']:
            if object_name.lower() in obj['name'].lower():
                return obj
        return None
    
    def _suggest_recovery_actions(self, step: TaskStep, error: str) -> List[str]:
        """Suggest recovery actions for execution errors"""
        
        suggestions = []
        
        if 'collision' in error.lower():
            suggestions.append("Retry with modified approach path")
            suggestions.append("Clear obstacles from path")
        
        if 'grip' in error.lower() or 'grasp' in error.lower():
            suggestions.append("Adjust gripper width")
            suggestions.append("Retry grasp from different angle")
        
        if 'reach' in error.lower():
            suggestions.append("Move robot base closer")
            suggestions.append("Use alternative target position")
        
        if not suggestions:
            suggestions.append("Retry current step")
            suggestions.append("Skip to next step if safe")
        
        return suggestions
    
    def _get_llm_scene_analysis(self) -> str:
        """Get LLM analysis of the scene"""
        
        if not self.llm_client:
            return "Scene analysis not available (LLM not connected)"
        
        prompt = self.planning_templates['scene_analysis'].format(
            objects=json.dumps(self.scene_context['objects'], indent=2),
            workspace_bounds=self.scene_context['workspace_bounds']
        )
        
        try:
            return self.llm_client.generate_response(prompt)
        except Exception as e:
            logger.warning(f"LLM scene analysis failed: {e}")
            return "Scene analysis failed"

def create_advanced_task_planner() -> AdvancedTaskPlanner:
    """Factory function to create advanced task planner"""
    
    planner = AdvancedTaskPlanner()
    planner.initialize_llm()
    return planner