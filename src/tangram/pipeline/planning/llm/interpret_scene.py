#!/usr/bin/env python3

import requests
import json
import os
from typing import List, Dict, Any

# Import local LLM client
try:
    from .local_llm_client import LocalLLMClient, LocalDeepSeekInterpreter
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False

from config import LLM_CONFIG

def create_scene_interpreter():
    """
    Factory function to create scene interpreter with unified LLM support.
    
    Supports multiple backends: Hugging Face, Ollama, and Google Gemini.
    Automatically selects the best available option.
    
    Returns:
        Scene interpreter instance with unified LLM backend
        
    Raises:
        RuntimeError: If no LLM backend is available
    """
    try:
        # Import the unified LLM client
        from .unified_llm_client import UnifiedLLMClient
        
        # Create unified client (will auto-select best backend)
        llm_client = UnifiedLLMClient()
        
        # Create scene interpreter with unified client
        return UnifiedSceneInterpreter(llm_client)
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM backend: {e}")


class UnifiedSceneInterpreter:
    """
    Scene interpreter that works with the unified LLM client.
    Supports multiple backends through the unified client.
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.system_prompt = """You are TANGRAM, an AI assistant for robotic scene understanding and task planning.

Your role is to analyze 3D scenes and provide actionable insights for robot manipulation tasks.

Key capabilities:
- Interpret object relationships and spatial arrangements
- Generate step-by-step task plans for robotic manipulation
- Understand scene context and object affordances
- Provide clear, executable instructions

Always be concise, practical, and focused on actionable robot tasks."""
    
    def interpret_scene(self, scene_data: Dict[str, Any], goal: str = None) -> Dict[str, Any]:
        """
        Interpret scene data and generate task plan.
        
        Args:
            scene_data: Scene graph and object information
            goal: Optional goal description
            
        Returns:
            Interpretation results including task sequence
        """
        try:
            # Build comprehensive prompt
            prompt = self._build_scene_prompt(scene_data, goal)
            
            # Generate response using unified client
            response = self.llm_client.generate_response(prompt)
            
            # Parse response into structured format
            interpretation = self._parse_response(response)
            
            return interpretation
            
        except Exception as e:
            return {
                "error": str(e),
                "scene_description": "Failed to interpret scene",
                "task_sequence": [],
                "confidence": 0.0
            }
    
    def _build_scene_prompt(self, scene_data: Dict[str, Any], goal: str = None) -> str:
        """Build comprehensive scene analysis prompt"""
        prompt = f"{self.system_prompt}\n\n"
        
        # Add scene data
        prompt += "SCENE DATA:\n"
        if "nodes" in scene_data:
            prompt += "Objects detected:\n"
            for node_id, node_data in scene_data["nodes"].items():
                prompt += f"- {node_id}: {node_data}\n"
        
        if "edges" in scene_data:
            prompt += "\nSpatial relationships:\n"
            for edge in scene_data["edges"]:
                if len(edge) >= 3:
                    prompt += f"- {edge[0]} {edge[2]} {edge[1]}\n"
        
        # Add goal if provided
        if goal:
            prompt += f"\nGOAL: {goal}\n"
        
        prompt += """
TASK: Analyze this scene and provide:
1. Scene description (2-3 sentences)
2. Key objects and their relationships
3. Step-by-step task plan to achieve the goal
4. Confidence score (0-1)

Respond in JSON format:
{
    "scene_description": "Brief description of the scene",
    "key_objects": ["object1", "object2", ...],
    "relationships": ["relationship1", "relationship2", ...],
    "task_sequence": [
        {"step": 1, "action": "action_description", "target": "object_name"},
        {"step": 2, "action": "action_description", "target": "object_name"}
    ],
    "confidence": 0.8
}
"""
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback to basic parsing
                return {
                    "scene_description": response.strip(),
                    "key_objects": [],
                    "relationships": [],
                    "task_sequence": [],
                    "confidence": 0.5
                }
        except Exception:
            return {
                "scene_description": response.strip(),
                "key_objects": [],
                "relationships": [],
                "task_sequence": [],
                "confidence": 0.3
            }
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the LLM backend"""
        return self.llm_client.get_backend_info()


class DeepSeekSceneInterpreter:
    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.scene_context = {}
    
    def analyze_scene_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scene graph and extract key information"""
        
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        # Extract objects (excluding scene nodes)
        objects = []
        scene_info = None
        
        for node in nodes:
            node_props = node.get("properties", {})
            if node_props.get("type") == "scene_context":
                scene_info = node_props
            else:
                objects.append({
                    "id": node["id"],
                    "class": node_props.get("class_name", "unknown"),
                    "position": node_props.get("position_3d", [0, 0, 0]),
                    "track_id": node_props.get("track_id", -1)
                })
        
        # Extract relationships
        relationships = []
        for edge in edges:
            edge_props = edge.get("properties", {})
            relations = edge_props.get("relations", [edge_props.get("relation", "unknown")])
            
            for relation in relations:
                relationships.append({
                    "source": edge["source"],
                    "target": edge["target"],
                    "type": relation,
                    "distance": edge_props.get("distance_3d", 0)
                })
        
        interpretation = {
            "scene_type": scene_info.get("scene_type", "unknown") if scene_info else "unknown",
            "objects": objects,
            "relationships": relationships,
            "object_classes": list(set([obj["class"] for obj in objects])),
            "num_objects": len(objects)
        }
        
        return interpretation
    
    def generate_task_sequence(self, goal: str, scene_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate robotic task sequence using DeepSeek R1"""
        
        # Create structured prompt for DeepSeek
        scene_summary = self._create_scene_summary(scene_data)
        
        prompt = f"""You are a robotic task planner. Given a scene description and a goal, generate a detailed sequence of robotic tasks.

SCENE DESCRIPTION:
{scene_summary}

GOAL: {goal}

Generate a JSON array of robotic tasks. Each task should have:
- "type": One of ["move_to", "pick", "place", "move_arm", "grasp", "release"]
- "object_id": Target object identifier (or null for movement tasks)
- "position": [x, y, z] coordinates
- "description": Human-readable description
- "prerequisites": List of previous task IDs that must complete first
- "estimated_duration": Estimated time in seconds

Focus on practical, executable robot actions. Consider object positions, accessibility, and logical task ordering.

Return only the JSON array, no additional text."""

        if not self.api_key:
            print("Warning: No DeepSeek API key found. Generating fallback tasks.")
            return self._generate_fallback_tasks(goal, scene_data)
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "deepseek-reasoner",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Extract JSON from response
                try:
                    # Find JSON array in response
                    start_idx = content.find('[')
                    end_idx = content.rfind(']') + 1
                    if start_idx != -1 and end_idx != 0:
                        json_str = content[start_idx:end_idx]
                        tasks = json.loads(json_str)
                        return tasks
                except json.JSONDecodeError:
                    print("Failed to parse DeepSeek response as JSON")
                    
            else:
                print(f"DeepSeek API error: {response.status_code}")
                
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
        
        # Fallback if API fails
        return self._generate_fallback_tasks(goal, scene_data)
    
    def _create_scene_summary(self, scene_data: Dict[str, Any]) -> str:
        """Create human-readable scene summary"""
        scene_type = scene_data.get("scene_type", "unknown")
        objects = scene_data.get("objects", [])
        relationships = scene_data.get("relationships", [])
        
        summary = f"Scene Type: {scene_type}\n\n"
        summary += f"Objects ({len(objects)}):\n"
        
        for obj in objects:
            pos = obj["position"]
            summary += f"- {obj['class']} (ID: {obj['id']}) at position [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]\n"
        
        summary += f"\nSpatial Relationships ({len(relationships)}):\n"
        for rel in relationships[:10]:  # Limit to first 10 relationships
            summary += f"- {rel['source']} {rel['type']} {rel['target']}\n"
        
        return summary
    
    def _generate_fallback_tasks(self, goal: str, scene_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate simple fallback tasks when API is unavailable"""
        objects = scene_data.get("objects", [])
        
        if "clear" in goal.lower() and "table" in goal.lower():
            # Generate table clearing tasks
            tasks = []
            task_id = 0
            
            # Move to starting position
            tasks.append({
                "id": task_id,
                "type": "move_to",
                "object_id": None,
                "position": [0, 0, 0.5],
                "description": "Move robot to starting position",
                "prerequisites": [],
                "estimated_duration": 3
            })
            task_id += 1
            
            # Pick up each object
            for obj in objects:
                if obj["class"] != "scene":
                    # Move to object
                    tasks.append({
                        "id": task_id,
                        "type": "move_to",
                        "object_id": obj["id"],
                        "position": obj["position"],
                        "description": f"Move to {obj['class']}",
                        "prerequisites": [task_id - 1] if task_id > 0 else [],
                        "estimated_duration": 2
                    })
                    task_id += 1
                    
                    # Pick up object
                    tasks.append({
                        "id": task_id,
                        "type": "pick",
                        "object_id": obj["id"],
                        "position": obj["position"],
                        "description": f"Pick up {obj['class']}",
                        "prerequisites": [task_id - 1],
                        "estimated_duration": 3
                    })
                    task_id += 1
                    
                    # Place in storage area
                    storage_pos = [-0.5, 0.5, 0.1]  # Designated storage area
                    tasks.append({
                        "id": task_id,
                        "type": "place",
                        "object_id": obj["id"],
                        "position": storage_pos,
                        "description": f"Place {obj['class']} in storage area",
                        "prerequisites": [task_id - 1],
                        "estimated_duration": 3
                    })
                    task_id += 1
            
            return tasks
        
        # Default single task
        return [{
            "id": 0,
            "type": "move_to",
            "object_id": None,
            "position": [0, 0, 0.3],
            "description": f"Execute goal: {goal}",
            "prerequisites": [],
            "estimated_duration": 5
        }]
    
    def explain_scene(self, scene_data: Dict[str, Any]) -> str:
        """Generate natural language explanation of the scene"""
        
        scene_summary = self._create_scene_summary(scene_data)
        
        if not self.api_key:
            return f"Scene analysis (without LLM):\n{scene_summary}"
        
        prompt = f"""Provide a natural, human-readable description of this robotic scene:

{scene_summary}

Describe what you see, the spatial relationships between objects, and any notable patterns or arrangements. Keep it concise but informative."""

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "deepseek-reasoner",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            print(f"Error generating scene explanation: {e}")
        
        return f"Scene contains {len(scene_data.get('objects', []))} objects in a {scene_data.get('scene_type', 'unknown')} setting."
    
    def process_scene_graph_file(self, graph_file: str, goal: str = "Organize the scene", 
                                output_dir: str = "data/outputs/scene_graphs"):
        """Process scene graph file and generate complete interpretation"""
        
        if not os.path.exists(graph_file):
            print(f"Scene graph file not found: {graph_file}")
            return None
        
        # Load scene graph
        with open(graph_file, 'r') as f:
            graph_data = json.load(f)
        
        print("Analyzing scene graph...")
        scene_analysis = self.analyze_scene_graph(graph_data)
        
        print("Generating scene explanation...")
        scene_explanation = self.explain_scene(scene_analysis)
        
        print("Generating task sequence...")
        task_sequence = self.generate_task_sequence(goal, scene_analysis)
        
        # Combine results
        interpretation = {
            "scene_analysis": scene_analysis,
            "scene_explanation": scene_explanation,
            "goal": goal,
            "task_sequence": task_sequence,
            "metadata": {
                "num_tasks": len(task_sequence),
                "estimated_total_duration": sum(task.get("estimated_duration", 0) for task in task_sequence)
            }
        }
        
        # Save interpretation
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "llm_interpretation.json")
        with open(output_file, 'w') as f:
            json.dump(interpretation, f, indent=2)
        
        print(f"LLM interpretation saved to {output_file}")
        
        # Print summary
        print(f"\nScene Analysis Summary:")
        print(f"Scene type: {scene_analysis['scene_type']}")
        print(f"Objects: {scene_analysis['num_objects']}")
        print(f"Object classes: {scene_analysis['object_classes']}")
        print(f"Generated {len(task_sequence)} tasks")
        print(f"Estimated duration: {interpretation['metadata']['estimated_total_duration']} seconds")
        
        return interpretation

def main():
    print("DeepSeek R1 Scene Interpreter Module")
    
    interpreter = DeepSeekSceneInterpreter()
    
    # Check for scene graph
    graph_file = "data/outputs/scene_graphs/scene_graph.json"
    if os.path.exists(graph_file):
        print("Processing scene graph with DeepSeek R1...")
        
        # Test different goals
        goals = [
            "Clear the table",
            "Organize objects by type", 
            "Stack similar items together"
        ]
        
        for goal in goals:
            print(f"\nProcessing goal: {goal}")
            result = interpreter.process_scene_graph_file(graph_file, goal)
            if result:
                print("✓ Task sequence generated successfully")
    else:
        print("Scene graph not found. Run scene graph builder first.")
        print("Expected location: data/outputs/scene_graphs/scene_graph.json")

if __name__ == "__main__":
    main()