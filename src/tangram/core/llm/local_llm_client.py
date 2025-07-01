#!/usr/bin/env python3
"""
Local LLM Client for Thunder Compute

This module provides local LLM inference capabilities using Ollama or similar
local inference servers. Designed to run DeepSeek models locally on Thunder
Compute instances instead of using external APIs.

Features:
- Ollama integration for local model hosting
- Automatic model downloading and setup
- Compatible with existing DeepSeek scene interpretation
- Fallback to API when local inference unavailable

Author: TANGRAM Team
License: MIT
"""

import requests
import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from config import LLM_CONFIG
from src.tangram.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class LocalLLMClient:
    """Local LLM client using Ollama for Thunder Compute deployment."""
    
    def __init__(self, 
                 model_name: str = "deepseek-r1:latest",
                 ollama_host: str = "localhost",
                 ollama_port: int = 11434,
                 fallback_to_api: bool = False):
        """
        Initialize local LLM client.
        
        Args:
            model_name: Ollama model name (e.g., 'deepseek-r1:latest')
            ollama_host: Host where Ollama is running
            ollama_port: Port for Ollama API
            fallback_to_api: Whether to fallback to external API if local fails
        """
        self.model_name = model_name
        self.base_url = f"http://{ollama_host}:{ollama_port}"
        self.fallback_to_api = fallback_to_api
        self.is_available = self._check_availability()
        
        if not self.is_available:
            logger.warning("Local LLM not available. Will use fallback if enabled.")
    
    def _check_availability(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if self.model_name not in model_names:
                logger.info(f"Model {self.model_name} not found. Available models: {model_names}")
                # Try to pull the model
                return self._pull_model()
            
            return True
            
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            return False
    
    def _pull_model(self) -> bool:
        """Download the specified model using Ollama."""
        try:
            logger.info(f"Pulling model {self.model_name}...")
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                stream=True,
                timeout=600  # 10 minutes for model download
            )
            
            if response.status_code == 200:
                # Stream the download progress
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if "status" in data:
                            logger.info(f"Model pull: {data['status']}")
                
                logger.info(f"Successfully pulled model {self.model_name}")
                return True
            else:
                logger.error(f"Failed to pull model: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: str = None,
                         max_tokens: int = 2000,
                         temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate response using local LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System context prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.is_available:
            raise RuntimeError(
                "Local LLM not available. External APIs are disabled. "
                "Please ensure Ollama is running and DeepSeek model is installed. "
                "Run: docs/scripts/setup_deepseek_thunder.sh"
            )
        
        try:
            # Prepare the prompt
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Make request to Ollama
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=120  # 2 minutes for generation
            )
            
            if response.status_code == 200:
                result = response.json()
                generation_time = time.time() - start_time
                
                return {
                    "content": result["message"]["content"],
                    "model": self.model_name,
                    "usage": {
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                    },
                    "generation_time": generation_time,
                    "source": "local"
                }
            else:
                logger.error(f"Local LLM request failed: {response.text}")
                raise RuntimeError(f"Local LLM request failed: {response.text}. External APIs disabled.")
                    
        except Exception as e:
            logger.error(f"Error with local LLM: {e}")
            raise RuntimeError(f"Local LLM error: {e}. External APIs disabled.")
    


class LocalDeepSeekInterpreter:
    """Local version of DeepSeek scene interpreter using local LLM."""
    
    def __init__(self, llm_client: LocalLLMClient = None):
        """Initialize with local LLM client."""
        self.llm_client = llm_client or LocalLLMClient()
        self.scene_context = {}
    
    def analyze_scene_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scene graph using local LLM."""
        
        # Extract scene information (same as original)
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        objects = []
        for node in nodes:
            node_props = node.get("properties", {})
            if node_props.get("type") != "scene_context":
                objects.append({
                    "id": node["id"],
                    "class": node_props.get("class_name", "unknown"),
                    "position": node_props.get("position_3d", [0, 0, 0]),
                    "track_id": node_props.get("track_id", -1)
                })
        
        relationships = []
        for edge in edges:
            edge_props = edge.get("properties", {})
            relations = edge_props.get("relations", [edge_props.get("relation", "unknown")])
            
            for relation in relations:
                relationships.append({
                    "source": edge["source"],
                    "target": edge["target"],
                    "type": relation,
                    "confidence": edge_props.get("confidence", 0.5)
                })
        
        # Create prompt for local LLM
        scene_prompt = self._create_scene_analysis_prompt(objects, relationships)
        system_prompt = """You are an AI assistant specialized in robotic scene understanding and task planning.
Your role is to analyze 3D scenes with objects and their relationships, then generate practical robotic manipulation tasks.
Respond with structured JSON containing scene analysis and actionable robot tasks."""
        
        # Get response from local LLM
        response = self.llm_client.generate_response(
            prompt=scene_prompt,
            system_prompt=system_prompt,
            max_tokens=2000,
            temperature=0.7
        )
        
        # Parse response (same logic as original)
        return self._parse_llm_response(response["content"], objects, relationships)
    
    def _create_scene_analysis_prompt(self, objects: List[Dict], relationships: List[Dict]) -> str:
        """Create detailed prompt for scene analysis."""
        prompt = """Analyze this robotic scene and generate manipulation tasks:

OBJECTS:
"""
        for obj in objects:
            prompt += f"- {obj['class']} (ID: {obj['id']}) at position {obj['position']}\n"
        
        prompt += "\nRELATIONSHIPS:\n"
        for rel in relationships:
            prompt += f"- Object {rel['source']} {rel['type']} Object {rel['target']} (confidence: {rel['confidence']})\n"
        
        prompt += """
Generate a JSON response with:
1. Scene analysis (what's happening, object arrangements)
2. Practical robot tasks (pick, place, move operations)
3. Task sequence with specific object IDs and target positions
4. Safety considerations

Format as valid JSON with keys: scene_analysis, task_sequence, safety_notes
"""
        return prompt
    
    def _parse_llm_response(self, response_text: str, objects: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                # Fallback: create structured response from text
                response_data = {
                    "scene_analysis": response_text[:500],
                    "task_sequence": [],
                    "safety_notes": "Unable to parse structured response"
                }
            
            # Ensure required fields exist
            if "task_sequence" not in response_data:
                response_data["task_sequence"] = []
            
            # Add metadata
            response_data.update({
                "objects": objects,
                "relationships": relationships,
                "model_used": self.llm_client.model_name,
                "source": "local_llm"
            })
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                "scene_analysis": f"Error parsing response: {e}",
                "task_sequence": [],
                "objects": objects,
                "relationships": relationships,
                "error": str(e)
            }