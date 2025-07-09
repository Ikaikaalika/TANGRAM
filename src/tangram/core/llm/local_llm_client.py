#!/usr/bin/env python3
"""
LLM Client for TANGRAM

This module provides LLM inference capabilities using multiple backends:
- Local inference via Ollama (DeepSeek, etc.)
- Google Gemini API
- Other cloud providers

Features:
- Multiple LLM backend support
- Automatic model downloading and setup for local models
- API key management for cloud providers
- Fallback mechanisms between providers

Author: TANGRAM Team
License: MIT
"""

import requests
import json
import time
import logging
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

from config import LLM_CONFIG
from src.tangram.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

# Import Google Gemini SDK if available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Gemini SDK not available. Install with: pip install google-generativeai")

# Import MLX for Apple Silicon optimization
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.utils import load_model
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX not available. Install with: pip install mlx mlx-lm")

# Import Hugging Face Transformers as fallback
try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers torch")


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


class GeminiLLMClient:
    """Google Gemini LLM client for cloud-based inference."""
    
    def __init__(self, 
                 api_key: str = None,
                 model_name: str = "gemini-1.5-flash",
                 timeout: int = 30):
        """
        Initialize Gemini LLM client.
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY environment variable)
            model_name: Gemini model name (gemini-1.5-flash, gemini-1.5-pro, etc.)
            timeout: Request timeout in seconds
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Gemini SDK not available. Install with: pip install google-generativeai")
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        self.model_name = model_name
        self.timeout = timeout
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        self.is_available = self._check_availability()
        
        if not self.is_available:
            logger.warning("Gemini API not available.")
    
    def _check_availability(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            # Simple test to verify API access
            response = self.model.generate_content("Test", request_options={"timeout": 5})
            return True
        except Exception as e:
            logger.debug(f"Gemini availability check failed: {e}")
            return False
    
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: str = None,
                         max_tokens: int = 2000,
                         temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate response using Gemini API.
        
        Args:
            prompt: User prompt
            system_prompt: System context prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.is_available:
            raise RuntimeError("Gemini API not available. Check your API key and internet connection.")
        
        try:
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Make request to Gemini
            start_time = time.time()
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                request_options={"timeout": self.timeout}
            )
            generation_time = time.time() - start_time
            
            # Extract response content
            content = response.text if response.text else ""
            
            # Calculate token usage (approximate)
            prompt_tokens = len(full_prompt.split())
            completion_tokens = len(content.split())
            
            return {
                "content": content,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "generation_time": generation_time,
                "source": "gemini",
                "finish_reason": response.candidates[0].finish_reason if response.candidates else "unknown"
            }
            
        except Exception as e:
            logger.error(f"Error with Gemini API: {e}")
            raise RuntimeError(f"Gemini API error: {e}")


class MLXLLMClient:
    """MLX-based LLM client optimized for Apple Silicon (M1/M2/M3 Macs)."""
    
    def __init__(self, 
                 model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
                 max_tokens: int = 2000,
                 temperature: float = 0.7,
                 cache_dir: str = None):
        """
        Initialize MLX LLM client.
        
        Args:
            model_name: HuggingFace model name or path (MLX-compatible)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            cache_dir: Directory to cache downloaded models
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX not available. Install with: pip install mlx mlx-lm")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/mlx")
        
        # Model and tokenizer will be loaded lazily
        self.model = None
        self.tokenizer = None
        self.is_available = False
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the MLX model and tokenizer."""
        try:
            logger.info(f"Loading MLX model: {self.model_name}")
            
            # Create cache directory
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Load model and tokenizer using MLX
            self.model, self.tokenizer = load(
                self.model_name,
                tokenizer_config={"trust_remote_code": True}
            )
            
            self.is_available = True
            logger.info(f"MLX model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load MLX model {self.model_name}: {e}")
            self.is_available = False
            
            # Try a smaller fallback model
            fallback_models = [
                "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "mlx-community/SmolLM2-1.7B-Instruct-4bit"
            ]
            
            for fallback in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback}")
                    self.model, self.tokenizer = load(
                        fallback,
                        tokenizer_config={"trust_remote_code": True}
                    )
                    self.model_name = fallback
                    self.is_available = True
                    logger.info(f"Fallback MLX model loaded: {fallback}")
                    break
                except Exception as fallback_e:
                    logger.debug(f"Fallback model {fallback} failed: {fallback_e}")
                    continue
    
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: str = None,
                         max_tokens: int = None,
                         temperature: float = None) -> Dict[str, Any]:
        """
        Generate response using MLX model.
        
        Args:
            prompt: User prompt
            system_prompt: System context prompt
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.is_available:
            raise RuntimeError("MLX model not available. Check model loading.")
        
        # Use provided parameters or defaults
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        try:
            # Format the prompt
            if system_prompt:
                # Use chat template if available
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                else:
                    formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = prompt
            
            # Generate response
            start_time = time.time()
            
            # MLX generate only accepts max_tokens parameter
            response = generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                verbose=False
            )
            
            generation_time = time.time() - start_time
            
            # Count tokens (approximate)
            input_tokens = len(self.tokenizer.encode(formatted_prompt))
            output_tokens = len(self.tokenizer.encode(response))
            
            return {
                "content": response,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                "generation_time": generation_time,
                "source": "mlx"
            }
            
        except Exception as e:
            logger.error(f"Error with MLX generation: {e}")
            raise RuntimeError(f"MLX generation error: {e}")


class HuggingFaceLLMClient:
    """Hugging Face Transformers LLM client with PyTorch backend."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-small",
                 device: str = "auto",
                 max_tokens: int = 2000,
                 temperature: float = 0.7):
        """
        Initialize Hugging Face LLM client.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (auto, cpu, cuda, mps)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Model and tokenizer will be loaded lazily
        self.model = None
        self.tokenizer = None
        self.is_available = False
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model and tokenizer."""
        try:
            logger.info(f"Loading HF model: {self.model_name} on {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
                device_map=self.device if self.device != "mps" else None
            )
            
            # Move to MPS if needed (device_map doesn't work with MPS)
            if self.device == "mps":
                self.model = self.model.to("mps")
            
            self.is_available = True
            logger.info(f"HF model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load HF model {self.model_name}: {e}")
            self.is_available = False
            
            # Try smaller fallback models
            fallback_models = [
                "microsoft/DialoGPT-small",
                "gpt2",
                "distilgpt2"
            ]
            
            for fallback in fallback_models:
                if fallback == self.model_name:
                    continue
                    
                try:
                    logger.info(f"Trying fallback model: {fallback}")
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        fallback,
                        torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
                        device_map=self.device if self.device != "mps" else None
                    )
                    
                    if self.device == "mps":
                        self.model = self.model.to("mps")
                    
                    self.model_name = fallback
                    self.is_available = True
                    logger.info(f"Fallback HF model loaded: {fallback}")
                    break
                    
                except Exception as fallback_e:
                    logger.debug(f"Fallback model {fallback} failed: {fallback_e}")
                    continue
    
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: str = None,
                         max_tokens: int = None,
                         temperature: float = None) -> Dict[str, Any]:
        """
        Generate response using Hugging Face model.
        
        Args:
            prompt: User prompt
            system_prompt: System context prompt
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.is_available:
            raise RuntimeError("Hugging Face model not available. Check model loading.")
        
        # Use provided parameters or defaults
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        try:
            # Format the prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = prompt
            
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generation_time = time.time() - start_time
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Count tokens
            input_tokens = inputs.input_ids.shape[1]
            output_tokens = outputs.shape[1] - input_tokens
            
            return {
                "content": response,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                "generation_time": generation_time,
                "source": "huggingface"
            }
            
        except Exception as e:
            logger.error(f"Error with HF generation: {e}")
            raise RuntimeError(f"HuggingFace generation error: {e}")


class UnifiedLLMClient:
    """Unified LLM client that can use multiple backends with fallback."""
    
    def __init__(self, 
                 prefer_local: bool = True,
                 gemini_api_key: str = None,
                 local_model: str = "deepseek-r1:latest",
                 gemini_model: str = "gemini-1.5-flash",
                 mlx_model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
                 hf_model: str = "microsoft/DialoGPT-small"):
        """
        Initialize unified LLM client with multiple fallback options.
        
        Args:
            prefer_local: Whether to prefer local models over cloud APIs
            gemini_api_key: Google API key for Gemini
            local_model: Local model name for Ollama
            gemini_model: Gemini model name
            mlx_model: MLX model name for Apple Silicon
            hf_model: Hugging Face model name
        """
        self.prefer_local = prefer_local
        self.clients = {}
        
        # Initialize Ollama client
        try:
            self.clients["ollama"] = LocalLLMClient(model_name=local_model)
        except Exception as e:
            logger.warning(f"Ollama LLM client initialization failed: {e}")
        
        # Initialize MLX client (optimized for Apple Silicon)
        if MLX_AVAILABLE:
            try:
                self.clients["mlx"] = MLXLLMClient(model_name=mlx_model)
            except Exception as e:
                logger.warning(f"MLX LLM client initialization failed: {e}")
        
        # Initialize Hugging Face client
        if TRANSFORMERS_AVAILABLE:
            try:
                self.clients["huggingface"] = HuggingFaceLLMClient(model_name=hf_model)
            except Exception as e:
                logger.warning(f"Hugging Face LLM client initialization failed: {e}")
        
        # Initialize Gemini client
        if gemini_api_key or os.getenv("GOOGLE_API_KEY"):
            try:
                self.clients["gemini"] = GeminiLLMClient(
                    api_key=gemini_api_key,
                    model_name=gemini_model
                )
            except Exception as e:
                logger.warning(f"Gemini LLM client initialization failed: {e}")
        
        # Determine available clients
        self.available_clients = [
            name for name, client in self.clients.items() 
            if client.is_available
        ]
        
        if not self.available_clients:
            raise RuntimeError(
                "No LLM clients available. Please set up one of the following:\n"
                "- Local Ollama with DeepSeek model\n"
                "- MLX with compatible model (Apple Silicon)\n"
                "- Hugging Face Transformers\n"
                "- Gemini API key"
            )
        
        logger.info(f"Available LLM clients: {self.available_clients}")
    
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: str = None,
                         max_tokens: int = 2000,
                         temperature: float = 0.7,
                         preferred_client: str = None) -> Dict[str, Any]:
        """
        Generate response using the best available client.
        
        Args:
            prompt: User prompt
            system_prompt: System context prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            preferred_client: Override client selection ("ollama", "mlx", "huggingface", "gemini")
            
        Returns:
            Dictionary with response and metadata
        """
        # Determine client priority
        if preferred_client and preferred_client in self.available_clients:
            client_priority = [preferred_client]
        else:
            if self.prefer_local:
                # Priority: Ollama → MLX → Hugging Face → Gemini
                client_priority = ["ollama", "mlx", "huggingface", "gemini"]
            else:
                # Priority: Gemini → MLX → Hugging Face → Ollama
                client_priority = ["gemini", "mlx", "huggingface", "ollama"]
        
        # Filter to available clients
        client_priority = [c for c in client_priority if c in self.available_clients]
        
        # Try clients in order
        last_error = None
        for client_name in client_priority:
            try:
                client = self.clients[client_name]
                logger.info(f"Using {client_name} LLM client")
                
                response = client.generate_response(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Add client info to response
                response["client_used"] = client_name
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"Client {client_name} failed: {e}")
                continue
        
        # All clients failed
        raise RuntimeError(f"All LLM clients failed. Last error: {last_error}")


class UnifiedDeepSeekInterpreter:
    """Unified DeepSeek scene interpreter that can use multiple LLM backends."""
    
    def __init__(self, 
                 llm_client: UnifiedLLMClient = None,
                 prefer_local: bool = True,
                 gemini_api_key: str = None):
        """Initialize with unified LLM client."""
        self.llm_client = llm_client or UnifiedLLMClient(
            prefer_local=prefer_local,
            gemini_api_key=gemini_api_key
        )
        self.scene_context = {}
    
    def analyze_scene_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scene graph using available LLM backend."""
        
        # Extract scene information (same as LocalDeepSeekInterpreter)
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
        
        # Create prompt for LLM
        scene_prompt = self._create_scene_analysis_prompt(objects, relationships)
        system_prompt = """You are an AI assistant specialized in robotic scene understanding and task planning.
Your role is to analyze 3D scenes with objects and their relationships, then generate practical robotic manipulation tasks.
Respond with structured JSON containing scene analysis and actionable robot tasks."""
        
        # Get response from unified LLM client
        response = self.llm_client.generate_response(
            prompt=scene_prompt,
            system_prompt=system_prompt,
            max_tokens=2000,
            temperature=0.7
        )
        
        # Parse response
        return self._parse_llm_response(response["content"], objects, relationships, response)
    
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
    
    def _parse_llm_response(self, response_text: str, objects: List[Dict], relationships: List[Dict], full_response: Dict) -> Dict[str, Any]:
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
                "model_used": full_response.get("model", "unknown"),
                "client_used": full_response.get("client_used", "unknown"),
                "source": full_response.get("source", "unknown"),
                "generation_time": full_response.get("generation_time", 0),
                "usage": full_response.get("usage", {})
            })
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                "scene_analysis": f"Error parsing response: {e}",
                "task_sequence": [],
                "objects": objects,
                "relationships": relationships,
                "model_used": full_response.get("model", "unknown"),
                "client_used": full_response.get("client_used", "unknown"),
                "error": str(e)
            }