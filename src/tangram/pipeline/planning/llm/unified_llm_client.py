"""
Unified LLM Client for TANGRAM
Supports multiple LLM backends: Hugging Face, Ollama, and Google Gemini
"""

import os
import json
import logging
import getpass
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from config import LLM_CONFIG


class UnifiedLLMClient:
    """
    Unified LLM client that automatically selects the best available backend
    Priority: Hugging Face > Ollama > Google Gemini
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.config = LLM_CONFIG
        
        # Initialize the best available backend
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the best available LLM backend"""
        self.logger.info("Initializing LLM backend...")
        
        if self.config["provider"] == "auto":
            # Auto-select best available backend
            if self.config["prefer_local"]:
                if self._try_huggingface():
                    return
                if self._try_ollama():
                    return
            if self._try_gemini():
                return
            
            # If all else fails, try non-preferred local options
            if not self.config["prefer_local"]:
                if self._try_huggingface():
                    return
                if self._try_ollama():
                    return
        
        elif self.config["provider"] == "huggingface":
            if not self._try_huggingface():
                raise RuntimeError("Hugging Face backend not available")
        
        elif self.config["provider"] == "ollama":
            if not self._try_ollama():
                raise RuntimeError("Ollama backend not available")
        
        elif self.config["provider"] == "gemini":
            if not self._try_gemini():
                raise RuntimeError("Google Gemini backend not available")
        
        else:
            raise ValueError(f"Unknown provider: {self.config['provider']}")
        
        if self.backend is None:
            raise RuntimeError("No LLM backend available")
        
        self.logger.info(f"✅ LLM backend initialized: {self.backend}")
    
    def _try_huggingface(self) -> bool:
        """Try to initialize Hugging Face backend"""
        if not HUGGINGFACE_AVAILABLE or not self.config["huggingface"]["enabled"]:
            self.logger.debug("Hugging Face not available or disabled")
            return False
        
        try:
            self.logger.info("Attempting to load Hugging Face model...")
            hf_config = self.config["huggingface"]
            
            # Load tokenizer and model
            model_name = hf_config["model"]
            device = hf_config["device"]
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=getattr(torch, hf_config["torch_dtype"]),
                device_map=device if device != "mps" else None
            )
            
            if device == "mps":
                self.model = self.model.to(device)
            
            self.backend = "huggingface"
            self.logger.info(f"✅ Hugging Face model loaded: {model_name}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Hugging Face: {e}")
            return False
    
    def _try_ollama(self) -> bool:
        """Try to initialize Ollama backend"""
        if not REQUESTS_AVAILABLE or not self.config["ollama"]["enabled"]:
            self.logger.debug("Ollama not available or disabled")
            return False
        
        try:
            ollama_config = self.config["ollama"]
            host = ollama_config["host"]
            port = ollama_config["port"]
            
            # Test Ollama connection
            response = requests.get(f"http://{host}:{port}/api/tags", timeout=5)
            if response.status_code == 200:
                self.backend = "ollama"
                self.logger.info(f"✅ Ollama backend available at {host}:{port}")
                return True
            else:
                self.logger.warning(f"Ollama not responding: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Failed to connect to Ollama: {e}")
            return False
    
    def _try_gemini(self) -> bool:
        """Try to initialize Google Gemini backend"""
        if not GEMINI_AVAILABLE or not self.config["gemini"]["enabled"]:
            self.logger.debug("Google Gemini not available or disabled")
            return False
        
        try:
            gemini_config = self.config["gemini"]
            api_key_env = gemini_config["api_key_env"]
            
            # Try to get API key from environment
            api_key = os.getenv(api_key_env)
            
            # If no API key and prompting is enabled, ask user
            if not api_key and self.config.get("prompt_for_api_key", False):
                print(f"\n🔑 Google Gemini API key required")
                print(f"Get your API key from: https://makersuite.google.com/app/apikey")
                api_key = getpass.getpass("Enter your Google Gemini API key: ")
                
                if api_key:
                    # Optionally save to environment for this session
                    os.environ[api_key_env] = api_key
            
            if not api_key:
                self.logger.warning("No Google Gemini API key available")
                return False
            
            # Configure and test Gemini
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(gemini_config["model"])
            
            # Test with a simple prompt
            response = self.model.generate_content("Hello", 
                                                  generation_config=genai.types.GenerationConfig(
                                                      max_output_tokens=10))
            
            if response.text:
                self.backend = "gemini"
                self.logger.info(f"✅ Google Gemini backend initialized: {gemini_config['model']}")
                return True
            else:
                self.logger.warning("Gemini test failed")
                return False
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize Google Gemini: {e}")
            return False
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using the active backend"""
        if self.backend is None:
            raise RuntimeError("No LLM backend available")
        
        try:
            if self.backend == "huggingface":
                return self._generate_huggingface(prompt, **kwargs)
            elif self.backend == "ollama":
                return self._generate_ollama(prompt, **kwargs)
            elif self.backend == "gemini":
                return self._generate_gemini(prompt, **kwargs)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
                
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            raise
    
    def _generate_huggingface(self, prompt: str, **kwargs) -> str:
        """Generate response using Hugging Face"""
        hf_config = self.config["huggingface"]
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hf_config["device"] == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=hf_config["max_new_tokens"],
                temperature=hf_config["temperature"],
                do_sample=hf_config["do_sample"],
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def _generate_ollama(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama"""
        ollama_config = self.config["ollama"]
        
        url = f"http://{ollama_config['host']}:{ollama_config['port']}/api/generate"
        payload = {
            "model": ollama_config["model"],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config["temperature"],
                "num_predict": self.config["max_tokens"]
            }
        }
        
        response = requests.post(url, json=payload, timeout=ollama_config["timeout"])
        response.raise_for_status()
        
        return response.json()["response"]
    
    def _generate_gemini(self, prompt: str, **kwargs) -> str:
        """Generate response using Google Gemini"""
        gemini_config = self.config["gemini"]
        
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"]
        )
        
        response = self.model.generate_content(prompt, generation_config=generation_config)
        return response.text
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the active backend"""
        if self.backend is None:
            return {"backend": None, "status": "not_initialized"}
        
        info = {
            "backend": self.backend,
            "status": "active"
        }
        
        if self.backend == "huggingface":
            info["model"] = self.config["huggingface"]["model"]
            info["device"] = self.config["huggingface"]["device"]
        elif self.backend == "ollama":
            info["model"] = self.config["ollama"]["model"]
            info["host"] = f"{self.config['ollama']['host']}:{self.config['ollama']['port']}"
        elif self.backend == "gemini":
            info["model"] = self.config["gemini"]["model"]
            info["api_key_set"] = bool(os.getenv(self.config["gemini"]["api_key_env"]))
        
        return info


# Factory function for backward compatibility
def create_unified_llm_client() -> UnifiedLLMClient:
    """Create and return a unified LLM client instance"""
    return UnifiedLLMClient()