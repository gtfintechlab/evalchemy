"""
Ollama model integration for Evalchemy.

This module provides a model wrapper that enables using Ollama models
with the Evalchemy evaluation framework.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import requests

import lm_eval.api.registry
from lm_eval.api.model import LM


logger = logging.getLogger(__name__)


@lm_eval.api.registry.register_model("ollama")
class OllamaLM(LM):
    """Ollama model wrapper for Evalchemy."""
    
    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        base_url: str = None,
        temperature: float = 0.0,
        timeout: int = 120,
        **kwargs
    ):
        """Initialize Ollama model.
        
        Args:
            model: Name of the Ollama model to use
            base_url: Base URL for Ollama API (defaults to OLLAMA_BASE_URL env var or localhost)
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            **kwargs: Additional arguments (ignored)
        """
        super().__init__()
        
        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.temperature = temperature
        self.timeout = timeout
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            if self.model not in model_names:
                logger.warning(f"Model {self.model} not found in Ollama. Available models: {model_names}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama at {self.base_url}: {e}")
    
    @property
    def rank(self):
        """Process rank (always 0 for single-process Ollama)."""
        return 0
    
    @property
    def world_size(self):
        """World size (always 1 for single-process Ollama)."""
        return 1
    
    @property
    def eot_token_id(self):
        """End of text token ID (not applicable for Ollama)."""
        return None
    
    @property
    def max_length(self):
        """Maximum context length."""
        return 8192  # Most Ollama models support at least this
    
    @property
    def max_gen_toks(self):
        """Maximum generation tokens."""
        return 4096
    
    @property
    def batch_size(self):
        """Ollama processes one at a time."""
        return 1
    
    @property
    def device(self):
        """Device string."""
        return "cuda"  # Ollama handles device management
    
    def tok_encode(self, string: str) -> List[int]:
        """Tokenize string (not implemented for Ollama)."""
        # Ollama doesn't expose tokenization, return dummy tokens
        return list(range(len(string.split())))
    
    def tok_decode(self, tokens: List[int]) -> str:
        """Decode tokens (not implemented for Ollama)."""
        # Return dummy string
        return " ".join(["token"] * len(tokens))
    
    def generate_until(self, request_list) -> List[str]:
        """Generate completions for a list of prompts.
        
        Args:
            request_list: List of Instance objects or tuples
            
        Returns:
            List of generated strings
        """
        results = []
        
        # Handle both Instance objects and raw tuples
        if not isinstance(request_list, list):
            request_list = [request_list]
        
        for request in tqdm(request_list, desc="Running generate_until requests"):
            # Extract prompt and gen_kwargs from the request
            if hasattr(request, 'args'):
                # Instance object
                args = request.args
                if isinstance(args, tuple) and len(args) >= 2:
                    prompt, gen_kwargs = args[0], args[1]
                else:
                    prompt = args[0] if isinstance(args, tuple) else args
                    gen_kwargs = {}
            elif isinstance(request, tuple):
                # Raw tuple
                prompt, gen_kwargs = request[0], request[1] if len(request) > 1 else {}
            else:
                prompt = str(request)
                gen_kwargs = {}
            
            # Extract generation parameters
            max_tokens = gen_kwargs.get("max_new_tokens", 256)
            temperature = gen_kwargs.get("temperature", self.temperature)
            do_sample = gen_kwargs.get("do_sample", temperature > 0)
            
            # Set temperature to 0 if not sampling
            if not do_sample:
                temperature = 0.0
            
            try:
                # Call Ollama API
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        }
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                generated_text = result.get("response", "")
                results.append(generated_text)
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                results.append("")  # Return empty string on error
        
        return results
    
    def loglikelihood(self, request_list):
        """Compute log-likelihood of completions (not implemented)."""
        raise NotImplementedError("Ollama doesn't support log-likelihood computation")
    
    def loglikelihood_rolling(self, request_list):
        """Compute rolling log-likelihood (not implemented)."""
        raise NotImplementedError("Ollama doesn't support log-likelihood computation")