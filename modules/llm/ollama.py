"""
Ollama API client for local LLM integration.

This module provides a client for interacting with Ollama API
for text generation, chat, and embeddings.
"""
import os
import json
import requests
from typing import Dict, List, Any, Optional, Union

# Default Ollama URL
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"

class OllamaClient:
    """
    Client for Ollama API.
    """
    
    def __init__(
        self, 
        base_url: Optional[str] = None,
        default_model: str = DEFAULT_MODEL
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API URL (defaults to env var or localhost)
            default_model: Default model to use
        """
        self.base_url = base_url or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.default_model = default_model
        
    def is_available(self) -> bool:
        """
        Check if Ollama is available.
        
        Returns:
            bool: True if Ollama is available
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
            
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Returns:
            List of model information
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Optional[str]:
        """
        Generate text with Ollama.
        
        Args:
            prompt: The user prompt
            model: Model to use (defaults to default_model)
            system: Optional system prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text or None if failed
        """
        model = model or self.default_model
        
        try:
            data = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False
            }
            
            if system:
                data["system"] = system
                
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data
            )
            response.raise_for_status()
            return response.json().get("response", "")
            
        except Exception as e:
            print(f"Error generating text: {e}")
            return None
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Optional[str]:
        """
        Chat with Ollama.
        
        Args:
            messages: List of message objects with role and content
            model: Model to use (defaults to default_model)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response or None if failed
        """
        model = model or self.default_model
        
        try:
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False
            }
                
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=data
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return None
    
    def embed(
        self,
        text: str,
        model: Optional[str] = None
    ) -> Optional[List[float]]:
        """
        Generate embeddings with Ollama.
        
        Args:
            text: Text to embed
            model: Model to use (defaults to default_model)
            
        Returns:
            Embedding vector or None if failed
        """
        model = model or self.default_model
        
        try:
            data = {
                "model": model,
                "prompt": text
            }
                
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=data
            )
            response.raise_for_status()
            return response.json().get("embedding", [])
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

# Singleton instance
_client_instance = None

def get_client() -> OllamaClient:
    """
    Get or create the Ollama client instance.
    
    Returns:
        OllamaClient instance
    """
    global _client_instance
    
    if _client_instance is None:
        _client_instance = OllamaClient()
    
    return _client_instance

def is_available() -> bool:
    """
    Check if Ollama is available.
    
    Returns:
        bool: True if Ollama is available
    """
    try:
        client = get_client()
        return client.is_available()
    except Exception:
        return False