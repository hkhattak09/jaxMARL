"""GPT API Client for code generation.

Simple, synchronous GPT client optimized for one-time code generation
(not for use in training loops).
"""

import os
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class GPTResponse:
    """Container for GPT API response."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str


class GPTClient:
    """Simple GPT client for code generation.
    
    This is a synchronous client designed for one-time code generation,
    not for use in training loops.
    
    Example:
        client = GPTClient(api_key="sk-...")
        response = client.chat("Generate a reward function for...")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        api_base: str = "https://api.openai.com/v1",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """Initialize GPT client.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model to use (default: gpt-4o)
            api_base: API base URL
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
        """
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI package required. Install with: pip install openai"
            )
        
        # Get API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Either:\n"
                "  1. Pass api_key argument\n"
                "  2. Set OPENAI_API_KEY environment variable"
            )
        
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize client
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=api_base,
            timeout=timeout,
        )
    
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> GPTResponse:
        """Send a chat message to GPT.
        
        Args:
            prompt: User message/prompt
            system_prompt: Optional system message
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum tokens in response
            
        Returns:
            GPTResponse with content and metadata
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                return GPTResponse(
                    content=response.choices[0].message.content,
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    finish_reason=response.choices[0].finish_reason,
                )
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
        
        raise RuntimeError(f"GPT request failed after {self.max_retries} attempts: {last_error}")


def call_gpt(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Convenience function to call GPT and return content.
    
    Args:
        prompt: User message/prompt
        api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
        model: Model to use
        system_prompt: Optional system message
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Returns:
        Response content as string
    """
    client = GPTClient(api_key=api_key, model=model)
    response = client.chat(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.content


# Alias for clarity
call_gpt_sync = call_gpt
