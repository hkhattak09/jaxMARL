"""LLM module for GPT-based reward and policy generation.

This module provides tools for:
- Calling GPT API to generate reward/policy functions
- Parsing LLM responses to extract code
- Validating and loading generated functions

The LLM is called BEFORE training (not in the JAX training loop).
Generated functions are saved and loaded for use during training.
"""

from .gpt_client import GPTClient, call_gpt, call_gpt_sync
from .prompts import (
    build_generation_prompt,
    GENERATION_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)
from .parser import (
    parse_code_blocks,
    parse_json_block,
    extract_functions,
    validate_generated_code,
    parse_llm_response,
    get_function_source,
)
from .generator import (
    generate_reward_and_policy,
    GeneratedCode,
    save_generated_code,
    load_generated_code,
    load_functions_from_file,
)

__all__ = [
    # GPT Client
    "GPTClient",
    "call_gpt",
    "call_gpt_sync",
    
    # Prompts
    "build_generation_prompt",
    "GENERATION_PROMPT_TEMPLATE",
    "SYSTEM_PROMPT",
    
    # Parser
    "parse_code_blocks",
    "parse_json_block",
    "extract_functions",
    "validate_generated_code",
    "parse_llm_response",
    "get_function_source",
    
    # Generator
    "generate_reward_and_policy",
    "GeneratedCode",
    "save_generated_code",
    "load_generated_code",
    "load_functions_from_file",
]
