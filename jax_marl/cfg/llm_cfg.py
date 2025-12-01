"""LLM Configuration for JAX MARL.

This module provides configuration for GPT-based reward and prior policy
generation. The LLM is used BEFORE training to generate:
1. A reward function for the task
2. A prior policy function for regularization

The generated code is then used during JAX training.

Usage:
    from cfg import LLMConfig, create_llm_config
    
    config = create_llm_config(api_key="your-openai-key")
    
Environment Variable:
    Set OPENAI_API_KEY to avoid passing key in code.
"""

from typing import Optional, NamedTuple
import os


class LLMConfig(NamedTuple):
    """Configuration for GPT-based code generation.
    
    Attributes:
        # API Settings
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        api_base: API base URL (default: OpenAI)
        model: Model to use for generation
        
        # Generation Settings
        temperature: Sampling temperature (0 = deterministic)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        max_retries: Number of retries on failure
        
        # Task Settings
        task_name: Name of the task ("assembly")
        
        # Output Settings
        output_dir: Directory to save generated code
        save_responses: Whether to save raw LLM responses
    """
    # API Settings
    api_key: Optional[str] = None  # Falls back to OPENAI_API_KEY env var
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"  # or "gpt-4-turbo", "o1-preview"
    
    # Generation Settings
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 3
    
    # Task Settings
    task_name: str = "assembly"
    
    # Output Settings
    output_dir: Optional[str] = None
    save_responses: bool = True


def create_llm_config(
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    **overrides,
) -> LLMConfig:
    """Create LLM configuration with API key handling.
    
    Args:
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        model: Model name to use
        **overrides: Additional config overrides
        
    Returns:
        LLMConfig instance
        
    Raises:
        ValueError: If no API key is provided or found in environment
    """
    # Get API key from argument or environment
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key is None:
        raise ValueError(
            "OpenAI API key required. Either:\n"
            "  1. Pass api_key argument\n"
            "  2. Set OPENAI_API_KEY environment variable"
        )
    
    return LLMConfig(
        api_key=api_key,
        model=model,
        **overrides,
    )


def get_api_key(config: LLMConfig) -> str:
    """Get the API key from config or environment.
    
    Args:
        config: LLM configuration
        
    Returns:
        API key string
        
    Raises:
        ValueError: If no API key available
    """
    if config.api_key is not None:
        return config.api_key
    
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key is not None:
        return env_key
    
    raise ValueError("No OpenAI API key found in config or environment")


# ============================================================================
# Task Descriptions (for prompt construction)
# ============================================================================

TASK_DESCRIPTIONS = {
    "assembly": """The robot swarm needs to assemble into a specific shape that aligns with the shape of the target region. In such a shape, robots should avoid collisions, synchronize movements with neighbors to reduce oscillations, and attempt to explore unoccupied cells.""",
}

ENVIRONMENT_DESCRIPTIONS = {
    "assembly": """There are a group of robots and a target region on a 2D plane. The target region is an arbitrarily shaped, simply connected area. It is discretized into grids, divided by lines parallel to the x and y axes, meaning the target region is composed of a collection of cells. Each robot has a collision radius and a sensing radius. A collision occurs when the distance between two robots is less than twice the collision radius. Additionally, a cell is considered occupied by a robot if the distance between the robot and the cell is less than the collision radius.""",
}

ROBOT_API_DESCRIPTIONS = {
    "assembly": """
def get_neighbor_id_list(id):
    '''Get the IDs of neighboring robots within sensing radius.
    Returns: list of neighbor robot IDs'''

def get_robot_position_and_velocity(id):
    '''Get position and velocity for a robot.
    Input: robot ID (int)
    Returns: (position, velocity) as numpy arrays of shape (2,)'''

def get_unoccupied_cells_position(id):
    '''Get positions of sensed unoccupied target cells.
    Input: robot ID (int)
    Returns: numpy array of shape (n_cells, 2)'''

def get_target_cell_position(id):
    '''Get the nearest target cell position for the robot.
    Input: robot ID (int)
    Returns: numpy array of shape (2,)'''

def is_within_target_region(id):
    '''Check if robot is within the target shape.
    Input: robot ID (int)
    Returns: bool'''
""",
}


def get_task_description(task_name: str) -> str:
    """Get the task description for a given task."""
    return TASK_DESCRIPTIONS.get(task_name, TASK_DESCRIPTIONS["assembly"])


def get_environment_description(task_name: str) -> str:
    """Get the environment description for a given task."""
    return ENVIRONMENT_DESCRIPTIONS.get(task_name, ENVIRONMENT_DESCRIPTIONS["assembly"])


def get_robot_api_description(task_name: str) -> str:
    """Get the robot API description for a given task."""
    return ROBOT_API_DESCRIPTIONS.get(task_name, ROBOT_API_DESCRIPTIONS["assembly"])
