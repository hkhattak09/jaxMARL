"""Configuration module for JAX MARL training.

To configure training, edit the values directly in assembly_cfg.py.
"""

from .assembly_cfg import (
    # Main config class
    AssemblyTrainConfig,
    
    # Get the config (reads from class defaults)
    get_config,
    
    # Path utilities
    get_shape_file_path,
    get_checkpoint_dir,
    get_log_dir,
    get_eval_dir,
    
    # Config conversion (used internally by training)
    config_to_maddpg_config,
    config_to_assembly_params,
)

from .preprocess_shapes import (
    process_image,
    process_image_folder,
)

from .llm_cfg import (
    LLMConfig,
    create_llm_config,
    get_api_key,
    get_task_description,
    get_environment_description,
    get_robot_api_description,
)

__all__ = [
    # Main config
    "AssemblyTrainConfig",
    "get_config",
    
    # Paths
    "get_shape_file_path",
    "get_checkpoint_dir",
    "get_log_dir",
    "get_eval_dir",
    
    # Conversion (internal use)
    "config_to_maddpg_config",
    "config_to_assembly_params",
    
    # Preprocessing
    "process_image",
    "process_image_folder",
    
    # LLM
    "LLMConfig",
    "create_llm_config",
    "get_api_key",
    "get_task_description",
    "get_environment_description",
    "get_robot_api_description",
]
