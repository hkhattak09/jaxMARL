"""Prompt templates for LLM-based code generation.

Contains the main prompt template for generating reward and policy functions,
along with utilities for building complete prompts.
"""

from typing import Optional
import sys
from pathlib import Path

# Add cfg to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "cfg"))

try:
    from cfg.llm_cfg import (
        get_task_description,
        get_environment_description, 
        get_robot_api_description,
    )
except ImportError:
    # Fallback if cfg not available
    def get_task_description(task_name: str) -> str:
        return ""
    def get_environment_description(task_name: str) -> str:
        return ""
    def get_robot_api_description(task_name: str) -> str:
        return ""


# ============================================================================
# Main Generation Prompt Template
# ============================================================================

GENERATION_PROMPT_TEMPLATE: str = """
## Environment Description:
{env_des}

## Task Description:
{task_des}

## Available Robot APIs:
The following APIs are already implemented and can be called directly:
```python
{api_des}
```

## Your Role:
You are a task analysis assistant for multi-agent reinforcement learning. Analyze the task and provide two functions:

1. **Reward Function**: Returns a reward for each robot based on task completion.
2. **Prior Policy Function**: A rule-based policy that provides basic behaviors (collision avoidance, target seeking, etc.)

## Reasoning Steps:
Before writing code, reason through:
1. What constraints must be satisfied to complete the task?
2. What are the basic vs complex constraints?
3. Which constraints should be in the reward function?
4. What basic capabilities should robots have in the prior policy?
{auxiliary_cot}

## Output Format:
First provide your reasoning, then output the code in this exact format:

### Reasoning:
(Your step-by-step analysis here)

### Code:
```python
import numpy as np

def compute_reward(
    positions,
    velocities, 
    grid_centers,
    l_cell,
    in_target,
    is_colliding,
    exploration,
    reward_entering=1.0,
    penalty_collision=-0.5,
    reward_exploration=0.1,
):
    '''
    Compute reward for each robot in the assembly task.
    
    This function defines the reward logic. During JAX training, the actual
    reward computation uses JAX arrays, but the LOGIC you define here will
    be used to guide reward shaping.
    
    Args:
        positions: Robot positions, shape (n_agents, 2)
        velocities: Robot velocities, shape (n_agents, 2)
        grid_centers: Target grid cell centers, shape (n_grid, 2)
        l_cell: Grid cell size (float)
        in_target: Boolean array of which robots are in target, shape (n_agents,)
        is_colliding: Boolean array of which robots are colliding, shape (n_agents,)
        exploration: Float array of exploration reward per robot, shape (n_agents,)
        reward_entering: Reward for entering target region (float)
        penalty_collision: Penalty for collision (float, typically negative)
        reward_exploration: Reward scale for exploration (float)
        
    Returns:
        rewards: Reward for each robot, shape (n_agents,)
    '''
    n_agents = positions.shape[0]
    
    # Example implementation:
    # Task complete if in target AND not colliding AND exploring
    task_complete = in_target & (~is_colliding) & (exploration > 0)
    
    rewards = np.zeros(n_agents)
    rewards = np.where(task_complete, reward_entering, rewards)
    rewards = rewards + is_colliding.astype(np.float32) * penalty_collision
    
    return rewards

def compute_prior_policy(
    positions,
    velocities,
    grid_centers,
    grid_mask,
    l_cell,
    r_avoid,
    d_sen,
    attraction_strength=2.0,
    repulsion_strength=1.0,
    sync_strength=2.0,
):
    '''
    Compute rule-based prior policy for all agents.
    
    This function provides a hand-crafted baseline policy that the learned
    policy can use for regularization. The prior policy helps with:
    1. Collision avoidance (repulsion)
    2. Target seeking (attraction)
    3. Velocity synchronization with neighbors (coordination)
    
    Args:
        positions: All robot positions, shape (n_agents, 2)
        velocities: All robot velocities, shape (n_agents, 2)  
        grid_centers: Target grid cell centers, shape (n_grid, 2)
        grid_mask: Boolean mask for valid grid cells, shape (n_grid,)
        l_cell: Grid cell size (float)
        r_avoid: Collision avoidance radius (float)
        d_sen: Sensing radius for neighbors (float)
        attraction_strength: Strength of attraction to target (float)
        repulsion_strength: Strength of repulsion from neighbors (float)
        sync_strength: Strength of velocity synchronization (float)
        
    Returns:
        actions: Acceleration commands for all robots, shape (n_agents, 2),
                 each action clipped to [-1, 1]
    '''
    n_agents = positions.shape[0]
    actions = np.zeros((n_agents, 2))
    
    for i in range(n_agents):
        pos = positions[i]
        vel = velocities[i]
        
        # 1. Attraction to nearest unoccupied target
        rel_to_grid = grid_centers - pos
        dist_to_grid = np.linalg.norm(rel_to_grid, axis=-1)
        dist_to_grid = np.where(grid_mask, dist_to_grid, 1e10)
        
        nearest_idx = np.argmin(dist_to_grid)
        nearest_dist = dist_to_grid[nearest_idx]
        target_pos = grid_centers[nearest_idx]
        
        direction_to_target = target_pos - pos
        dist_to_target = max(nearest_dist, 1e-8)
        attraction_force = attraction_strength * direction_to_target / dist_to_target
        
        # Zero attraction if already in target
        threshold = np.sqrt(2.0) * l_cell / 2.0
        if nearest_dist < threshold:
            attraction_force = np.zeros(2)
        
        # 2. Repulsion from nearby agents
        repulsion_force = np.zeros(2)
        for j in range(n_agents):
            if i == j:
                continue
            rel = positions[j] - pos
            dist = np.linalg.norm(rel)
            if dist < r_avoid and dist > 1e-8:
                repulsion_dir = -rel / dist
                repulsion_mag = repulsion_strength * (r_avoid / dist - 1.0)
                repulsion_force += repulsion_mag * repulsion_dir
        
        # 3. Velocity synchronization with neighbors
        sync_force = np.zeros(2)
        neighbor_count = 0
        avg_velocity = np.zeros(2)
        for j in range(n_agents):
            if i == j:
                continue
            dist = np.linalg.norm(positions[j] - pos)
            if dist < d_sen:
                avg_velocity += velocities[j]
                neighbor_count += 1
        
        if neighbor_count > 0:
            avg_velocity /= neighbor_count
            sync_force = sync_strength * (avg_velocity - vel)
        
        # Combine forces
        total_force = attraction_force + repulsion_force + sync_force
        actions[i] = np.clip(total_force, -1.0, 1.0)
    
    return actions
```

```json
{{
    "key_task_sub_goals": ["goal1", "goal2", ...],
    "basic_capabilities": ["capability1", "capability2", ...]
}}
```

## Important Notes:
- Output must strictly follow the format above
- Reward should be per-robot (shape: n_agents,)
- Prior policy returns acceleration commands clipped to [-1, 1]
- Use numpy for all computations (will be converted to JAX later)
{auxiliary_notes}
""".strip()


# ============================================================================
# Auxiliary Information (Chain-of-thought hints)
# ============================================================================

AUXILIARY_INFO = {
    "assembly": {
        "cot": [
            "5. Consider how to determine the nearest unoccupied target cell for each robot.",
            "6. Consider how robots should balance exploration vs exploitation.",
        ],
        "notes": [
            "- Assembling into a shape is the goal, not a hard constraint.",
            "- Robots should prioritize avoiding collisions over reaching targets.",
            "- The prior policy should work even without perfect information.",
        ]
    }
}


def get_auxiliary_info(task_name: str) -> tuple:
    """Get chain-of-thought hints and notes for a task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        (cot_string, notes_string)
    """
    info = AUXILIARY_INFO.get(task_name, {"cot": [], "notes": []})
    
    cot = "\n".join(info.get("cot", []))
    notes = "\n".join(info.get("notes", []))
    
    return cot, notes


# ============================================================================
# Prompt Builder
# ============================================================================

def build_generation_prompt(
    task_name: str = "assembly",
    env_description: Optional[str] = None,
    task_description: Optional[str] = None,
    api_description: Optional[str] = None,
) -> str:
    """Build the complete prompt for reward/policy generation.
    
    Args:
        task_name: Name of the task (used to look up defaults)
        env_description: Override environment description
        task_description: Override task description
        api_description: Override API description
        
    Returns:
        Complete prompt string ready to send to GPT
    """
    # Get descriptions (use overrides or defaults)
    env_des = env_description or get_environment_description(task_name)
    task_des = task_description or get_task_description(task_name)
    api_des = api_description or get_robot_api_description(task_name)
    
    # Get auxiliary info
    cot, notes = get_auxiliary_info(task_name)
    
    # Build prompt
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        env_des=env_des,
        task_des=task_des,
        api_des=api_des,
        auxiliary_cot=cot,
        auxiliary_notes=notes,
    )
    
    return prompt


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are an expert in multi-agent reinforcement learning and swarm robotics. 
Your task is to design reward functions and prior policies for robot swarms.
Always provide complete, runnable Python code using numpy.
Be precise and follow the output format exactly."""
