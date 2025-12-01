"""Reward computation for multi-agent swarm assembly environments.

This module provides JAX-compatible reward functions for the assembly task.

Reward components:
1. Entering reward: Agent successfully enters and stays in target region
2. Interaction penalty: Penalty for colliding with other agents
3. Exploration reward: Reward for moving toward unoccupied grid cells

The reward can be:
- Individual: Each agent gets its own reward
- Shared (mean): All agents share the mean reward
- Shared (max): All agents share the max reward
"""

from typing import Tuple
from enum import Enum
import jax
import jax.numpy as jnp
from flax import struct

from physics import compute_pairwise_distances, compute_pairwise_distances_periodic


class RewardMode(Enum):
    """Reward sharing mode."""
    INDIVIDUAL = "individual"
    SHARED_MEAN = "shared_mean"
    SHARED_MAX = "shared_max"


@struct.dataclass
class RewardParams:
    """Parameters for reward computation.
    
    Attributes:
        reward_entering: Reward for being in target and not colliding
        penalty_collision: Penalty for colliding with another agent
        reward_exploration: Reward for exploring unoccupied areas
        collision_threshold: Distance threshold for collision penalty
        exploration_threshold: Distance threshold for exploration reward
        reward_mode: How to share rewards among agents
    """
    reward_entering: float = 1.0
    penalty_collision: float = -0.5
    reward_exploration: float = 0.1
    collision_threshold: float = 0.15  # r_avoid in original
    exploration_threshold: float = 0.07
    # Note: reward_mode is a string since Enum can't be in dataclass easily
    reward_mode: str = "individual"


def compute_in_target(
    positions: jax.Array,
    grid_centers: jax.Array,
    l_cell: float,
) -> jax.Array:
    """Determine which agents are inside target grid cells.
    
    An agent is "in target" if it's within sqrt(2)*l_cell/2 of a grid center.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        grid_centers: Target grid cell centers, shape (n_grid, 2)
        l_cell: Grid cell size
        
    Returns:
        in_target: Boolean array, shape (n_agents,)
    """
    # Distance from each agent to each grid cell
    rel_pos = grid_centers[None, :, :] - positions[:, None, :]  # (n_agents, n_grid, 2)
    distances = jnp.linalg.norm(rel_pos, axis=-1)  # (n_agents, n_grid)
    
    # Minimum distance to any grid cell
    min_distances = jnp.min(distances, axis=1)  # (n_agents,)
    
    # Threshold for being "in" a cell
    threshold = jnp.sqrt(2.0) * l_cell / 2.0
    
    in_target = min_distances < threshold
    
    return in_target


def compute_agent_collisions(
    positions: jax.Array,
    collision_threshold: float,
    is_periodic: bool = False,
    boundary_width: float = 2.4,
    boundary_height: float = 2.4,
) -> Tuple[jax.Array, jax.Array]:
    """Detect collisions between agents.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        collision_threshold: Distance below which agents are colliding
        is_periodic: Whether to use periodic boundaries
        boundary_width: Half-width of boundary
        boundary_height: Half-height of boundary
        
    Returns:
        is_colliding: Boolean, shape (n_agents,) - whether each agent is colliding
        collision_matrix: Boolean, shape (n_agents, n_agents) - pairwise collisions
    """
    n_agents = positions.shape[0]
    
    # Get pairwise distances
    if is_periodic:
        _, distances, _ = compute_pairwise_distances_periodic(
            positions, boundary_width, boundary_height
        )
    else:
        _, distances, _ = compute_pairwise_distances(positions)
    
    # Collision if distance < threshold (excluding self)
    collision_matrix = distances < collision_threshold
    # Zero diagonal (no self-collision)
    collision_matrix = collision_matrix & ~jnp.eye(n_agents, dtype=bool)
    
    # Agent is colliding if it collides with any other agent
    is_colliding = jnp.any(collision_matrix, axis=1)
    
    return is_colliding, collision_matrix


def compute_exploration_reward(
    positions: jax.Array,
    grid_centers: jax.Array,
    in_target: jax.Array,
    neighbor_indices: jax.Array,
    collision_threshold: float,
    exploration_threshold: float,
    d_sen: float,
) -> jax.Array:
    """Compute exploration reward for each agent.
    
    An agent gets exploration reward if:
    1. It is in the target region
    2. It is not colliding
    3. It is near the centroid of unoccupied grid cells
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        grid_centers: Target grid cell centers, shape (n_grid, 2)
        in_target: Whether each agent is in target, shape (n_agents,)
        neighbor_indices: Neighbor indices, shape (n_agents, k)
        collision_threshold: Distance for collision
        exploration_threshold: Distance to centroid for exploration reward
        d_sen: Sensing range
        
    Returns:
        exploration_reward: Shape (n_agents,), 1.0 if exploring, 0.0 otherwise
    """
    n_agents = positions.shape[0]
    n_grid = grid_centers.shape[0]
    
    # For each agent, find unoccupied grid cells within sensing range
    # A grid cell is "occupied" if any agent (including nearby agents) is close to it
    
    # Distance from each agent to each grid cell
    agent_to_grid = grid_centers[None, :, :] - positions[:, None, :]  # (n_agents, n_grid, 2)
    agent_to_grid_dist = jnp.linalg.norm(agent_to_grid, axis=-1)  # (n_agents, n_grid)
    
    # Grid cells within sensing range of each agent
    in_range = agent_to_grid_dist < d_sen  # (n_agents, n_grid)
    
    # For simplicity, consider a grid cell "unoccupied" if no agent is within
    # collision_threshold/2 of it
    # This is a simplified version - the original C++ code is more complex
    
    # Distance from all agents to all grid cells
    all_agent_to_grid_dist = jnp.linalg.norm(
        grid_centers[None, :, :] - positions[:, None, :], axis=-1
    )  # (n_agents, n_grid)
    
    # A grid cell is occupied if ANY agent is close to it
    occupied = jnp.any(all_agent_to_grid_dist < collision_threshold / 2, axis=0)  # (n_grid,)
    
    # Unoccupied and in range for each agent
    unoccupied_in_range = in_range & ~occupied[None, :]  # (n_agents, n_grid)
    
    # Compute centroid of unoccupied cells for each agent
    # Weight by whether cell is in range and unoccupied
    weights = unoccupied_in_range.astype(jnp.float32)  # (n_agents, n_grid)
    weight_sum = jnp.sum(weights, axis=1, keepdims=True)  # (n_agents, 1)
    weight_sum = jnp.maximum(weight_sum, 1e-8)  # Avoid division by zero
    
    # Weighted centroid
    centroids = jnp.sum(
        weights[:, :, None] * grid_centers[None, :, :], axis=1
    ) / weight_sum  # (n_agents, 2)
    
    # Distance from each agent to centroid
    dist_to_centroid = jnp.linalg.norm(positions - centroids, axis=1)  # (n_agents,)
    
    # Exploration reward if close to centroid
    is_exploring = dist_to_centroid < exploration_threshold
    
    # Only give reward if in target and has unoccupied cells
    has_unoccupied = jnp.sum(weights, axis=1) > 0
    exploration_reward = (in_target & is_exploring & has_unoccupied).astype(jnp.float32)
    
    return exploration_reward


def compute_rewards(
    positions: jax.Array,
    velocities: jax.Array,
    grid_centers: jax.Array,
    l_cell: float,
    neighbor_indices: jax.Array,
    reward_params: RewardParams,
    is_periodic: bool = False,
    boundary_width: float = 2.4,
    boundary_height: float = 2.4,
    d_sen: float = 3.0,
) -> Tuple[jax.Array, dict]:
    """Compute rewards for all agents.
    
    The reward for each agent is:
    - If in target AND not colliding AND exploring: +reward_entering
    - If colliding: +penalty_collision (usually negative)
    - Otherwise: 0
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        velocities: Agent velocities, shape (n_agents, 2)
        grid_centers: Target grid cell centers, shape (n_grid, 2)
        l_cell: Grid cell size
        neighbor_indices: Neighbor indices, shape (n_agents, k)
        reward_params: Reward parameters
        is_periodic: Whether to use periodic boundaries
        boundary_width: Half-width of boundary
        boundary_height: Half-height of boundary
        d_sen: Sensing range
        
    Returns:
        rewards: Rewards for each agent, shape (n_agents,)
        info: Dictionary with additional information
    """
    n_agents = positions.shape[0]
    
    # 1. Check if agents are in target region
    in_target = compute_in_target(positions, grid_centers, l_cell)
    
    # 2. Check for collisions
    is_colliding, collision_matrix = compute_agent_collisions(
        positions, reward_params.collision_threshold,
        is_periodic, boundary_width, boundary_height
    )
    
    # 3. Compute exploration component
    exploration = compute_exploration_reward(
        positions, grid_centers, in_target, neighbor_indices,
        reward_params.collision_threshold,
        reward_params.exploration_threshold,
        d_sen
    )
    
    # 4. Compute individual rewards
    # Agent gets full reward if: in_target AND not_colliding AND exploring
    task_complete = in_target & ~is_colliding & (exploration > 0)
    
    rewards = jnp.zeros(n_agents)
    rewards = jnp.where(task_complete, reward_params.reward_entering, rewards)
    
    # Add collision penalty
    rewards = rewards + is_colliding.astype(jnp.float32) * reward_params.penalty_collision
    
    # 5. Apply reward sharing mode
    if reward_params.reward_mode == "shared_mean":
        mean_reward = jnp.mean(rewards)
        rewards = jnp.full(n_agents, mean_reward)
    elif reward_params.reward_mode == "shared_max":
        max_reward = jnp.max(rewards)
        rewards = jnp.full(n_agents, max_reward)
    # else: individual (default), no change
    
    # Info dict
    info = {
        "in_target": in_target,
        "is_colliding": is_colliding,
        "exploration": exploration,
        "task_complete": task_complete,
        "num_in_target": jnp.sum(in_target),
        "num_collisions": jnp.sum(is_colliding),
    }
    
    return rewards, info


def compute_simple_rewards(
    positions: jax.Array,
    grid_centers: jax.Array,
    l_cell: float,
    collision_threshold: float,
    is_periodic: bool = False,
    boundary_width: float = 2.4,
    boundary_height: float = 2.4,
) -> jax.Array:
    """Simplified reward computation for basic use cases.
    
    Reward = +1 if in target and not colliding, 0 otherwise.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        grid_centers: Target grid cell centers, shape (n_grid, 2)
        l_cell: Grid cell size
        collision_threshold: Distance for collision detection
        is_periodic: Whether to use periodic boundaries
        boundary_width: Half-width of boundary
        boundary_height: Half-height of boundary
        
    Returns:
        rewards: Shape (n_agents,)
    """
    # Check if in target
    in_target = compute_in_target(positions, grid_centers, l_cell)
    
    # Check for collisions
    is_colliding, _ = compute_agent_collisions(
        positions, collision_threshold, is_periodic, boundary_width, boundary_height
    )
    
    # Reward if in target and not colliding
    rewards = (in_target & ~is_colliding).astype(jnp.float32)
    
    return rewards


# ============================================================================
# Metrics for evaluation (not used in training, but useful for monitoring)
# ============================================================================

def compute_coverage_rate(
    positions: jax.Array,
    grid_centers: jax.Array,
    l_cell: float,
) -> jax.Array:
    """Compute what fraction of grid cells are occupied by agents.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        grid_centers: Grid cell centers, shape (n_grid, 2)
        l_cell: Grid cell size
        
    Returns:
        coverage_rate: Scalar in [0, 1]
    """
    n_grid = grid_centers.shape[0]
    
    # Distance from each grid cell to nearest agent
    grid_to_agent = positions[None, :, :] - grid_centers[:, None, :]  # (n_grid, n_agents, 2)
    distances = jnp.linalg.norm(grid_to_agent, axis=-1)  # (n_grid, n_agents)
    min_distances = jnp.min(distances, axis=1)  # (n_grid,)
    
    # Cell is covered if an agent is within threshold
    threshold = jnp.sqrt(2.0) * l_cell / 2.0
    is_covered = min_distances < threshold
    
    coverage_rate = jnp.mean(is_covered.astype(jnp.float32))
    
    return coverage_rate


def compute_average_distance_to_target(
    positions: jax.Array,
    grid_centers: jax.Array,
) -> jax.Array:
    """Compute average distance from each agent to nearest grid cell.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        grid_centers: Grid cell centers, shape (n_grid, 2)
        
    Returns:
        avg_distance: Scalar
    """
    # Distance from each agent to each grid cell
    agent_to_grid = grid_centers[None, :, :] - positions[:, None, :]
    distances = jnp.linalg.norm(agent_to_grid, axis=-1)  # (n_agents, n_grid)
    
    # Minimum distance for each agent
    min_distances = jnp.min(distances, axis=1)  # (n_agents,)
    
    return jnp.mean(min_distances)


def compute_collision_count(
    positions: jax.Array,
    collision_threshold: float,
    is_periodic: bool = False,
    boundary_width: float = 2.4,
    boundary_height: float = 2.4,
) -> jax.Array:
    """Count number of collision pairs.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        collision_threshold: Distance for collision
        is_periodic: Whether to use periodic boundaries
        boundary_width: Half-width of boundary
        boundary_height: Half-height of boundary
        
    Returns:
        num_collisions: Scalar (integer)
    """
    _, collision_matrix = compute_agent_collisions(
        positions, collision_threshold, is_periodic, boundary_width, boundary_height
    )
    
    # Each collision is counted twice in the matrix, so divide by 2
    num_collisions = jnp.sum(collision_matrix) // 2
    
    return num_collisions
