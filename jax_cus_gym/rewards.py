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
import jax
import jax.numpy as jnp
from flax import struct

from physics import compute_pairwise_distances, compute_pairwise_distances_periodic


# Reward mode constants (integers for JAX tracing compatibility)
# 0 = individual (each agent gets own reward)
# 1 = shared_mean (all agents get mean reward)  
# 2 = shared_max (all agents get max reward)
REWARD_MODE_INDIVIDUAL = 0
REWARD_MODE_SHARED_MEAN = 1
REWARD_MODE_SHARED_MAX = 2


@struct.dataclass
class RewardParams:
    """Parameters for reward computation.
    
    Attributes:
        reward_entering: Reward for being in target and not colliding
        penalty_collision: Penalty for colliding with another agent (0.0 = no penalty, matches MARL)
        reward_exploration: Reward for exploring unoccupied areas
        collision_threshold: Distance threshold for collision penalty
        exploration_threshold: Distance threshold for exploration reward (norm of weighted centroid)
        cosine_decay_delta: Delta parameter for cosine decay function (0.0 matches C++ MARL)
        reward_mode: How to share rewards (0=individual, 1=shared_mean, 2=shared_max)
    """
    reward_entering: float = 1.0
    penalty_collision: float = 0.0  # No penalty (matches original MARL C++ code)
    reward_exploration: float = 0.1
    collision_threshold: float = 0.15  # r_avoid in original
    exploration_threshold: float = 0.05  # Matches MARL's 0.05 threshold for ||v_exp|| check
    cosine_decay_delta: float = 0.0  # Delta for cosine decay weighting (0.0 matches C++ MARL)
    # Use integer for JAX tracing compatibility: 0=individual, 1=shared_mean, 2=shared_max
    reward_mode: int = REWARD_MODE_INDIVIDUAL


def rho_cos_dec(z: jax.Array, r: float, delta: float = 0.0) -> jax.Array:
    """Cosine decay weighting function (matches C++ _rho_cos_dec).
    
    This function provides smooth distance-based weighting:
    - Returns 1.0 when z < delta * r (close range)
    - Smoothly decays via cosine when delta * r <= z < r
    - Returns 0.0 when z >= r (far away)
    
    Args:
        z: Distance values, any shape
        r: Maximum range (e.g., d_sen)
        delta: Transition point as fraction of r (default 0.5)
        
    Returns:
        Weights in [0, 1], same shape as z
    """
    # Normalize distance by r
    z_normalized = z / r
    
    # Compute cosine decay for transition zone
    # cos_arg = pi * (z/r - delta) / (1 - delta)
    cos_arg = jnp.pi * (z_normalized - delta) / (1.0 - delta)
    cos_decay = 0.5 * (1.0 + jnp.cos(cos_arg))
    
    # Apply conditions:
    # z < delta * r -> 1.0
    # delta * r <= z < r -> cos_decay  
    # z >= r -> 0.0
    result = jnp.where(
        z < delta * r,
        1.0,
        jnp.where(z < r, cos_decay, 0.0)
    )
    
    return result


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
    cosine_decay_delta: float = 0.0,
) -> jax.Array:
    """Compute exploration/uniformity reward for each agent.
    
    Matches the C++ MARL implementation:
    - For each agent, compute relative positions to sensed grid cells
    - Weight each relative position by cosine-decay function based on distance
    - Compute weighted centroid of relative positions: v_exp = Σ(ψ × rel_pos) / Σ(ψ)
    - Agent is "uniform" if ||v_exp|| < exploration_threshold (default 0.05)
    
    An agent gets exploration reward if:
    1. It is in the target region  
    2. The weighted centroid of relative positions to sensed grids has small norm
       (meaning the agent is well-centered among sensed grid cells)
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        grid_centers: Target grid cell centers, shape (n_grid, 2)
        in_target: Whether each agent is in target, shape (n_agents,)
        neighbor_indices: Neighbor indices, shape (n_agents, k) - unused but kept for API
        collision_threshold: Distance for collision - unused in exploration
        exploration_threshold: Threshold for ||v_exp|| check (default 0.05)
        d_sen: Sensing range for grid cells
        cosine_decay_delta: Delta parameter for cosine decay weighting
        
    Returns:
        exploration_reward: Shape (n_agents,), 1.0 if uniform, 0.0 otherwise
    """
    # Compute relative positions from each agent to each grid cell
    # rel_pos[i, j] = grid_centers[j] - positions[i]
    rel_pos = grid_centers[None, :, :] - positions[:, None, :]  # (n_agents, n_grid, 2)
    
    # Compute distances from each agent to each grid cell
    distances = jnp.linalg.norm(rel_pos, axis=-1)  # (n_agents, n_grid)
    
    # Compute cosine-decay weights based on distance
    # ψ(d) = rho_cos_dec(d, d_sen, delta)
    # Grids within sensing range get weight > 0, outside get 0
    psi_weights = rho_cos_dec(distances, d_sen, cosine_decay_delta)  # (n_agents, n_grid)
    
    # Compute weighted centroid of relative positions for each agent
    # v_exp_i = Σ_j(ψ_ij × rel_pos_ij) / Σ_j(ψ_ij)
    weight_sum = jnp.sum(psi_weights, axis=1, keepdims=True)  # (n_agents, 1)
    weight_sum = jnp.maximum(weight_sum, 1e-8)  # Avoid division by zero
    
    # Weighted sum of relative positions
    weighted_rel_pos = psi_weights[:, :, None] * rel_pos  # (n_agents, n_grid, 2)
    v_exp = jnp.sum(weighted_rel_pos, axis=1) / weight_sum  # (n_agents, 2)
    
    # Compute norm of weighted centroid
    v_exp_norm = jnp.linalg.norm(v_exp, axis=1)  # (n_agents,)
    
    # Agent is "uniform" if ||v_exp|| < threshold
    # This means the agent is approximately centered among sensed grid cells
    is_uniform = v_exp_norm < exploration_threshold
    
    # Only give reward if in target region
    # (Collision check is done separately in compute_rewards)
    exploration_reward = (in_target & is_uniform).astype(jnp.float32)
    
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
    
    # 3. Compute exploration/uniformity component (matches C++ MARL logic)
    exploration = compute_exploration_reward(
        positions, grid_centers, in_target, neighbor_indices,
        reward_params.collision_threshold,
        reward_params.exploration_threshold,
        d_sen,
        reward_params.cosine_decay_delta,
    )
    
    # 4. Compute individual rewards (matches MARL C++ logic)
    # Agent gets +1 reward ONLY if: in_target AND not_colliding AND exploring
    # Otherwise gets 0 (no negative penalties - this matches the original MARL behavior)
    task_complete = in_target & ~is_colliding & (exploration > 0)
    
    rewards = jnp.where(task_complete, reward_params.reward_entering, 0.0)
    
    # Optional collision penalty (disabled by default to match MARL, set penalty_collision < 0 to enable)
    rewards = jnp.where(
        reward_params.penalty_collision < 0,
        rewards + is_colliding.astype(jnp.float32) * reward_params.penalty_collision,
        rewards
    )
    
    # Note: No reward clipping (matches C++ MARL which outputs rewards as 0 or 1 directly)
    
    # 5. Apply reward sharing mode (using jnp.where for JIT compatibility)
    mean_reward = jnp.mean(rewards)
    max_reward = jnp.max(rewards)
    
    # Use nested where for JIT-compatible branching
    rewards = jnp.where(
        reward_params.reward_mode == REWARD_MODE_SHARED_MEAN,
        jnp.full(n_agents, mean_reward),
        jnp.where(
            reward_params.reward_mode == REWARD_MODE_SHARED_MAX,
            jnp.full(n_agents, max_reward),
            rewards  # individual mode
        )
    )
    
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
