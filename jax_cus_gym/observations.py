"""Observation computation for multi-agent swarm environments.

This module provides JAX-compatible functions for computing agent observations.

Each agent's observation includes:
1. Self state: own position and velocity (optional)
2. Neighbor states: relative positions and velocities of k nearest neighbors
3. Target grid: relative positions of sensed grid cells in the target shape

Key design considerations:
- Fixed observation dimension (padded with zeros if fewer neighbors/grid cells)
- All computations vectorized with JAX operations
- No Python loops for GPU efficiency
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from flax import struct

from physics import compute_pairwise_distances, compute_pairwise_distances_periodic


@struct.dataclass
class ObservationParams:
    """Parameters for observation computation.
    
    Attributes:
        topo_nei_max: Maximum number of neighbors to observe
        num_obs_grid_max: Maximum number of grid cells to observe
        d_sen: Sensing range for neighbors
        include_self_state: Whether to include agent's own state
        normalize_obs: Whether to normalize observations
        l_max: Maximum arena dimension (for normalization) - should match arena_size/2
        vel_max: Maximum velocity (for normalization)
    """
    topo_nei_max: int = 6
    num_obs_grid_max: int = 80
    d_sen: float = 3.0
    include_self_state: bool = True
    normalize_obs: bool = True
    l_max: float = 2.5  # Fixed: matches default arena_size/2 = 5.0/2
    vel_max: float = 0.8


def compute_observation_dim(obs_params: ObservationParams) -> int:
    """Compute the total observation dimension per agent.
    
    Observation structure:
    - Self state (if included): 2*dim (position + velocity)
    - Neighbors: 2*dim*topo_nei_max (relative pos + relative vel for each)
    - Target grid: dim*num_obs_grid_max (relative positions of grid cells)
    - Target info: 2*dim (nearest target position + velocity)
    
    For dim=2:
    - Self: 4 (if included)
    - Neighbors: 4 * topo_nei_max
    - Grid: 2 * num_obs_grid_max  
    - Target: 4
    
    Args:
        obs_params: Observation parameters
        
    Returns:
        Total observation dimension
    """
    dim = 2
    self_dim = 2 * dim if obs_params.include_self_state else 0
    neighbor_dim = 2 * dim * obs_params.topo_nei_max
    target_dim = 2 * dim  # Target position and velocity (relative)
    grid_dim = dim * obs_params.num_obs_grid_max
    
    return self_dim + neighbor_dim + target_dim + grid_dim


def get_k_nearest_neighbors(
    agent_idx: int,
    positions: jax.Array,
    velocities: jax.Array,
    k: int,
    d_sen: float,
    is_periodic: bool = False,
    boundary_width: float = 2.4,
    boundary_height: float = 2.4,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Get k nearest neighbors for a single agent.
    
    Args:
        agent_idx: Index of the agent
        positions: All agent positions, shape (n_agents, 2)
        velocities: All agent velocities, shape (n_agents, 2)
        k: Maximum number of neighbors to return
        d_sen: Sensing range
        is_periodic: Whether to use periodic boundaries
        boundary_width: Half-width of boundary
        boundary_height: Half-height of boundary
        
    Returns:
        rel_positions: Relative positions of neighbors, shape (k, 2), zero-padded
        rel_velocities: Relative velocities of neighbors, shape (k, 2), zero-padded
        neighbor_mask: Boolean mask of valid neighbors, shape (k,)
        neighbor_indices: Indices of neighbors, shape (k,), -1 for invalid
    """
    n_agents = positions.shape[0]
    
    # Compute distances from this agent to all others
    if is_periodic:
        rel_pos_all, distances_all, _ = compute_pairwise_distances_periodic(
            positions, boundary_width, boundary_height
        )
    else:
        rel_pos_all, distances_all, _ = compute_pairwise_distances(positions)
    
    # Get distances from this agent
    distances = distances_all[agent_idx]  # Shape: (n_agents,)
    
    # Set self-distance to infinity so it's not selected
    distances = distances.at[agent_idx].set(jnp.inf)
    
    # Set distances beyond sensing range to infinity
    distances = jnp.where(distances > d_sen, jnp.inf, distances)
    
    # Get indices of k nearest (argsort and take first k)
    sorted_indices = jnp.argsort(distances)
    neighbor_indices = sorted_indices[:k]
    
    # Get sorted distances
    sorted_distances = distances[sorted_indices][:k]
    
    # Create mask for valid neighbors (within sensing range)
    neighbor_mask = sorted_distances < jnp.inf
    
    # Get relative positions and velocities
    # rel_pos_all[agent_idx, j] = positions[j] - positions[agent_idx]
    rel_positions = rel_pos_all[agent_idx, neighbor_indices]  # Shape: (k, 2)
    
    # Relative velocities
    my_velocity = velocities[agent_idx]
    rel_velocities = velocities[neighbor_indices] - my_velocity  # Shape: (k, 2)
    
    # Zero out invalid neighbors
    rel_positions = jnp.where(neighbor_mask[:, None], rel_positions, 0.0)
    rel_velocities = jnp.where(neighbor_mask[:, None], rel_velocities, 0.0)
    
    # Set invalid indices to -1
    neighbor_indices = jnp.where(neighbor_mask, neighbor_indices, -1)
    
    return rel_positions, rel_velocities, neighbor_mask, neighbor_indices


def get_k_nearest_neighbors_all_agents(
    positions: jax.Array,
    velocities: jax.Array,
    k: int,
    d_sen: float,
    is_periodic: bool = False,
    boundary_width: float = 2.4,
    boundary_height: float = 2.4,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Get k nearest neighbors for all agents (vectorized).
    
    Args:
        positions: All agent positions, shape (n_agents, 2)
        velocities: All agent velocities, shape (n_agents, 2)
        k: Maximum number of neighbors
        d_sen: Sensing range
        is_periodic: Whether to use periodic boundaries
        boundary_width: Half-width of boundary
        boundary_height: Half-height of boundary
        
    Returns:
        rel_positions: Shape (n_agents, k, 2)
        rel_velocities: Shape (n_agents, k, 2)
        neighbor_masks: Shape (n_agents, k)
        neighbor_indices: Shape (n_agents, k)
    """
    n_agents = positions.shape[0]
    
    # Compute all pairwise distances
    if is_periodic:
        rel_pos_all, distances_all, _ = compute_pairwise_distances_periodic(
            positions, boundary_width, boundary_height
        )
    else:
        rel_pos_all, distances_all, _ = compute_pairwise_distances(positions)
    
    # Set diagonal to infinity (no self-neighbors)
    distances_all = distances_all.at[jnp.arange(n_agents), jnp.arange(n_agents)].set(jnp.inf)
    
    # Set distances beyond sensing range to infinity
    distances_all = jnp.where(distances_all > d_sen, jnp.inf, distances_all)
    
    # For each agent, get k nearest
    sorted_indices = jnp.argsort(distances_all, axis=1)[:, :k]  # (n_agents, k)
    
    # Get sorted distances
    # Use advanced indexing to get the distances
    agent_indices = jnp.arange(n_agents)[:, None]  # (n_agents, 1)
    sorted_distances = distances_all[agent_indices, sorted_indices]  # (n_agents, k)
    
    # Create masks
    neighbor_masks = sorted_distances < jnp.inf  # (n_agents, k)
    
    # Get relative positions
    # rel_pos_all[i, j] = positions[j] - positions[i]
    # We want rel_pos_all[i, sorted_indices[i, :]]
    rel_positions = rel_pos_all[agent_indices, sorted_indices]  # (n_agents, k, 2)
    
    # Get relative velocities
    neighbor_velocities = velocities[sorted_indices]  # (n_agents, k, 2)
    my_velocities = velocities[:, None, :]  # (n_agents, 1, 2)
    rel_velocities = neighbor_velocities - my_velocities  # (n_agents, k, 2)
    
    # Zero out invalid entries
    rel_positions = jnp.where(neighbor_masks[:, :, None], rel_positions, 0.0)
    rel_velocities = jnp.where(neighbor_masks[:, :, None], rel_velocities, 0.0)
    
    # Invalid indices to -1
    neighbor_indices = jnp.where(neighbor_masks, sorted_indices, -1)
    
    return rel_positions, rel_velocities, neighbor_masks, neighbor_indices


def compute_grid_observations(
    positions: jax.Array,
    grid_centers: jax.Array,
    d_sen: float,
    num_obs_grid_max: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute grid cell observations for all agents.
    
    Each agent observes the relative positions of nearby grid cells.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        grid_centers: Target grid cell centers, shape (n_grid, 2)
        d_sen: Sensing range
        num_obs_grid_max: Maximum number of grid cells to include
        
    Returns:
        rel_grid_positions: Relative positions to grid cells, shape (n_agents, num_obs_grid_max, 2)
        grid_masks: Valid grid cell mask, shape (n_agents, num_obs_grid_max)
        nearest_grid_idx: Index of nearest grid cell for each agent, shape (n_agents,)
    """
    n_agents = positions.shape[0]
    n_grid = grid_centers.shape[0]
    
    # Compute relative positions from each agent to each grid cell
    # Shape: (n_agents, n_grid, 2)
    rel_pos = grid_centers[None, :, :] - positions[:, None, :]
    
    # Compute distances
    distances = jnp.linalg.norm(rel_pos, axis=-1)  # (n_agents, n_grid)
    
    # Find nearest grid cell for each agent
    nearest_grid_idx = jnp.argmin(distances, axis=1)  # (n_agents,)
    
    # Mask for cells within sensing range
    in_range = distances < d_sen  # (n_agents, n_grid)
    
    # Set out-of-range distances to infinity for sorting
    distances_masked = jnp.where(in_range, distances, jnp.inf)
    
    # Get indices of nearest grid cells (up to num_obs_grid_max)
    # We need to handle the case where num_obs_grid_max > n_grid
    k = jnp.minimum(num_obs_grid_max, n_grid)
    
    sorted_indices = jnp.argsort(distances_masked, axis=1)[:, :num_obs_grid_max]  # (n_agents, num_obs_grid_max)
    
    # Get sorted distances
    agent_indices = jnp.arange(n_agents)[:, None]
    sorted_distances = distances_masked[agent_indices, sorted_indices]  # (n_agents, num_obs_grid_max)
    
    # Create mask
    grid_masks = sorted_distances < jnp.inf  # (n_agents, num_obs_grid_max)
    
    # Get relative positions
    rel_grid_positions = rel_pos[agent_indices, sorted_indices]  # (n_agents, num_obs_grid_max, 2)
    
    # Zero out invalid entries
    rel_grid_positions = jnp.where(grid_masks[:, :, None], rel_grid_positions, 0.0)
    
    return rel_grid_positions, grid_masks, nearest_grid_idx


def compute_target_state(
    positions: jax.Array,
    velocities: jax.Array,
    grid_centers: jax.Array,
    l_cell: float,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute target grid state for each agent.
    
    Determines if each agent is "in" the target region (within a grid cell)
    and computes the relative position/velocity to the nearest target.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        velocities: Agent velocities, shape (n_agents, 2)
        grid_centers: Target grid cell centers, shape (n_grid, 2)
        l_cell: Grid cell size
        
    Returns:
        in_target: Boolean, whether agent is inside a grid cell, shape (n_agents,)
        target_rel_pos: Relative position to target, shape (n_agents, 2)
        target_rel_vel: Relative velocity to target, shape (n_agents, 2)
    """
    n_agents = positions.shape[0]
    
    # Compute distances to all grid cells
    rel_pos = grid_centers[None, :, :] - positions[:, None, :]  # (n_agents, n_grid, 2)
    distances = jnp.linalg.norm(rel_pos, axis=-1)  # (n_agents, n_grid)
    
    # Find nearest grid cell
    nearest_idx = jnp.argmin(distances, axis=1)  # (n_agents,)
    nearest_distances = jnp.min(distances, axis=1)  # (n_agents,)
    
    # Check if agent is "in" a grid cell (within sqrt(2)*l_cell/2 of center)
    in_threshold = jnp.sqrt(2.0) * l_cell / 2.0
    in_target = nearest_distances < in_threshold  # (n_agents,)
    
    # Get nearest grid center
    nearest_grid = grid_centers[nearest_idx]  # (n_agents, 2)
    
    # If in target, target position is agent's own position (stay put)
    # If not in target, target position is nearest grid cell
    target_pos = jnp.where(in_target[:, None], positions, nearest_grid)
    
    # Target velocity: 0 if in target, otherwise 0 (grid doesn't move)
    target_vel = jnp.zeros_like(velocities)
    
    # Relative values
    target_rel_pos = target_pos - positions
    target_rel_vel = target_vel - velocities
    
    return in_target, target_rel_pos, target_rel_vel


def compute_observations(
    positions: jax.Array,
    velocities: jax.Array,
    grid_centers: jax.Array,
    l_cell: float,
    obs_params: ObservationParams,
    is_periodic: bool = False,
    boundary_width: float = 2.4,
    boundary_height: float = 2.4,
) -> jax.Array:
    """Compute observations for all agents.
    
    This is the main observation function that combines all components.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        velocities: Agent velocities, shape (n_agents, 2)
        grid_centers: Target grid cell centers, shape (n_grid, 2)
        l_cell: Grid cell size
        obs_params: Observation parameters
        is_periodic: Whether boundaries are periodic
        boundary_width: Half-width of boundary
        boundary_height: Half-height of boundary
        
    Returns:
        observations: Observations for all agents, shape (n_agents, obs_dim)
    """
    n_agents = positions.shape[0]
    
    # 1. Get neighbor information
    rel_pos_neighbors, rel_vel_neighbors, neighbor_masks, neighbor_indices = \
        get_k_nearest_neighbors_all_agents(
            positions, velocities,
            obs_params.topo_nei_max,
            obs_params.d_sen,
            is_periodic, boundary_width, boundary_height
        )
    
    # 2. Get target/grid information
    in_target, target_rel_pos, target_rel_vel = compute_target_state(
        positions, velocities, grid_centers, l_cell
    )
    
    # 3. Get grid cell observations
    rel_grid_positions, grid_masks, nearest_grid_idx = compute_grid_observations(
        positions, grid_centers, obs_params.d_sen, obs_params.num_obs_grid_max
    )
    
    # 4. Normalize if requested
    if obs_params.normalize_obs:
        norm_pos = 2.0 * obs_params.l_max
        norm_vel = 2.0 * obs_params.vel_max
        
        # Normalize positions
        positions_norm = positions / obs_params.l_max
        rel_pos_neighbors = rel_pos_neighbors / norm_pos
        rel_vel_neighbors = rel_vel_neighbors / norm_vel
        target_rel_pos = target_rel_pos / norm_pos
        target_rel_vel = target_rel_vel / norm_vel
        rel_grid_positions = rel_grid_positions / norm_pos
        velocities_norm = velocities / obs_params.vel_max
    else:
        positions_norm = positions
        velocities_norm = velocities
    
    # 5. Flatten and concatenate into observation vector
    obs_parts = []
    
    # Self state (position, velocity)
    if obs_params.include_self_state:
        obs_parts.append(positions_norm)  # (n_agents, 2)
        obs_parts.append(velocities_norm)  # (n_agents, 2)
    
    # Neighbor states (flattened)
    obs_parts.append(rel_pos_neighbors.reshape(n_agents, -1))  # (n_agents, k*2)
    obs_parts.append(rel_vel_neighbors.reshape(n_agents, -1))  # (n_agents, k*2)
    
    # Target state
    obs_parts.append(target_rel_pos)  # (n_agents, 2)
    obs_parts.append(target_rel_vel)  # (n_agents, 2)
    
    # Grid observations (flattened)
    obs_parts.append(rel_grid_positions.reshape(n_agents, -1))  # (n_agents, num_grid*2)
    
    # Concatenate all parts
    observations = jnp.concatenate(obs_parts, axis=1)
    
    return observations


def get_neighbor_indices(
    positions: jax.Array,
    k: int,
    d_sen: float,
    is_periodic: bool = False,
    boundary_width: float = 2.4,
    boundary_height: float = 2.4,
) -> jax.Array:
    """Get neighbor indices for all agents (utility function).
    
    Useful for reward computation and other calculations that need neighbor info.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        k: Maximum neighbors per agent
        d_sen: Sensing range
        is_periodic: Whether to use periodic boundaries
        boundary_width: Half-width of boundary
        boundary_height: Half-height of boundary
        
    Returns:
        neighbor_indices: Shape (n_agents, k), -1 for invalid
    """
    n_agents = positions.shape[0]
    
    if is_periodic:
        _, distances_all, _ = compute_pairwise_distances_periodic(
            positions, boundary_width, boundary_height
        )
    else:
        _, distances_all, _ = compute_pairwise_distances(positions)
    
    # Set diagonal and out-of-range to infinity
    distances_all = distances_all.at[jnp.arange(n_agents), jnp.arange(n_agents)].set(jnp.inf)
    distances_all = jnp.where(distances_all > d_sen, jnp.inf, distances_all)
    
    # Sort and take k nearest
    sorted_indices = jnp.argsort(distances_all, axis=1)[:, :k]
    
    # Get sorted distances for masking
    agent_indices = jnp.arange(n_agents)[:, None]
    sorted_distances = distances_all[agent_indices, sorted_indices]
    
    # Mask invalid
    valid_mask = sorted_distances < jnp.inf
    neighbor_indices = jnp.where(valid_mask, sorted_indices, -1)
    
    return neighbor_indices
