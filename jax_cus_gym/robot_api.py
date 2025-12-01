"""Robot API for LLM-generated policies in JAX swarm environments.

This module provides a stable API that LLM-generated code can call.
All functions are pure (no side effects) and JAX-compatible.

The API mirrors the cus_gym interface but is adapted for JAX:
- Functions take state/arrays as input instead of using self
- All operations are JIT-compilable
- Vectorized versions available for operating on all agents at once

Usage for LLM-generated policies:
    ```python
    def llm_policy(state, agent_id, api):
        # Get this agent's info
        position, velocity = api.get_position_and_velocity(state, agent_id)
        
        # Get neighbors
        neighbor_ids = api.get_neighbor_ids(state, agent_id)
        
        # Check if in target
        in_target = api.is_within_target(state, agent_id)
        
        # Get target position
        target_pos = api.get_target_position(state, agent_id)
        
        # Compute action...
        return action
    ```
"""

from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
from flax import struct

from physics import compute_pairwise_distances


# ============================================================================
# Data Structures for API Results
# ============================================================================

class NeighborInfo(NamedTuple):
    """Information about an agent's neighbors."""
    ids: jnp.ndarray          # (k,) neighbor indices, -1 for invalid
    positions: jnp.ndarray    # (k, 2) neighbor positions
    velocities: jnp.ndarray   # (k, 2) neighbor velocities
    distances: jnp.ndarray    # (k,) distances to neighbors
    count: int                # Number of valid neighbors


class TargetInfo(NamedTuple):
    """Information about an agent's target."""
    position: jnp.ndarray     # (2,) position of nearest target cell
    distance: float           # Distance to nearest target cell
    in_target: bool           # Whether agent is inside target region


class GridInfo(NamedTuple):
    """Information about sensed grid cells."""
    positions: jnp.ndarray    # (max_cells, 2) positions of sensed cells
    occupied: jnp.ndarray     # (max_cells,) whether each cell is occupied
    mask: jnp.ndarray         # (max_cells,) valid cell mask
    count: int                # Number of valid cells


# ============================================================================
# Single-Agent API Functions
# ============================================================================

def get_position_and_velocity(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    agent_id: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get position and velocity for a single agent.
    
    Args:
        positions: All agent positions (n_agents, 2)
        velocities: All agent velocities (n_agents, 2)
        agent_id: Index of the agent
        
    Returns:
        position: Agent's position (2,)
        velocity: Agent's velocity (2,)
    """
    return positions[agent_id], velocities[agent_id]


def get_neighbor_ids(
    positions: jnp.ndarray,
    agent_id: int,
    d_sen: float,
    k_max: int,
) -> jnp.ndarray:
    """Get IDs of k nearest neighbors within sensing range.
    
    Args:
        positions: All agent positions (n_agents, 2)
        agent_id: Index of the agent
        d_sen: Sensing distance
        k_max: Maximum number of neighbors to return
        
    Returns:
        neighbor_ids: Array of neighbor indices (k_max,), -1 for invalid
    """
    n_agents = positions.shape[0]
    
    # Compute distances to all agents
    agent_pos = positions[agent_id]
    rel_positions = positions - agent_pos
    distances = jnp.linalg.norm(rel_positions, axis=-1)
    
    # Exclude self
    distances = jnp.where(jnp.arange(n_agents) == agent_id, jnp.inf, distances)
    
    # Mask agents outside sensing range
    distances = jnp.where(distances > d_sen, jnp.inf, distances)
    
    # Get k nearest
    sorted_indices = jnp.argsort(distances)
    neighbor_ids = sorted_indices[:k_max]
    
    # Mark invalid neighbors (outside sensing range) with -1
    neighbor_distances = distances[neighbor_ids]
    neighbor_ids = jnp.where(neighbor_distances < jnp.inf, neighbor_ids, -1)
    
    return neighbor_ids


def get_neighbor_info(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    agent_id: int,
    d_sen: float,
    k_max: int,
) -> NeighborInfo:
    """Get full information about an agent's neighbors.
    
    Args:
        positions: All agent positions (n_agents, 2)
        velocities: All agent velocities (n_agents, 2)
        agent_id: Index of the agent
        d_sen: Sensing distance
        k_max: Maximum number of neighbors
        
    Returns:
        NeighborInfo with ids, positions, velocities, distances, count
    """
    n_agents = positions.shape[0]
    agent_pos = positions[agent_id]
    
    # Compute distances
    rel_positions = positions - agent_pos
    distances = jnp.linalg.norm(rel_positions, axis=-1)
    
    # Exclude self and out-of-range
    distances = jnp.where(jnp.arange(n_agents) == agent_id, jnp.inf, distances)
    in_range = distances <= d_sen
    distances = jnp.where(in_range, distances, jnp.inf)
    
    # Sort by distance and get k nearest
    sorted_indices = jnp.argsort(distances)
    neighbor_ids = sorted_indices[:k_max]
    
    # Get neighbor data
    neighbor_positions = positions[neighbor_ids]
    neighbor_velocities = velocities[neighbor_ids]
    neighbor_distances = distances[neighbor_ids]
    
    # Create validity mask
    valid = neighbor_distances < jnp.inf
    neighbor_ids = jnp.where(valid, neighbor_ids, -1)
    count = jnp.sum(valid.astype(jnp.int32))
    
    return NeighborInfo(
        ids=neighbor_ids,
        positions=neighbor_positions,
        velocities=neighbor_velocities,
        distances=neighbor_distances,
        count=count,
    )


def is_within_target(
    positions: jnp.ndarray,
    grid_centers: jnp.ndarray,
    grid_mask: jnp.ndarray,
    l_cell: float,
    agent_id: int,
) -> bool:
    """Check if an agent is within the target region.
    
    An agent is "in target" if its distance to any valid grid cell center
    is less than sqrt(2) * l_cell / 2.
    
    Args:
        positions: All agent positions (n_agents, 2)
        grid_centers: Target grid cell centers (n_grid, 2)
        grid_mask: Valid grid cell mask (n_grid,)
        l_cell: Grid cell size
        agent_id: Index of the agent
        
    Returns:
        True if agent is within target region
    """
    agent_pos = positions[agent_id]
    
    # Distance to each grid cell
    rel_to_grid = grid_centers - agent_pos
    distances = jnp.linalg.norm(rel_to_grid, axis=-1)
    
    # Only consider valid grid cells
    distances = jnp.where(grid_mask, distances, jnp.inf)
    
    # Check if within threshold of any cell
    threshold = jnp.sqrt(2.0) * l_cell / 2.0
    min_distance = jnp.min(distances)
    
    return min_distance < threshold


def get_target_position(
    positions: jnp.ndarray,
    grid_centers: jnp.ndarray,
    grid_mask: jnp.ndarray,
    agent_id: int,
) -> jnp.ndarray:
    """Get the position of the nearest target grid cell.
    
    Args:
        positions: All agent positions (n_agents, 2)
        grid_centers: Target grid cell centers (n_grid, 2)
        grid_mask: Valid grid cell mask (n_grid,)
        agent_id: Index of the agent
        
    Returns:
        Position of nearest valid grid cell (2,)
    """
    agent_pos = positions[agent_id]
    
    # Distance to each grid cell
    rel_to_grid = grid_centers - agent_pos
    distances = jnp.linalg.norm(rel_to_grid, axis=-1)
    
    # Only consider valid grid cells
    distances = jnp.where(grid_mask, distances, jnp.inf)
    
    # Find nearest
    nearest_idx = jnp.argmin(distances)
    return grid_centers[nearest_idx]


def get_target_info(
    positions: jnp.ndarray,
    grid_centers: jnp.ndarray,
    grid_mask: jnp.ndarray,
    l_cell: float,
    agent_id: int,
) -> TargetInfo:
    """Get full information about an agent's target.
    
    Args:
        positions: All agent positions (n_agents, 2)
        grid_centers: Target grid cell centers (n_grid, 2)
        grid_mask: Valid grid cell mask (n_grid,)
        l_cell: Grid cell size
        agent_id: Index of the agent
        
    Returns:
        TargetInfo with position, distance, in_target
    """
    agent_pos = positions[agent_id]
    
    # Distance to each grid cell
    rel_to_grid = grid_centers - agent_pos
    distances = jnp.linalg.norm(rel_to_grid, axis=-1)
    
    # Only consider valid grid cells
    distances = jnp.where(grid_mask, distances, jnp.inf)
    
    # Find nearest
    nearest_idx = jnp.argmin(distances)
    nearest_pos = grid_centers[nearest_idx]
    nearest_dist = distances[nearest_idx]
    
    # Check if in target
    threshold = jnp.sqrt(2.0) * l_cell / 2.0
    in_target = nearest_dist < threshold
    
    return TargetInfo(
        position=nearest_pos,
        distance=nearest_dist,
        in_target=in_target,
    )


def get_unoccupied_cells(
    positions: jnp.ndarray,
    grid_centers: jnp.ndarray,
    grid_mask: jnp.ndarray,
    agent_id: int,
    d_sen: float,
    r_occupy: float,
    max_cells: int = 80,
) -> GridInfo:
    """Get positions of unoccupied grid cells within sensing range.
    
    A cell is "occupied" if any agent is within r_occupy of it.
    
    Args:
        positions: All agent positions (n_agents, 2)
        grid_centers: Target grid cell centers (n_grid, 2)
        grid_mask: Valid grid cell mask (n_grid,)
        agent_id: Index of the agent
        d_sen: Sensing distance
        r_occupy: Radius for occupation check
        max_cells: Maximum cells to return
        
    Returns:
        GridInfo with positions, occupied flags, mask, count
    """
    n_agents = positions.shape[0]
    n_grid = grid_centers.shape[0]
    agent_pos = positions[agent_id]
    
    # Distance from agent to each grid cell
    rel_to_grid = grid_centers - agent_pos
    dist_to_grid = jnp.linalg.norm(rel_to_grid, axis=-1)
    
    # Grid cells within sensing range
    in_range = (dist_to_grid <= d_sen) & grid_mask
    
    # Check which cells are occupied (any agent within r_occupy)
    # Distance from each agent to each grid cell
    agent_to_grid = grid_centers[None, :, :] - positions[:, None, :]  # (n_agents, n_grid, 2)
    agent_to_grid_dist = jnp.linalg.norm(agent_to_grid, axis=-1)  # (n_agents, n_grid)
    
    # A cell is occupied if any agent is close
    occupied = jnp.any(agent_to_grid_dist < r_occupy, axis=0)  # (n_grid,)
    
    # Filter: in range AND not occupied
    unoccupied_in_range = in_range & ~occupied
    
    # Sort by distance and get closest cells
    # Set distance to inf for cells not in range or occupied
    sorted_dist = jnp.where(unoccupied_in_range, dist_to_grid, jnp.inf)
    sorted_indices = jnp.argsort(sorted_dist)[:max_cells]
    
    # Get cell data
    cell_positions = grid_centers[sorted_indices]
    cell_occupied = occupied[sorted_indices]
    cell_valid = sorted_dist[sorted_indices] < jnp.inf
    count = jnp.sum(cell_valid.astype(jnp.int32))
    
    return GridInfo(
        positions=cell_positions,
        occupied=cell_occupied,
        mask=cell_valid,
        count=count,
    )


# ============================================================================
# Vectorized API Functions (operate on all agents at once)
# ============================================================================

def get_all_positions_and_velocities(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get positions and velocities for all agents.
    
    Args:
        positions: All agent positions (n_agents, 2)
        velocities: All agent velocities (n_agents, 2)
        
    Returns:
        positions: (n_agents, 2)
        velocities: (n_agents, 2)
    """
    return positions, velocities


def get_all_in_target(
    positions: jnp.ndarray,
    grid_centers: jnp.ndarray,
    grid_mask: jnp.ndarray,
    l_cell: float,
) -> jnp.ndarray:
    """Check which agents are within the target region.
    
    Args:
        positions: All agent positions (n_agents, 2)
        grid_centers: Target grid cell centers (n_grid, 2)
        grid_mask: Valid grid cell mask (n_grid,)
        l_cell: Grid cell size
        
    Returns:
        in_target: Boolean array (n_agents,)
    """
    # Distance from each agent to each grid cell
    rel_to_grid = grid_centers[None, :, :] - positions[:, None, :]  # (n_agents, n_grid, 2)
    distances = jnp.linalg.norm(rel_to_grid, axis=-1)  # (n_agents, n_grid)
    
    # Mask invalid grid cells
    distances = jnp.where(grid_mask[None, :], distances, jnp.inf)
    
    # Minimum distance for each agent
    min_distances = jnp.min(distances, axis=1)  # (n_agents,)
    
    # Check threshold
    threshold = jnp.sqrt(2.0) * l_cell / 2.0
    return min_distances < threshold


def get_all_target_positions(
    positions: jnp.ndarray,
    grid_centers: jnp.ndarray,
    grid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Get nearest target position for all agents.
    
    Args:
        positions: All agent positions (n_agents, 2)
        grid_centers: Target grid cell centers (n_grid, 2)
        grid_mask: Valid grid cell mask (n_grid,)
        
    Returns:
        target_positions: (n_agents, 2)
    """
    # Distance from each agent to each grid cell
    rel_to_grid = grid_centers[None, :, :] - positions[:, None, :]  # (n_agents, n_grid, 2)
    distances = jnp.linalg.norm(rel_to_grid, axis=-1)  # (n_agents, n_grid)
    
    # Mask invalid grid cells
    distances = jnp.where(grid_mask[None, :], distances, jnp.inf)
    
    # Find nearest for each agent
    nearest_indices = jnp.argmin(distances, axis=1)  # (n_agents,)
    
    return grid_centers[nearest_indices]


def get_all_neighbor_ids(
    positions: jnp.ndarray,
    d_sen: float,
    k_max: int,
) -> jnp.ndarray:
    """Get neighbor IDs for all agents.
    
    Args:
        positions: All agent positions (n_agents, 2)
        d_sen: Sensing distance
        k_max: Maximum neighbors per agent
        
    Returns:
        neighbor_ids: (n_agents, k_max), -1 for invalid
    """
    n_agents = positions.shape[0]
    
    # Pairwise distances
    _, distances, _ = compute_pairwise_distances(positions)
    
    # Exclude self by setting diagonal to inf
    distances = distances.at[jnp.arange(n_agents), jnp.arange(n_agents)].set(jnp.inf)
    
    # Sort by distance for each agent
    sorted_indices = jnp.argsort(distances, axis=1)
    
    # Get the k nearest indices
    k_nearest_indices = sorted_indices[:, :k_max]
    
    # Get the distances for these k nearest
    k_nearest_distances = jnp.take_along_axis(distances, k_nearest_indices, axis=1)
    
    # Mark invalid (out of range) with -1
    valid = k_nearest_distances <= d_sen
    neighbor_ids = jnp.where(valid, k_nearest_indices, -1)
    
    return neighbor_ids


def get_all_collisions(
    positions: jnp.ndarray,
    collision_threshold: float,
) -> jnp.ndarray:
    """Check which agents are colliding.
    
    Args:
        positions: All agent positions (n_agents, 2)
        collision_threshold: Distance for collision
        
    Returns:
        is_colliding: Boolean array (n_agents,)
    """
    n_agents = positions.shape[0]
    
    # Pairwise distances
    _, distances, _ = compute_pairwise_distances(positions)
    
    # Collision if close (excluding self)
    collision_matrix = (distances < collision_threshold) & ~jnp.eye(n_agents, dtype=bool)
    
    # Agent is colliding if any neighbor is too close
    return jnp.any(collision_matrix, axis=1)


# ============================================================================
# State-based API (convenience wrappers using AssemblyState)
# ============================================================================

@struct.dataclass
class RobotAPI:
    """Stateful API wrapper for convenience.
    
    Wraps the functional API with state for cleaner LLM-generated code.
    
    Usage:
        api = RobotAPI.from_state(state, params)
        pos, vel = api.get_position_and_velocity(agent_id)
        neighbors = api.get_neighbor_ids(agent_id)
    """
    positions: jnp.ndarray
    velocities: jnp.ndarray
    grid_centers: jnp.ndarray
    grid_mask: jnp.ndarray
    l_cell: float
    d_sen: float
    k_max: int
    r_avoid: float
    
    @classmethod
    def from_state(cls, state, params) -> "RobotAPI":
        """Create API from environment state and params.
        
        Args:
            state: AssemblyState
            params: AssemblyParams
            
        Returns:
            RobotAPI instance
        """
        return cls(
            positions=state.positions,
            velocities=state.velocities,
            grid_centers=state.grid_centers,
            grid_mask=state.grid_mask,
            l_cell=state.l_cell,
            d_sen=params.d_sen,
            k_max=params.k_neighbors,
            r_avoid=params.reward_params.collision_threshold,
        )
    
    def get_position_and_velocity(self, agent_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get position and velocity for an agent."""
        return get_position_and_velocity(self.positions, self.velocities, agent_id)
    
    def get_neighbor_ids(self, agent_id: int) -> jnp.ndarray:
        """Get neighbor IDs for an agent."""
        return get_neighbor_ids(self.positions, agent_id, self.d_sen, self.k_max)
    
    def get_neighbor_info(self, agent_id: int) -> NeighborInfo:
        """Get full neighbor info for an agent."""
        return get_neighbor_info(
            self.positions, self.velocities, agent_id, self.d_sen, self.k_max
        )
    
    def is_within_target(self, agent_id: int) -> bool:
        """Check if agent is in target region."""
        return is_within_target(
            self.positions, self.grid_centers, self.grid_mask, self.l_cell, agent_id
        )
    
    def get_target_position(self, agent_id: int) -> jnp.ndarray:
        """Get nearest target position for an agent."""
        return get_target_position(
            self.positions, self.grid_centers, self.grid_mask, agent_id
        )
    
    def get_target_info(self, agent_id: int) -> TargetInfo:
        """Get full target info for an agent."""
        return get_target_info(
            self.positions, self.grid_centers, self.grid_mask, self.l_cell, agent_id
        )
    
    def get_unoccupied_cells(self, agent_id: int, max_cells: int = 80) -> GridInfo:
        """Get unoccupied cells near an agent."""
        return get_unoccupied_cells(
            self.positions, self.grid_centers, self.grid_mask,
            agent_id, self.d_sen, self.r_avoid / 2, max_cells
        )
    
    # Vectorized methods
    def get_all_in_target(self) -> jnp.ndarray:
        """Check which agents are in target."""
        return get_all_in_target(
            self.positions, self.grid_centers, self.grid_mask, self.l_cell
        )
    
    def get_all_target_positions(self) -> jnp.ndarray:
        """Get target positions for all agents."""
        return get_all_target_positions(
            self.positions, self.grid_centers, self.grid_mask
        )
    
    def get_all_neighbor_ids(self) -> jnp.ndarray:
        """Get neighbor IDs for all agents."""
        return get_all_neighbor_ids(self.positions, self.d_sen, self.k_max)
    
    def get_all_collisions(self) -> jnp.ndarray:
        """Check which agents are colliding."""
        return get_all_collisions(self.positions, self.r_avoid)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Data structures
    "NeighborInfo",
    "TargetInfo", 
    "GridInfo",
    "RobotAPI",
    # Single-agent functions
    "get_position_and_velocity",
    "get_neighbor_ids",
    "get_neighbor_info",
    "is_within_target",
    "get_target_position",
    "get_target_info",
    "get_unoccupied_cells",
    # Vectorized functions
    "get_all_positions_and_velocities",
    "get_all_in_target",
    "get_all_target_positions",
    "get_all_neighbor_ids",
    "get_all_collisions",
]
