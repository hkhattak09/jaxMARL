"""Physics simulation for multi-agent swarm environments.

This module provides JAX-compatible physics functions for:
1. Agent-to-agent collision detection and response
2. Agent-to-wall collision detection and response  
3. Position/velocity integration
4. Periodic boundary conditions

All functions are designed to be JIT-compilable and vmap-compatible.

Key design decisions:
- All computations are vectorized over agents using JAX operations
- No Python loops - everything uses jnp operations for GPU acceleration
- Collision forces use spring-damper models for stability
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class PhysicsParams:
    """Parameters for physics simulation.
    
    Attributes:
        k_ball: Spring stiffness for ball-ball collisions (N/m)
        k_wall: Spring stiffness for ball-wall collisions (N/m)
        c_wall: Damping coefficient for wall collisions (N·s/m)
        c_aero: Aerodynamic drag coefficient (N·s/m)
        agent_radius: Radius of each agent (m)
        agent_mass: Mass of each agent (kg)
        dt: Time step (s)
        vel_max: Maximum velocity magnitude (m/s)
    """
    k_ball: float = 30.0
    k_wall: float = 100.0
    c_wall: float = 5.0
    c_aero: float = 1.2
    agent_radius: float = 0.035
    agent_mass: float = 1.0
    dt: float = 0.1
    vel_max: float = 0.8


def compute_pairwise_distances(
    positions: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute pairwise distances between all agents.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        
    Returns:
        rel_pos: Relative positions, shape (n_agents, n_agents, 2)
                 rel_pos[i, j] = positions[j] - positions[i]
        distances: Pairwise distances, shape (n_agents, n_agents)
        directions: Unit direction vectors, shape (n_agents, n_agents, 2)
                    directions[i, j] points from i to j
    """
    n_agents = positions.shape[0]
    
    # Compute relative positions: rel_pos[i, j] = pos[j] - pos[i]
    # Shape: (n_agents, 1, 2) - (1, n_agents, 2) = (n_agents, n_agents, 2)
    rel_pos = positions[None, :, :] - positions[:, None, :]
    
    # Compute distances
    distances = jnp.linalg.norm(rel_pos, axis=-1)
    
    # Compute unit direction vectors (handle zero distance with small epsilon)
    safe_distances = jnp.maximum(distances, 1e-8)
    directions = rel_pos / safe_distances[:, :, None]
    
    return rel_pos, distances, directions


def compute_pairwise_distances_periodic(
    positions: jax.Array,
    boundary_width: float,
    boundary_height: float,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute pairwise distances with periodic boundary conditions.
    
    In periodic boundaries, the shortest distance might be across the boundary.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        boundary_width: Half-width of the boundary (arena spans [-w, w])
        boundary_height: Half-height of the boundary (arena spans [-h, h])
        
    Returns:
        rel_pos: Relative positions (shortest path), shape (n_agents, n_agents, 2)
        distances: Pairwise distances, shape (n_agents, n_agents)
        directions: Unit direction vectors, shape (n_agents, n_agents, 2)
    """
    # Compute relative positions
    rel_pos = positions[None, :, :] - positions[:, None, :]
    
    # Apply periodic wrapping to relative positions
    # If rel_pos > boundary, subtract 2*boundary (wrap around)
    # If rel_pos < -boundary, add 2*boundary
    full_width = 2.0 * boundary_width
    full_height = 2.0 * boundary_height
    
    # Wrap x coordinate
    rel_pos_x = rel_pos[:, :, 0]
    rel_pos_x = jnp.where(rel_pos_x > boundary_width, rel_pos_x - full_width, rel_pos_x)
    rel_pos_x = jnp.where(rel_pos_x < -boundary_width, rel_pos_x + full_width, rel_pos_x)
    
    # Wrap y coordinate
    rel_pos_y = rel_pos[:, :, 1]
    rel_pos_y = jnp.where(rel_pos_y > boundary_height, rel_pos_y - full_height, rel_pos_y)
    rel_pos_y = jnp.where(rel_pos_y < -boundary_height, rel_pos_y + full_height, rel_pos_y)
    
    rel_pos = jnp.stack([rel_pos_x, rel_pos_y], axis=-1)
    
    # Compute distances
    distances = jnp.linalg.norm(rel_pos, axis=-1)
    
    # Compute unit direction vectors
    safe_distances = jnp.maximum(distances, 1e-8)
    directions = rel_pos / safe_distances[:, :, None]
    
    return rel_pos, distances, directions


def compute_ball_to_ball_forces(
    positions: jax.Array,
    agent_radius: float,
    k_ball: float,
    is_periodic: bool = False,
    boundary_width: float = 2.4,
    boundary_height: float = 2.4,
) -> Tuple[jax.Array, jax.Array]:
    """Compute collision forces between all pairs of agents.
    
    Uses a spring model: when agents overlap, a repulsive force pushes them apart.
    Force magnitude is proportional to overlap distance.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        agent_radius: Radius of each agent
        k_ball: Spring stiffness for collisions
        is_periodic: Whether to use periodic boundaries
        boundary_width: Half-width of boundary (for periodic)
        boundary_height: Half-height of boundary (for periodic)
        
    Returns:
        forces: Net collision force on each agent, shape (n_agents, 2)
        is_colliding: Boolean collision matrix, shape (n_agents, n_agents)
    """
    # Get pairwise distances
    if is_periodic:
        rel_pos, distances, directions = compute_pairwise_distances_periodic(
            positions, boundary_width, boundary_height
        )
    else:
        rel_pos, distances, directions = compute_pairwise_distances(positions)
    
    n_agents = positions.shape[0]
    
    # Compute collision threshold (sum of radii for two agents)
    collision_threshold = 2.0 * agent_radius
    
    # Detect collisions (excluding self-collision via diagonal)
    is_colliding = distances < collision_threshold
    # Zero out diagonal (no self-collision)
    is_colliding = is_colliding & ~jnp.eye(n_agents, dtype=bool)
    
    # Compute overlap (penetration depth)
    overlap = collision_threshold - distances
    overlap = jnp.maximum(overlap, 0.0)  # Only positive overlap
    
    # Compute force magnitude (spring force)
    force_magnitude = k_ball * overlap
    
    # Force direction: from j to i (repulsive), so negate direction
    # directions[i, j] points from i to j, so force on i from j is -directions[i, j]
    # Force on agent i from agent j: -force_magnitude[i,j] * directions[i,j]
    pairwise_forces = -force_magnitude[:, :, None] * directions
    
    # Zero out non-colliding pairs
    pairwise_forces = jnp.where(
        is_colliding[:, :, None],
        pairwise_forces,
        jnp.zeros_like(pairwise_forces)
    )
    
    # Sum forces on each agent (sum over j for each i)
    forces = jnp.sum(pairwise_forces, axis=1)
    
    return forces, is_colliding


def compute_wall_distances(
    positions: jax.Array,
    agent_radius: float,
    boundary_left: float,
    boundary_right: float,
    boundary_bottom: float,
    boundary_top: float,
) -> Tuple[jax.Array, jax.Array]:
    """Compute distances from agents to walls.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        agent_radius: Radius of each agent
        boundary_left: x-coordinate of left wall
        boundary_right: x-coordinate of right wall
        boundary_bottom: y-coordinate of bottom wall
        boundary_top: y-coordinate of top wall
        
    Returns:
        wall_distances: Distance to each wall, shape (n_agents, 4)
                       Order: [left, right, bottom, top]
        is_colliding: Boolean, shape (n_agents, 4)
    """
    x = positions[:, 0]
    y = positions[:, 1]
    
    # Distance from agent edge to wall (negative means penetrating)
    dist_left = x - agent_radius - boundary_left
    dist_right = boundary_right - (x + agent_radius)
    dist_bottom = y - agent_radius - boundary_bottom
    dist_top = boundary_top - (y + agent_radius)
    
    wall_distances = jnp.stack([dist_left, dist_right, dist_bottom, dist_top], axis=1)
    is_colliding = wall_distances < 0
    
    return wall_distances, is_colliding


def compute_ball_to_wall_forces(
    positions: jax.Array,
    velocities: jax.Array,
    agent_radius: float,
    k_wall: float,
    c_wall: float,
    boundary_left: float,
    boundary_right: float,
    boundary_bottom: float,
    boundary_top: float,
) -> Tuple[jax.Array, jax.Array]:
    """Compute collision forces between agents and walls.
    
    Uses spring-damper model for stability.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        velocities: Agent velocities, shape (n_agents, 2)
        agent_radius: Radius of each agent
        k_wall: Spring stiffness for wall collisions
        c_wall: Damping coefficient
        boundary_*: Wall positions
        
    Returns:
        forces: Wall collision force on each agent, shape (n_agents, 2)
        is_colliding: Boolean collision with each wall, shape (n_agents, 4)
    """
    wall_distances, is_colliding = compute_wall_distances(
        positions, agent_radius,
        boundary_left, boundary_right, boundary_bottom, boundary_top
    )
    
    # Penetration depth (positive when penetrating)
    penetration = jnp.maximum(-wall_distances, 0.0)
    
    # Spring forces (push away from wall)
    # Left wall: push in +x direction
    # Right wall: push in -x direction
    # Bottom wall: push in +y direction
    # Top wall: push in -y direction
    
    vx = velocities[:, 0]
    vy = velocities[:, 1]
    
    # Spring forces
    f_left = k_wall * penetration[:, 0]
    f_right = -k_wall * penetration[:, 1]
    f_bottom = k_wall * penetration[:, 2]
    f_top = -k_wall * penetration[:, 3]
    
    # Damping forces (oppose velocity when colliding)
    d_left = -c_wall * vx * is_colliding[:, 0]
    d_right = -c_wall * vx * is_colliding[:, 1]
    d_bottom = -c_wall * vy * is_colliding[:, 2]
    d_top = -c_wall * vy * is_colliding[:, 3]
    
    # Total force in x and y
    force_x = f_left + f_right + d_left + d_right
    force_y = f_bottom + f_top + d_bottom + d_top
    
    forces = jnp.stack([force_x, force_y], axis=1)
    
    return forces, is_colliding


def apply_periodic_boundary(
    positions: jax.Array,
    boundary_width: float,
    boundary_height: float,
) -> jax.Array:
    """Wrap positions to stay within periodic boundaries.
    
    Args:
        positions: Agent positions, shape (n_agents, 2)
        boundary_width: Half-width of boundary
        boundary_height: Half-height of boundary
        
    Returns:
        wrapped_positions: Positions wrapped to [-width, width] x [-height, height]
    """
    x = positions[:, 0]
    y = positions[:, 1]
    
    # Wrap x
    x = jnp.where(x < -boundary_width, x + 2 * boundary_width, x)
    x = jnp.where(x > boundary_width, x - 2 * boundary_width, x)
    
    # Wrap y
    y = jnp.where(y < -boundary_height, y + 2 * boundary_height, y)
    y = jnp.where(y > boundary_height, y - 2 * boundary_height, y)
    
    return jnp.stack([x, y], axis=1)


def integrate_dynamics(
    positions: jax.Array,
    velocities: jax.Array,
    control_forces: jax.Array,
    collision_forces_b2b: jax.Array,
    collision_forces_b2w: jax.Array,
    physics_params: PhysicsParams,
    is_boundary: bool = True,
    boundary_width: float = 2.4,
    boundary_height: float = 2.4,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Integrate physics for one timestep.
    
    Uses Euler integration with velocity clamping.
    
    Args:
        positions: Current positions, shape (n_agents, 2)
        velocities: Current velocities, shape (n_agents, 2)
        control_forces: Forces from agent actions, shape (n_agents, 2)
        collision_forces_b2b: Ball-to-ball collision forces, shape (n_agents, 2)
        collision_forces_b2w: Ball-to-wall collision forces, shape (n_agents, 2)
        physics_params: Physics parameters
        is_boundary: If True, use wall boundaries. If False, use periodic.
        boundary_width: Half-width of arena
        boundary_height: Half-height of arena
        
    Returns:
        new_positions: Updated positions, shape (n_agents, 2)
        new_velocities: Updated velocities, shape (n_agents, 2)
        accelerations: Accelerations this step, shape (n_agents, 2)
    """
    # Total force
    if is_boundary:
        total_force = control_forces + collision_forces_b2b + collision_forces_b2w
    else:
        total_force = control_forces + collision_forces_b2b
    
    # Compute acceleration (F = ma)
    accelerations = total_force / physics_params.agent_mass
    
    # Update velocity (Euler integration)
    new_velocities = velocities + accelerations * physics_params.dt
    
    # Clamp velocity
    new_velocities = jnp.clip(
        new_velocities, 
        -physics_params.vel_max, 
        physics_params.vel_max
    )
    
    # Update position
    new_positions = positions + new_velocities * physics_params.dt
    
    # Handle boundaries
    if not is_boundary:
        # Periodic boundaries
        new_positions = apply_periodic_boundary(
            new_positions, boundary_width, boundary_height
        )
    
    return new_positions, new_velocities, accelerations


def physics_step(
    positions: jax.Array,
    velocities: jax.Array,
    actions: jax.Array,
    physics_params: PhysicsParams,
    is_boundary: bool = True,
    boundary_width: float = 2.4,
    boundary_height: float = 2.4,
    action_sensitivity: float = 1.0,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Perform one complete physics step.
    
    This is the main entry point for physics simulation.
    
    Args:
        positions: Current positions, shape (n_agents, 2)
        velocities: Current velocities, shape (n_agents, 2)
        actions: Agent actions (interpreted as force), shape (n_agents, 2)
        physics_params: Physics parameters
        is_boundary: If True, use wall boundaries. If False, use periodic.
        boundary_width: Half-width of arena
        boundary_height: Half-height of arena
        action_sensitivity: Scaling factor for actions
        
    Returns:
        new_positions: Updated positions
        new_velocities: Updated velocities
        accelerations: Accelerations this step
        b2b_collisions: Ball-to-ball collision matrix
        b2w_collisions: Ball-to-wall collision matrix
    """
    # Compute ball-to-ball collision forces
    b2b_forces, b2b_collisions = compute_ball_to_ball_forces(
        positions,
        physics_params.agent_radius,
        physics_params.k_ball,
        is_periodic=not is_boundary,
        boundary_width=boundary_width,
        boundary_height=boundary_height,
    )
    
    # Compute ball-to-wall collision forces (only if using boundaries)
    if is_boundary:
        b2w_forces, b2w_collisions = compute_ball_to_wall_forces(
            positions,
            velocities,
            physics_params.agent_radius,
            physics_params.k_wall,
            physics_params.c_wall,
            boundary_left=-boundary_width,
            boundary_right=boundary_width,
            boundary_bottom=-boundary_height,
            boundary_top=boundary_height,
        )
    else:
        b2w_forces = jnp.zeros_like(positions)
        b2w_collisions = jnp.zeros((positions.shape[0], 4), dtype=bool)
    
    # Control forces from actions
    control_forces = action_sensitivity * actions
    
    # Integrate
    new_positions, new_velocities, accelerations = integrate_dynamics(
        positions,
        velocities,
        control_forces,
        b2b_forces,
        b2w_forces,
        physics_params,
        is_boundary=is_boundary,
        boundary_width=boundary_width,
        boundary_height=boundary_height,
    )
    
    return new_positions, new_velocities, accelerations, b2b_collisions, b2w_collisions
