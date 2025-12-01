"""Tests for the physics module.

Run with: python tests/test_physics.py
"""

import jax
import jax.numpy as jnp
from jax import random

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from physics import (
    PhysicsParams,
    compute_pairwise_distances,
    compute_pairwise_distances_periodic,
    compute_ball_to_ball_forces,
    compute_wall_distances,
    compute_ball_to_wall_forces,
    apply_periodic_boundary,
    integrate_dynamics,
    physics_step,
)


def test_pairwise_distances():
    """Test pairwise distance computation."""
    print("Testing pairwise distances...")
    
    # Simple test case: 3 agents
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    
    rel_pos, distances, directions = compute_pairwise_distances(positions)
    
    # Check shapes
    assert rel_pos.shape == (3, 3, 2)
    assert distances.shape == (3, 3)
    assert directions.shape == (3, 3, 2)
    
    # Check distances
    # Distance from 0 to 1 should be 1.0
    assert jnp.isclose(distances[0, 1], 1.0)
    # Distance from 0 to 2 should be 1.0
    assert jnp.isclose(distances[0, 2], 1.0)
    # Distance from 1 to 2 should be sqrt(2)
    assert jnp.isclose(distances[1, 2], jnp.sqrt(2.0))
    # Diagonal should be 0
    assert jnp.allclose(jnp.diag(distances), 0.0)
    
    # Check symmetry
    assert jnp.allclose(distances, distances.T)
    
    print("  ✓ Pairwise distances tests passed")


def test_pairwise_distances_periodic():
    """Test pairwise distances with periodic boundaries."""
    print("Testing periodic pairwise distances...")
    
    boundary_width = 2.0
    boundary_height = 2.0
    
    # Two agents at opposite edges - should be close in periodic space
    positions = jnp.array([
        [-1.9, 0.0],  # Near left edge
        [1.9, 0.0],   # Near right edge
    ])
    
    # Non-periodic distance
    _, distances_nonperiodic, _ = compute_pairwise_distances(positions)
    
    # Periodic distance
    _, distances_periodic, _ = compute_pairwise_distances_periodic(
        positions, boundary_width, boundary_height
    )
    
    # Non-periodic: distance is 3.8
    assert jnp.isclose(distances_nonperiodic[0, 1], 3.8)
    
    # Periodic: distance should be 0.2 (going through boundary)
    assert jnp.isclose(distances_periodic[0, 1], 0.2, atol=1e-5), \
        f"Expected ~0.2, got {distances_periodic[0, 1]}"
    
    print("  ✓ Periodic pairwise distances tests passed")


def test_ball_to_ball_forces_no_collision():
    """Test that distant agents have no collision force."""
    print("Testing ball-to-ball forces (no collision)...")
    
    # Two agents far apart
    positions = jnp.array([
        [0.0, 0.0],
        [2.0, 0.0],  # Far from first agent
    ])
    
    agent_radius = 0.1
    k_ball = 30.0
    
    forces, is_colliding = compute_ball_to_ball_forces(
        positions, agent_radius, k_ball
    )
    
    # No collision
    assert not jnp.any(is_colliding), "Should have no collisions"
    
    # Zero forces
    assert jnp.allclose(forces, 0.0), "Forces should be zero"
    
    print("  ✓ Ball-to-ball forces (no collision) tests passed")


def test_ball_to_ball_forces_collision():
    """Test collision forces between overlapping agents."""
    print("Testing ball-to-ball forces (with collision)...")
    
    # Two agents overlapping
    agent_radius = 0.1
    positions = jnp.array([
        [0.0, 0.0],
        [0.15, 0.0],  # Overlapping (distance < 2*radius = 0.2)
    ])
    
    k_ball = 30.0
    
    forces, is_colliding = compute_ball_to_ball_forces(
        positions, agent_radius, k_ball
    )
    
    # Should detect collision
    assert is_colliding[0, 1], "Should detect collision between 0 and 1"
    assert is_colliding[1, 0], "Should detect collision between 1 and 0"
    
    # Forces should push apart
    # Agent 0 should be pushed in -x direction (away from agent 1)
    assert forces[0, 0] < 0, f"Agent 0 should be pushed in -x, got {forces[0]}"
    # Agent 1 should be pushed in +x direction
    assert forces[1, 0] > 0, f"Agent 1 should be pushed in +x, got {forces[1]}"
    
    # Forces should be equal and opposite (Newton's 3rd law)
    assert jnp.allclose(forces[0], -forces[1]), "Forces should be equal and opposite"
    
    # Y forces should be zero (collision along x-axis)
    assert jnp.allclose(forces[:, 1], 0.0), "Y forces should be zero"
    
    print("  ✓ Ball-to-ball forces (with collision) tests passed")


def test_ball_to_ball_forces_multiple_agents():
    """Test collision forces with multiple agents."""
    print("Testing ball-to-ball forces (multiple agents)...")
    
    agent_radius = 0.1
    # Agent 0 in center, agents 1,2 colliding with it from different sides
    positions = jnp.array([
        [0.0, 0.0],    # Center
        [0.15, 0.0],   # Right, colliding
        [-0.15, 0.0],  # Left, colliding
        [0.0, 1.0],    # Far away, not colliding
    ])
    
    k_ball = 30.0
    
    forces, is_colliding = compute_ball_to_ball_forces(
        positions, agent_radius, k_ball
    )
    
    # Agent 0 collides with 1 and 2
    assert is_colliding[0, 1] and is_colliding[0, 2]
    # Agent 3 doesn't collide with anyone
    assert not jnp.any(is_colliding[3, :])
    
    # Agent 0 forces from 1 and 2 should roughly cancel (symmetric setup)
    # But not exactly because they're slightly different distances
    assert jnp.abs(forces[0, 0]) < 0.1, "Agent 0 x-force should nearly cancel"
    
    print("  ✓ Ball-to-ball forces (multiple agents) tests passed")


def test_wall_distances():
    """Test wall distance computation."""
    print("Testing wall distances...")
    
    positions = jnp.array([
        [0.0, 0.0],    # Center
        [2.3, 0.0],    # Near right wall
        [-2.3, 0.0],   # Near left wall
    ])
    
    agent_radius = 0.1
    
    wall_distances, is_colliding = compute_wall_distances(
        positions, agent_radius,
        boundary_left=-2.4,
        boundary_right=2.4,
        boundary_bottom=-2.4,
        boundary_top=2.4,
    )
    
    # Center agent should not collide with any wall
    assert not jnp.any(is_colliding[0]), "Center agent shouldn't collide"
    
    # Agent near right wall (at x=2.3) with radius 0.1
    # Right edge at 2.4, agent edge at 2.4, distance = 0
    # Actually: agent at 2.3, edge at 2.3+0.1=2.4, wall at 2.4
    # Distance to right wall = 2.4 - (2.3 + 0.1) = 0
    assert wall_distances[1, 1] <= 0.01, f"Should be at/past right wall, got {wall_distances[1, 1]}"
    
    print("  ✓ Wall distances tests passed")


def test_ball_to_wall_forces():
    """Test wall collision forces."""
    print("Testing ball-to-wall forces...")
    
    # Agent penetrating right wall
    positions = jnp.array([
        [2.35, 0.0],  # Past right wall (wall at 2.4, radius 0.1)
    ])
    velocities = jnp.array([
        [1.0, 0.0],   # Moving toward wall
    ])
    
    agent_radius = 0.1
    k_wall = 100.0
    c_wall = 5.0
    
    forces, is_colliding = compute_ball_to_wall_forces(
        positions, velocities, agent_radius, k_wall, c_wall,
        boundary_left=-2.4,
        boundary_right=2.4,
        boundary_bottom=-2.4,
        boundary_top=2.4,
    )
    
    # Should collide with right wall
    assert is_colliding[0, 1], "Should collide with right wall"
    
    # Force should push left (negative x)
    assert forces[0, 0] < 0, f"Force should push left, got {forces[0, 0]}"
    
    # Y force should be zero
    assert jnp.isclose(forces[0, 1], 0.0)
    
    print("  ✓ Ball-to-wall forces tests passed")


def test_periodic_boundary():
    """Test periodic boundary wrapping."""
    print("Testing periodic boundary...")
    
    boundary_width = 2.0
    boundary_height = 2.0
    
    # Positions outside boundaries
    positions = jnp.array([
        [2.5, 0.0],   # Past right boundary
        [-2.5, 0.0],  # Past left boundary
        [0.0, 2.5],   # Past top boundary
        [0.0, -2.5],  # Past bottom boundary
    ])
    
    wrapped = apply_periodic_boundary(positions, boundary_width, boundary_height)
    
    # Check all positions are within bounds
    assert jnp.all(wrapped[:, 0] >= -boundary_width)
    assert jnp.all(wrapped[:, 0] <= boundary_width)
    assert jnp.all(wrapped[:, 1] >= -boundary_height)
    assert jnp.all(wrapped[:, 1] <= boundary_height)
    
    # Check specific wrapping
    # 2.5 should wrap to 2.5 - 4.0 = -1.5
    assert jnp.isclose(wrapped[0, 0], -1.5)
    # -2.5 should wrap to -2.5 + 4.0 = 1.5
    assert jnp.isclose(wrapped[1, 0], 1.5)
    
    print("  ✓ Periodic boundary tests passed")


def test_physics_step_basic():
    """Test complete physics step."""
    print("Testing physics step (basic)...")
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    velocities = jnp.array([
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    actions = jnp.array([
        [1.0, 0.0],  # Push right
        [-1.0, 0.0], # Push left
    ])
    
    params = PhysicsParams()
    
    new_pos, new_vel, accel, b2b_col, b2w_col = physics_step(
        positions, velocities, actions, params,
        is_boundary=True,
        boundary_width=2.4,
        boundary_height=2.4,
    )
    
    # Agents should have moved
    assert new_pos[0, 0] > 0.0, "Agent 0 should move right"
    assert new_pos[1, 0] < 1.0, "Agent 1 should move left"
    
    # Velocities should have changed
    assert new_vel[0, 0] > 0.0
    assert new_vel[1, 0] < 0.0
    
    print("  ✓ Physics step (basic) tests passed")


def test_physics_jit_compatibility():
    """Test that physics functions work with JIT."""
    print("Testing physics JIT compatibility...")
    
    params = PhysicsParams()
    
    @jax.jit
    def step_fn(positions, velocities, actions):
        return physics_step(
            positions, velocities, actions, params,
            is_boundary=True,
            boundary_width=2.4,
            boundary_height=2.4,
        )
    
    n_agents = 10
    key = random.PRNGKey(0)
    
    positions = random.uniform(key, (n_agents, 2), minval=-2.0, maxval=2.0)
    velocities = jnp.zeros((n_agents, 2))
    actions = random.uniform(key, (n_agents, 2), minval=-1.0, maxval=1.0)
    
    # Should compile and run without error
    new_pos, new_vel, accel, b2b_col, b2w_col = step_fn(positions, velocities, actions)
    
    assert new_pos.shape == (n_agents, 2)
    assert new_vel.shape == (n_agents, 2)
    assert b2b_col.shape == (n_agents, n_agents)
    
    print("  ✓ Physics JIT compatibility tests passed")


def test_physics_vmap_compatibility():
    """Test that physics functions work with vmap (batched envs)."""
    print("Testing physics vmap compatibility...")
    
    n_envs = 4
    n_agents = 5
    params = PhysicsParams()
    
    @jax.jit
    def batched_step(positions, velocities, actions):
        return jax.vmap(
            lambda p, v, a: physics_step(p, v, a, params, is_boundary=True)
        )(positions, velocities, actions)
    
    key = random.PRNGKey(42)
    
    positions = random.uniform(key, (n_envs, n_agents, 2), minval=-2.0, maxval=2.0)
    velocities = jnp.zeros((n_envs, n_agents, 2))
    actions = random.uniform(key, (n_envs, n_agents, 2), minval=-1.0, maxval=1.0)
    
    new_pos, new_vel, accel, b2b_col, b2w_col = batched_step(positions, velocities, actions)
    
    assert new_pos.shape == (n_envs, n_agents, 2)
    assert new_vel.shape == (n_envs, n_agents, 2)
    assert b2b_col.shape == (n_envs, n_agents, n_agents)
    
    print("  ✓ Physics vmap compatibility tests passed")


def test_collision_conservation():
    """Test that collision forces conserve momentum."""
    print("Testing collision momentum conservation...")
    
    # Two equal mass agents colliding
    positions = jnp.array([
        [0.0, 0.0],
        [0.1, 0.0],  # Very close, will collide
    ])
    
    agent_radius = 0.1
    k_ball = 30.0
    
    forces, _ = compute_ball_to_ball_forces(positions, agent_radius, k_ball)
    
    # Total force should be zero (momentum conservation)
    total_force = jnp.sum(forces, axis=0)
    assert jnp.allclose(total_force, 0.0, atol=1e-6), \
        f"Total force should be zero, got {total_force}"
    
    print("  ✓ Collision momentum conservation tests passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running physics module tests")
    print("="*60 + "\n")
    
    test_pairwise_distances()
    test_pairwise_distances_periodic()
    test_ball_to_ball_forces_no_collision()
    test_ball_to_ball_forces_collision()
    test_ball_to_ball_forces_multiple_agents()
    test_wall_distances()
    test_ball_to_wall_forces()
    test_periodic_boundary()
    test_physics_step_basic()
    test_physics_jit_compatibility()
    test_physics_vmap_compatibility()
    test_collision_conservation()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")
