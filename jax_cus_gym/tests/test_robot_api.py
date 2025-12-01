"""Tests for the Robot API module."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random

from robot_api import (
    get_position_and_velocity,
    get_neighbor_ids,
    get_neighbor_info,
    is_within_target,
    get_target_position,
    get_target_info,
    get_unoccupied_cells,
    get_all_in_target,
    get_all_target_positions,
    get_all_neighbor_ids,
    get_all_collisions,
    RobotAPI,
)


def test_get_position_and_velocity():
    """Test single agent position/velocity retrieval."""
    print("Testing get_position_and_velocity...")
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
    ])
    velocities = jnp.array([
        [0.1, 0.0],
        [0.0, 0.1],
        [-0.1, 0.1],
    ])
    
    pos, vel = get_position_and_velocity(positions, velocities, 1)
    
    assert jnp.allclose(pos, jnp.array([1.0, 1.0])), f"Expected [1, 1], got {pos}"
    assert jnp.allclose(vel, jnp.array([0.0, 0.1])), f"Expected [0, 0.1], got {vel}"
    
    print("  ✓ Single agent position/velocity correct")


def test_get_neighbor_ids():
    """Test neighbor ID retrieval."""
    print("Testing get_neighbor_ids...")
    
    # 4 agents in a line: 0--1--2--3
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])
    
    # Agent 1 with d_sen=1.5, k_max=2 should see agents 0 and 2
    neighbor_ids = get_neighbor_ids(positions, agent_id=1, d_sen=1.5, k_max=2)
    
    # Should have 2 neighbors (0 and 2)
    valid_neighbors = neighbor_ids[neighbor_ids >= 0]
    assert len(valid_neighbors) == 2, f"Expected 2 neighbors, got {len(valid_neighbors)}"
    assert 0 in valid_neighbors, "Agent 0 should be a neighbor"
    assert 2 in valid_neighbors, "Agent 2 should be a neighbor"
    
    print("  ✓ Neighbor IDs correct")


def test_get_neighbor_info():
    """Test full neighbor info retrieval."""
    print("Testing get_neighbor_info...")
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])
    velocities = jnp.array([
        [0.1, 0.0],
        [0.0, 0.0],
        [-0.1, 0.0],
    ])
    
    info = get_neighbor_info(positions, velocities, agent_id=1, d_sen=1.5, k_max=3)
    
    assert info.count == 2, f"Expected 2 neighbors, got {info.count}"
    assert jnp.all(info.distances[:2] < 1.5), "Neighbors should be within sensing range"
    
    print("  ✓ Neighbor info correct")


def test_is_within_target():
    """Test target region detection."""
    print("Testing is_within_target...")
    
    positions = jnp.array([
        [0.0, 0.0],   # In target
        [5.0, 5.0],   # Far from target
    ])
    grid_centers = jnp.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
    ])
    grid_mask = jnp.array([True, True, True])
    l_cell = 0.2
    
    in_target_0 = is_within_target(positions, grid_centers, grid_mask, l_cell, agent_id=0)
    in_target_1 = is_within_target(positions, grid_centers, grid_mask, l_cell, agent_id=1)
    
    assert in_target_0, "Agent 0 should be in target"
    assert not in_target_1, "Agent 1 should NOT be in target"
    
    print("  ✓ Target detection correct")


def test_get_target_position():
    """Test nearest target position."""
    print("Testing get_target_position...")
    
    positions = jnp.array([
        [0.5, 0.5],
    ])
    grid_centers = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    grid_mask = jnp.array([True, True, True, True])
    
    target_pos = get_target_position(positions, grid_centers, grid_mask, agent_id=0)
    
    # Agent at (0.5, 0.5) is equidistant to all corners
    # Just verify it's one of the grid centers
    distances = jnp.linalg.norm(grid_centers - target_pos, axis=-1)
    assert jnp.min(distances) < 0.01, "Target should be a grid center"
    
    print("  ✓ Target position correct")


def test_get_all_in_target():
    """Test vectorized target detection."""
    print("Testing get_all_in_target...")
    
    positions = jnp.array([
        [0.0, 0.0],   # In target
        [0.05, 0.0],  # In target
        [5.0, 5.0],   # Far
    ])
    grid_centers = jnp.array([
        [0.0, 0.0],
    ])
    grid_mask = jnp.array([True])
    l_cell = 0.2
    
    in_target = get_all_in_target(positions, grid_centers, grid_mask, l_cell)
    
    assert in_target[0], "Agent 0 should be in target"
    assert in_target[1], "Agent 1 should be in target"
    assert not in_target[2], "Agent 2 should NOT be in target"
    
    print("  ✓ Vectorized target detection correct")


def test_get_all_neighbor_ids():
    """Test vectorized neighbor retrieval."""
    print("Testing get_all_neighbor_ids...")
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [10.0, 0.0],  # Far away
    ])
    
    neighbor_ids = get_all_neighbor_ids(positions, d_sen=1.5, k_max=2)
    
    assert neighbor_ids.shape == (4, 2), f"Expected shape (4, 2), got {neighbor_ids.shape}"
    
    # Agent 0 should have agent 1 as neighbor
    assert 1 in neighbor_ids[0], "Agent 0 should have agent 1 as neighbor"
    
    # Agent 3 should have no neighbors (too far)
    assert jnp.all(neighbor_ids[3] == -1), "Agent 3 should have no neighbors"
    
    print("  ✓ Vectorized neighbor IDs correct")


def test_get_all_collisions():
    """Test collision detection."""
    print("Testing get_all_collisions...")
    
    positions = jnp.array([
        [0.0, 0.0],
        [0.1, 0.0],  # Colliding with agent 0
        [5.0, 0.0],  # Far away
    ])
    
    is_colliding = get_all_collisions(positions, collision_threshold=0.2)
    
    assert is_colliding[0], "Agent 0 should be colliding"
    assert is_colliding[1], "Agent 1 should be colliding"
    assert not is_colliding[2], "Agent 2 should NOT be colliding"
    
    print("  ✓ Collision detection correct")


def test_robot_api_class():
    """Test the RobotAPI wrapper class."""
    print("Testing RobotAPI class...")
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])
    velocities = jnp.array([
        [0.1, 0.0],
        [0.0, 0.1],
        [-0.1, 0.0],
    ])
    grid_centers = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    grid_mask = jnp.array([True, True])
    
    api = RobotAPI(
        positions=positions,
        velocities=velocities,
        grid_centers=grid_centers,
        grid_mask=grid_mask,
        l_cell=0.2,
        d_sen=1.5,
        k_max=2,
        r_avoid=0.15,
    )
    
    # Test methods
    pos, vel = api.get_position_and_velocity(0)
    assert jnp.allclose(pos, jnp.array([0.0, 0.0])), "Position incorrect"
    
    neighbors = api.get_neighbor_ids(1)
    assert 0 in neighbors or 2 in neighbors, "Should have neighbors"
    
    in_target = api.is_within_target(0)
    assert in_target, "Agent 0 should be in target"
    
    all_in_target = api.get_all_in_target()
    assert all_in_target[0], "Agent 0 should be in target (vectorized)"
    
    print("  ✓ RobotAPI class works correctly")


def test_jit_compatibility():
    """Test that functions can be JIT compiled."""
    print("Testing JIT compatibility...")
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])
    velocities = jnp.zeros((3, 2))
    grid_centers = jnp.array([[0.0, 0.0]])
    grid_mask = jnp.array([True])
    
    # JIT compile and run
    jit_get_neighbors = jax.jit(lambda p: get_all_neighbor_ids(p, 1.5, 2))
    result = jit_get_neighbors(positions)
    assert result.shape == (3, 2), "JIT neighbor IDs shape incorrect"
    
    jit_get_in_target = jax.jit(lambda p: get_all_in_target(p, grid_centers, grid_mask, 0.2))
    result = jit_get_in_target(positions)
    assert result.shape == (3,), "JIT in_target shape incorrect"
    
    print("  ✓ JIT compilation works")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Robot API Tests")
    print("="*60 + "\n")
    
    tests = [
        test_get_position_and_velocity,
        test_get_neighbor_ids,
        test_get_neighbor_info,
        test_is_within_target,
        test_get_target_position,
        test_get_all_in_target,
        test_get_all_neighbor_ids,
        test_get_all_collisions,
        test_robot_api_class,
        test_jit_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
