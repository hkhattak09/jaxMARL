"""Tests for the rewards module.

Run with: python tests/test_rewards.py
"""

import jax
import jax.numpy as jnp
from jax import random

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rewards import (
    RewardParams,
    compute_in_target,
    compute_agent_collisions,
    compute_exploration_reward,
    compute_rewards,
    compute_simple_rewards,
    compute_coverage_rate,
    compute_average_distance_to_target,
    compute_collision_count,
)


def test_in_target_detection():
    """Test detection of agents in target region."""
    print("Testing in-target detection...")
    
    # Grid at origin
    grid_centers = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    l_cell = 0.5
    
    # Threshold is sqrt(2) * 0.5 / 2 ≈ 0.354
    positions = jnp.array([
        [0.0, 0.0],    # Exactly on grid center - IN
        [0.2, 0.2],    # Close to center (dist=0.28) - IN
        [0.5, 0.0],    # Between cells (dist=0.5 to nearest) - OUT
        [1.0, 0.0],    # On second grid center - IN
    ])
    
    in_target = compute_in_target(positions, grid_centers, l_cell)
    
    assert in_target[0], "Agent at origin should be in target"
    assert in_target[1], "Agent close to center should be in target"
    assert not in_target[2], "Agent between cells should not be in target"
    assert in_target[3], "Agent at second center should be in target"
    
    print("  ✓ In-target detection tests passed")


def test_collision_detection():
    """Test collision detection between agents."""
    print("Testing collision detection...")
    
    collision_threshold = 0.2
    
    positions = jnp.array([
        [0.0, 0.0],
        [0.1, 0.0],    # Colliding with agent 0 (dist=0.1 < 0.2)
        [1.0, 0.0],    # Not colliding (dist=1.0 > 0.2)
        [1.1, 0.0],    # Colliding with agent 2 (dist=0.1 < 0.2)
    ])
    
    is_colliding, collision_matrix = compute_agent_collisions(
        positions, collision_threshold
    )
    
    # Agents 0 and 1 should be colliding
    assert is_colliding[0], "Agent 0 should be colliding"
    assert is_colliding[1], "Agent 1 should be colliding"
    
    # Agents 2 and 3 should be colliding
    assert is_colliding[2], "Agent 2 should be colliding"
    assert is_colliding[3], "Agent 3 should be colliding"
    
    # Check collision matrix
    assert collision_matrix[0, 1], "Collision matrix [0,1] should be True"
    assert collision_matrix[1, 0], "Collision matrix [1,0] should be True"
    assert not collision_matrix[0, 2], "Collision matrix [0,2] should be False"
    
    print("  ✓ Collision detection tests passed")


def test_no_collisions():
    """Test when no agents are colliding."""
    print("Testing no collisions...")
    
    collision_threshold = 0.2
    
    # All agents far apart
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])
    
    is_colliding, collision_matrix = compute_agent_collisions(
        positions, collision_threshold
    )
    
    assert not jnp.any(is_colliding), "No agents should be colliding"
    # Diagonal should be False (no self-collision)
    assert not jnp.any(jnp.diag(collision_matrix)), "Diagonal should be False"
    
    print("  ✓ No collisions tests passed")


def test_simple_rewards():
    """Test simplified reward computation."""
    print("Testing simple rewards...")
    
    grid_centers = jnp.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
    ])
    l_cell = 0.3
    collision_threshold = 0.15
    
    positions = jnp.array([
        [0.0, 0.0],    # In target, no collision -> reward
        [0.5, 0.0],    # In target, no collision -> reward
        [0.55, 0.0],   # In target, colliding with agent 1 -> no reward
        [2.0, 0.0],    # Not in target -> no reward
    ])
    
    rewards = compute_simple_rewards(
        positions, grid_centers, l_cell, collision_threshold
    )
    
    # Agent 0: in target, not colliding -> 1.0
    assert rewards[0] == 1.0, f"Agent 0 should get reward 1.0, got {rewards[0]}"
    
    # Agent 1: in target, colliding with agent 2 -> 0.0
    # Actually distance between agent 1 and 2 is 0.05 < 0.15, so collision
    assert rewards[1] == 0.0, f"Agent 1 should get reward 0.0 (collision), got {rewards[1]}"
    
    # Agent 2: same as agent 1
    assert rewards[2] == 0.0, f"Agent 2 should get reward 0.0 (collision), got {rewards[2]}"
    
    # Agent 3: not in target
    assert rewards[3] == 0.0, f"Agent 3 should get reward 0.0 (not in target), got {rewards[3]}"
    
    print("  ✓ Simple rewards tests passed")


def test_full_rewards():
    """Test full reward computation."""
    print("Testing full rewards...")
    
    grid_centers = jnp.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
    ])
    l_cell = 0.3
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],  # Far from grid
    ])
    velocities = jnp.zeros((3, 2))
    
    neighbor_indices = jnp.array([
        [-1, -1],
        [-1, -1],
        [-1, -1],
    ])
    
    reward_params = RewardParams(
        reward_entering=1.0,
        penalty_collision=-0.5,
        collision_threshold=0.15,
    )
    
    rewards, info = compute_rewards(
        positions, velocities, grid_centers, l_cell,
        neighbor_indices, reward_params, d_sen=2.0
    )
    
    assert rewards.shape == (3,), f"Expected shape (3,), got {rewards.shape}"
    assert "in_target" in info
    assert "is_colliding" in info
    
    print("  ✓ Full rewards tests passed")


def test_reward_sharing_mean():
    """Test mean reward sharing."""
    print("Testing reward sharing (mean)...")
    
    grid_centers = jnp.array([[0.0, 0.0]])
    l_cell = 0.5
    
    # One agent in target, one out
    positions = jnp.array([
        [0.0, 0.0],  # In target
        [5.0, 0.0],  # Far out
    ])
    velocities = jnp.zeros((2, 2))
    neighbor_indices = jnp.array([[-1], [-1]])
    
    reward_params = RewardParams(
        reward_entering=1.0,
        penalty_collision=0.0,
        collision_threshold=0.1,
        reward_mode="shared_mean",
    )
    
    rewards, _ = compute_rewards(
        positions, velocities, grid_centers, l_cell,
        neighbor_indices, reward_params, d_sen=0.5
    )
    
    # With mean sharing, both agents should have same reward
    assert jnp.allclose(rewards[0], rewards[1]), "Mean sharing should give equal rewards"
    
    print("  ✓ Reward sharing (mean) tests passed")


def test_reward_sharing_max():
    """Test max reward sharing."""
    print("Testing reward sharing (max)...")
    
    grid_centers = jnp.array([[0.0, 0.0]])
    l_cell = 0.5
    
    positions = jnp.array([
        [0.0, 0.0],  # In target
        [5.0, 0.0],  # Far out
    ])
    velocities = jnp.zeros((2, 2))
    neighbor_indices = jnp.array([[-1], [-1]])
    
    reward_params = RewardParams(
        reward_entering=1.0,
        penalty_collision=0.0,
        collision_threshold=0.1,
        reward_mode="shared_max",
    )
    
    rewards, _ = compute_rewards(
        positions, velocities, grid_centers, l_cell,
        neighbor_indices, reward_params, d_sen=0.5
    )
    
    # With max sharing, both agents should have the max reward
    assert jnp.allclose(rewards[0], rewards[1]), "Max sharing should give equal rewards"
    
    print("  ✓ Reward sharing (max) tests passed")


def test_coverage_rate():
    """Test coverage rate metric."""
    print("Testing coverage rate...")
    
    grid_centers = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])
    l_cell = 0.5
    
    # 2 agents covering 2 cells
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    
    coverage = compute_coverage_rate(positions, grid_centers, l_cell)
    
    # 2 out of 4 cells covered = 0.5
    assert jnp.isclose(coverage, 0.5), f"Expected coverage 0.5, got {coverage}"
    
    # Full coverage
    positions_full = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])
    
    coverage_full = compute_coverage_rate(positions_full, grid_centers, l_cell)
    assert jnp.isclose(coverage_full, 1.0), f"Expected coverage 1.0, got {coverage_full}"
    
    print("  ✓ Coverage rate tests passed")


def test_average_distance():
    """Test average distance to target metric."""
    print("Testing average distance to target...")
    
    grid_centers = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    
    # Agent at distance 0.5 from nearest grid cell
    positions = jnp.array([
        [0.5, 0.0],  # Distance to [0,0] = 0.5, to [1,0] = 0.5
    ])
    
    avg_dist = compute_average_distance_to_target(positions, grid_centers)
    assert jnp.isclose(avg_dist, 0.5), f"Expected avg distance 0.5, got {avg_dist}"
    
    print("  ✓ Average distance to target tests passed")


def test_collision_count():
    """Test collision count metric."""
    print("Testing collision count...")
    
    collision_threshold = 0.2
    
    # 2 collision pairs: (0,1) and (2,3)
    positions = jnp.array([
        [0.0, 0.0],
        [0.1, 0.0],   # Collides with 0
        [1.0, 0.0],
        [1.1, 0.0],   # Collides with 2
    ])
    
    count = compute_collision_count(positions, collision_threshold)
    assert count == 2, f"Expected 2 collision pairs, got {count}"
    
    print("  ✓ Collision count tests passed")


def test_rewards_jit():
    """Test that rewards work with JIT."""
    print("Testing rewards JIT compatibility...")
    
    reward_params = RewardParams()
    
    @jax.jit
    def compute_rewards_jit(positions, velocities, grid_centers, l_cell, neighbor_indices):
        return compute_rewards(
            positions, velocities, grid_centers, l_cell,
            neighbor_indices, reward_params
        )
    
    n_agents = 5
    n_grid = 10
    
    key = random.PRNGKey(0)
    positions = random.uniform(key, (n_agents, 2), minval=-2.0, maxval=2.0)
    velocities = random.uniform(key, (n_agents, 2), minval=-1.0, maxval=1.0)
    grid_centers = random.uniform(key, (n_grid, 2), minval=-1.0, maxval=1.0)
    neighbor_indices = -jnp.ones((n_agents, 3), dtype=jnp.int32)
    l_cell = 0.3
    
    rewards, info = compute_rewards_jit(
        positions, velocities, grid_centers, l_cell, neighbor_indices
    )
    
    assert rewards.shape == (n_agents,)
    
    print("  ✓ Rewards JIT compatibility tests passed")


def test_rewards_vmap():
    """Test that rewards work with vmap."""
    print("Testing rewards vmap compatibility...")
    
    n_envs = 4
    n_agents = 5
    n_grid = 8
    
    reward_params = RewardParams()
    
    @jax.jit
    def batch_rewards(positions, velocities, grid_centers, l_cell, neighbor_indices):
        return jax.vmap(
            lambda p, v, g, ni: compute_rewards(p, v, g, l_cell, ni, reward_params)
        )(positions, velocities, grid_centers, neighbor_indices)
    
    key = random.PRNGKey(42)
    positions = random.uniform(key, (n_envs, n_agents, 2), minval=-2.0, maxval=2.0)
    velocities = random.uniform(key, (n_envs, n_agents, 2), minval=-1.0, maxval=1.0)
    grid_centers = random.uniform(key, (n_envs, n_grid, 2), minval=-1.0, maxval=1.0)
    neighbor_indices = -jnp.ones((n_envs, n_agents, 3), dtype=jnp.int32)
    l_cell = 0.3
    
    rewards, info = batch_rewards(
        positions, velocities, grid_centers, l_cell, neighbor_indices
    )
    
    assert rewards.shape == (n_envs, n_agents)
    
    print("  ✓ Rewards vmap compatibility tests passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running rewards module tests")
    print("="*60 + "\n")
    
    test_in_target_detection()
    test_collision_detection()
    test_no_collisions()
    test_simple_rewards()
    test_full_rewards()
    test_reward_sharing_mean()
    test_reward_sharing_max()
    test_coverage_rate()
    test_average_distance()
    test_collision_count()
    test_rewards_jit()
    test_rewards_vmap()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")
