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
    rho_cos_dec,
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
        reward_mode=1,  # REWARD_MODE_SHARED_MEAN
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
        reward_mode=2,  # REWARD_MODE_SHARED_MAX
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


# ============================================================
# NEW TESTS FOR C++ MARL MATCHING
# ============================================================

def test_rho_cos_dec_basic():
    """Test cosine decay weighting function basic behavior.
    
    This function matches the C++ _rho_cos_dec implementation:
    - Returns 1.0 when z < delta * r (close range)
    - Smoothly decays when delta * r <= z < r
    - Returns 0.0 when z >= r (far away)
    """
    print("Testing rho_cos_dec basic behavior...")
    
    r = 1.0
    delta = 0.5
    
    # Test at key points
    z = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    weights = rho_cos_dec(z, r, delta)
    
    # z=0.0: well within delta*r=0.5, should be 1.0
    assert jnp.isclose(weights[0], 1.0), f"Expected 1.0 at z=0, got {weights[0]}"
    
    # z=0.25: still within delta*r=0.5, should be 1.0
    assert jnp.isclose(weights[1], 1.0), f"Expected 1.0 at z=0.25, got {weights[1]}"
    
    # z=0.5: exactly at delta*r, should be 1.0 (boundary of close range)
    assert jnp.isclose(weights[2], 1.0), f"Expected 1.0 at z=0.5, got {weights[2]}"
    
    # z=0.75: in decay zone, should be ~0.5 (midpoint of cosine decay)
    assert 0.4 < weights[3] < 0.6, f"Expected ~0.5 at z=0.75, got {weights[3]}"
    
    # z=1.0: at boundary, should be 0.0
    assert jnp.isclose(weights[4], 0.0, atol=1e-5), f"Expected 0.0 at z=1.0, got {weights[4]}"
    
    # z=1.5: beyond range, should be 0.0
    assert jnp.isclose(weights[5], 0.0), f"Expected 0.0 at z=1.5, got {weights[5]}"
    
    print("  ✓ rho_cos_dec basic behavior tests passed")


def test_rho_cos_dec_different_params():
    """Test rho_cos_dec with different r and delta values."""
    print("Testing rho_cos_dec with different parameters...")
    
    # Test with larger range
    r = 3.0
    delta = 0.5
    
    # z=1.0 should be in close range (< delta*r = 1.5)
    assert jnp.isclose(rho_cos_dec(jnp.array([1.0]), r, delta)[0], 1.0)
    
    # z=2.0 should be in decay zone (1.5 <= z < 3.0)
    weight_at_2 = rho_cos_dec(jnp.array([2.0]), r, delta)[0]
    assert 0.0 < weight_at_2 < 1.0, f"Expected decay at z=2.0, got {weight_at_2}"
    
    # Test with different delta
    delta = 0.8  # Larger close range
    r = 1.0
    
    # z=0.7 should still be in close range (< 0.8)
    assert jnp.isclose(rho_cos_dec(jnp.array([0.7]), r, delta)[0], 1.0)
    
    # z=0.9 should be in decay zone
    weight_at_09 = rho_cos_dec(jnp.array([0.9]), r, delta)[0]
    assert 0.0 < weight_at_09 < 1.0
    
    print("  ✓ rho_cos_dec different parameters tests passed")


def test_rho_cos_dec_vectorized():
    """Test that rho_cos_dec works with vectorized inputs."""
    print("Testing rho_cos_dec vectorization...")
    
    r = 2.0
    delta = 0.5
    
    # Test with 2D array (like distances from n_agents to n_grids)
    z = jnp.array([
        [0.5, 1.0, 1.5, 2.0, 2.5],
        [0.0, 0.5, 1.0, 1.5, 3.0],
    ])  # (2, 5)
    
    weights = rho_cos_dec(z, r, delta)
    
    assert weights.shape == (2, 5), f"Expected shape (2, 5), got {weights.shape}"
    
    # Check specific values
    assert jnp.isclose(weights[0, 0], 1.0)  # z=0.5 < delta*r=1.0
    assert jnp.isclose(weights[0, 3], 0.0, atol=1e-5)  # z=2.0 = r
    assert jnp.isclose(weights[1, 0], 1.0)  # z=0.0
    assert jnp.isclose(weights[1, 4], 0.0)  # z=3.0 > r
    
    print("  ✓ rho_cos_dec vectorization tests passed")


def test_exploration_reward_symmetric_grids():
    """Test exploration reward with symmetric grid arrangement.
    
    When an agent is at the center of symmetrically arranged grids,
    the weighted centroid of relative positions should be ~0,
    giving a reward of 1.0.
    """
    print("Testing exploration reward with symmetric grids...")
    
    # Agent at origin with 4 symmetric grids around it
    positions = jnp.array([[0.0, 0.0]])  # 1 agent at origin
    grid_centers = jnp.array([
        [0.5, 0.0],   # right
        [-0.5, 0.0],  # left  
        [0.0, 0.5],   # up
        [0.0, -0.5],  # down
    ])
    in_target = jnp.array([True])
    neighbor_indices = jnp.zeros((1, 1), dtype=jnp.int32)
    
    reward = compute_exploration_reward(
        positions, grid_centers, in_target, neighbor_indices,
        collision_threshold=0.15, exploration_threshold=0.05,
        d_sen=3.0, cosine_decay_delta=0.5
    )
    
    # Agent is centered, so weighted centroid should be ~0
    # Therefore reward should be 1.0
    assert reward[0] == 1.0, f"Expected reward 1.0 for centered agent, got {reward[0]}"
    
    print("  ✓ Exploration reward symmetric grids test passed")


def test_exploration_reward_asymmetric_grids():
    """Test exploration reward with asymmetric grid arrangement.
    
    When grids are all on one side, the weighted centroid of relative
    positions will have large norm, giving reward 0.0.
    """
    print("Testing exploration reward with asymmetric grids...")
    
    # Agent at origin with grids only to the right
    positions = jnp.array([[0.0, 0.0]])
    grid_centers = jnp.array([
        [0.5, 0.0],
        [1.0, 0.0],
        [1.5, 0.0],
        [2.0, 0.0],
    ])
    in_target = jnp.array([True])
    neighbor_indices = jnp.zeros((1, 1), dtype=jnp.int32)
    
    reward = compute_exploration_reward(
        positions, grid_centers, in_target, neighbor_indices,
        collision_threshold=0.15, exploration_threshold=0.05,
        d_sen=3.0, cosine_decay_delta=0.0  # Matches C++ MARL
    )
    
    # Agent is not centered (grids all to the right)
    # Weighted centroid will point right with large norm
    # Therefore reward should be 0.0
    assert reward[0] == 0.0, f"Expected reward 0.0 for off-center agent, got {reward[0]}"
    
    print("  ✓ Exploration reward asymmetric grids test passed")


def test_exploration_reward_not_in_target():
    """Test that exploration reward is 0 when agent is not in target."""
    print("Testing exploration reward when not in target...")
    
    # Agent at origin with symmetric grids, but NOT in target
    positions = jnp.array([[0.0, 0.0]])
    grid_centers = jnp.array([
        [0.5, 0.0],
        [-0.5, 0.0],
        [0.0, 0.5],
        [0.0, -0.5],
    ])
    in_target = jnp.array([False])  # NOT in target
    neighbor_indices = jnp.zeros((1, 1), dtype=jnp.int32)
    
    reward = compute_exploration_reward(
        positions, grid_centers, in_target, neighbor_indices,
        collision_threshold=0.15, exploration_threshold=0.05,
        d_sen=3.0, cosine_decay_delta=0.0  # Matches C++ MARL
    )
    
    # Even though centered, not in target -> no reward
    assert reward[0] == 0.0, f"Expected reward 0.0 when not in target, got {reward[0]}"
    
    print("  ✓ Exploration reward not in target test passed")


def test_exploration_reward_multiple_agents():
    """Test exploration reward with multiple agents."""
    print("Testing exploration reward with multiple agents...")
    
    # 3 agents with different configurations
    positions = jnp.array([
        [0.0, 0.0],   # Agent 0: centered among grids
        [2.0, 0.0],   # Agent 1: off to the side (grids on left)
        [0.0, 2.0],   # Agent 2: far from grids
    ])
    grid_centers = jnp.array([
        [0.5, 0.0],
        [-0.5, 0.0],
        [0.0, 0.5],
        [0.0, -0.5],
    ])
    in_target = jnp.array([True, True, True])
    neighbor_indices = jnp.zeros((3, 1), dtype=jnp.int32)
    
    rewards = compute_exploration_reward(
        positions, grid_centers, in_target, neighbor_indices,
        collision_threshold=0.15, exploration_threshold=0.05,
        d_sen=3.0, cosine_decay_delta=0.0  # Matches C++ MARL
    )
    
    assert rewards.shape == (3,)
    
    # Agent 0 should get reward (centered)
    assert rewards[0] == 1.0, f"Agent 0 should get reward, got {rewards[0]}"
    
    # Agent 1 should NOT get reward (grids all on left side relative to agent)
    assert rewards[1] == 0.0, f"Agent 1 should not get reward, got {rewards[1]}"
    
    print("  ✓ Exploration reward multiple agents test passed")


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
    
    # New tests for C++ MARL matching
    print("\n" + "-"*60)
    print("Testing C++ MARL matching (new)")
    print("-"*60 + "\n")
    
    test_rho_cos_dec_basic()
    test_rho_cos_dec_different_params()
    test_rho_cos_dec_vectorized()
    test_exploration_reward_symmetric_grids()
    test_exploration_reward_asymmetric_grids()
    test_exploration_reward_not_in_target()
    test_exploration_reward_multiple_agents()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")
