"""Tests for the observations module.

Run with: python tests/test_observations.py
"""

import jax
import jax.numpy as jnp
from jax import random

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from observations import (
    ObservationParams,
    compute_observation_dim,
    get_k_nearest_neighbors,
    get_k_nearest_neighbors_all_agents,
    compute_grid_observations,
    compute_target_state,
    compute_observations,
    get_neighbor_indices,
)


def test_observation_dim():
    """Test observation dimension calculation."""
    print("Testing observation dimension calculation...")
    
    params = ObservationParams(
        topo_nei_max=6,
        num_obs_grid_max=80,
        include_self_state=True,
    )
    
    obs_dim = compute_observation_dim(params)
    
    # Calculate expected:
    # Self: 4 (pos + vel)
    # Neighbors: 4 * 6 = 24 (rel_pos + rel_vel for each)
    # Target: 4 (rel_pos + rel_vel)
    # Grid: 2 * 80 = 160
    # Total: 4 + 24 + 4 + 160 = 192
    expected = 4 + 24 + 4 + 160
    
    assert obs_dim == expected, f"Expected {expected}, got {obs_dim}"
    
    # Test without self state
    params_no_self = ObservationParams(
        topo_nei_max=6,
        num_obs_grid_max=80,
        include_self_state=False,
    )
    obs_dim_no_self = compute_observation_dim(params_no_self)
    assert obs_dim_no_self == expected - 4
    
    print("  ✓ Observation dimension calculation tests passed")


def test_k_nearest_neighbors_single():
    """Test k-nearest neighbors for a single agent."""
    print("Testing k-nearest neighbors (single agent)...")
    
    # Setup: 5 agents in a line
    positions = jnp.array([
        [0.0, 0.0],   # Agent 0 (target)
        [0.5, 0.0],   # Agent 1 - closest
        [1.0, 0.0],   # Agent 2 - second closest
        [2.0, 0.0],   # Agent 3 - third
        [5.0, 0.0],   # Agent 4 - far away
    ])
    velocities = jnp.zeros((5, 2))
    
    k = 3
    d_sen = 3.0
    
    rel_pos, rel_vel, mask, indices = get_k_nearest_neighbors(
        agent_idx=0,
        positions=positions,
        velocities=velocities,
        k=k,
        d_sen=d_sen,
    )
    
    # Check shapes
    assert rel_pos.shape == (k, 2)
    assert rel_vel.shape == (k, 2)
    assert mask.shape == (k,)
    assert indices.shape == (k,)
    
    # Should find agents 1, 2, 3 (within sensing range)
    # Agent 4 is at distance 5.0 > d_sen=3.0
    assert mask[0] and mask[1] and mask[2], "Should find 3 neighbors"
    
    # First neighbor should be agent 1 (closest)
    assert indices[0] == 1, f"First neighbor should be agent 1, got {indices[0]}"
    
    # Relative position to agent 1 should be [0.5, 0]
    assert jnp.allclose(rel_pos[0], jnp.array([0.5, 0.0]))
    
    print("  ✓ k-nearest neighbors (single agent) tests passed")


def test_k_nearest_neighbors_all():
    """Test k-nearest neighbors for all agents."""
    print("Testing k-nearest neighbors (all agents)...")
    
    n_agents = 4
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    velocities = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ])
    
    k = 2
    d_sen = 2.0
    
    rel_pos, rel_vel, masks, indices = get_k_nearest_neighbors_all_agents(
        positions, velocities, k, d_sen
    )
    
    # Check shapes
    assert rel_pos.shape == (n_agents, k, 2)
    assert rel_vel.shape == (n_agents, k, 2)
    assert masks.shape == (n_agents, k)
    assert indices.shape == (n_agents, k)
    
    # All agents should have 2 neighbors (square arrangement, all within range)
    assert jnp.all(masks), "All agents should have 2 neighbors"
    
    # Agent 0's closest neighbors should be 1 and 2 (both at distance 1.0)
    assert jnp.isin(indices[0], jnp.array([1, 2])).all()
    
    print("  ✓ k-nearest neighbors (all agents) tests passed")


def test_k_nearest_neighbors_limited():
    """Test that sensing range limits neighbors."""
    print("Testing k-nearest neighbors (limited range)...")
    
    positions = jnp.array([
        [0.0, 0.0],
        [0.5, 0.0],  # Close
        [5.0, 0.0],  # Far
        [6.0, 0.0],  # Very far
    ])
    velocities = jnp.zeros((4, 2))
    
    k = 3
    d_sen = 1.0  # Only agent 1 is within range
    
    rel_pos, rel_vel, masks, indices = get_k_nearest_neighbors_all_agents(
        positions, velocities, k, d_sen
    )
    
    # Agent 0 should only see agent 1
    assert masks[0, 0] == True, "Agent 0 should see agent 1"
    assert masks[0, 1] == False, "Agent 0 should not see agent 2"
    assert masks[0, 2] == False, "Agent 0 should not see agent 3"
    
    print("  ✓ k-nearest neighbors (limited range) tests passed")


def test_grid_observations():
    """Test grid cell observations."""
    print("Testing grid observations...")
    
    # 2 agents
    positions = jnp.array([
        [0.0, 0.0],
        [2.0, 0.0],
    ])
    
    # 4 grid cells in a square
    grid_centers = jnp.array([
        [0.5, 0.5],
        [-0.5, 0.5],
        [0.5, -0.5],
        [-0.5, -0.5],
    ])
    
    d_sen = 2.0
    num_obs_grid_max = 3
    
    rel_grid_pos, grid_masks, nearest_idx = compute_grid_observations(
        positions, grid_centers, d_sen, num_obs_grid_max
    )
    
    # Check shapes
    assert rel_grid_pos.shape == (2, num_obs_grid_max, 2)
    assert grid_masks.shape == (2, num_obs_grid_max)
    assert nearest_idx.shape == (2,)
    
    # Agent 0 is at origin, all grid cells within range
    # Nearest should be one of the cells at distance sqrt(0.5)
    agent0_dist_to_nearest = jnp.linalg.norm(grid_centers[nearest_idx[0]])
    assert jnp.isclose(agent0_dist_to_nearest, jnp.sqrt(0.5), atol=1e-5)
    
    print("  ✓ Grid observations tests passed")


def test_target_state():
    """Test target state computation."""
    print("Testing target state...")
    
    positions = jnp.array([
        [0.0, 0.0],    # At a grid cell center
        [0.5, 0.0],    # Between grid cells
    ])
    velocities = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    
    grid_centers = jnp.array([
        [0.0, 0.0],    # Grid cell at origin
        [1.0, 0.0],    # Grid cell at (1, 0)
    ])
    
    l_cell = 0.5  # Cell size
    
    in_target, target_rel_pos, target_rel_vel = compute_target_state(
        positions, velocities, grid_centers, l_cell
    )
    
    # Check shapes
    assert in_target.shape == (2,)
    assert target_rel_pos.shape == (2, 2)
    assert target_rel_vel.shape == (2, 2)
    
    # Agent 0 is at grid cell center, should be "in target"
    assert in_target[0], "Agent 0 should be in target"
    
    # Agent 1 is between cells, may or may not be "in target" depending on threshold
    # The threshold is sqrt(2) * l_cell / 2 = sqrt(2) * 0.25 ≈ 0.354
    # Agent 1 is at distance 0.5 from grid center (0,0), which is > 0.354
    # But distance to grid (1,0) is also 0.5
    # So agent 1 should NOT be in target
    assert not in_target[1], "Agent 1 should not be in target"
    
    print("  ✓ Target state tests passed")


def test_compute_observations():
    """Test full observation computation."""
    print("Testing full observation computation...")
    
    n_agents = 4
    positions = jnp.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
    ])
    velocities = jnp.zeros((n_agents, 2))
    
    # Simple grid
    grid_centers = jnp.array([
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.75, 0.75],
    ])
    l_cell = 0.5
    
    obs_params = ObservationParams(
        topo_nei_max=2,
        num_obs_grid_max=4,
        d_sen=1.0,
        include_self_state=True,
        normalize_obs=False,
    )
    
    observations = compute_observations(
        positions, velocities, grid_centers, l_cell, obs_params
    )
    
    # Check shape
    expected_dim = compute_observation_dim(obs_params)
    assert observations.shape == (n_agents, expected_dim), \
        f"Expected shape ({n_agents}, {expected_dim}), got {observations.shape}"
    
    # Observations should not contain NaN
    assert not jnp.any(jnp.isnan(observations)), "Observations contain NaN"
    
    print("  ✓ Full observation computation tests passed")


def test_observations_jit():
    """Test that observations work with JIT."""
    print("Testing observations JIT compatibility...")
    
    obs_params = ObservationParams(
        topo_nei_max=4,
        num_obs_grid_max=10,
        d_sen=2.0,
        include_self_state=True,
        normalize_obs=True,
    )
    
    @jax.jit
    def compute_obs_jit(positions, velocities, grid_centers, l_cell):
        return compute_observations(
            positions, velocities, grid_centers, l_cell, obs_params
        )
    
    n_agents = 8
    n_grid = 20
    
    key = random.PRNGKey(0)
    positions = random.uniform(key, (n_agents, 2), minval=-2.0, maxval=2.0)
    velocities = random.uniform(key, (n_agents, 2), minval=-1.0, maxval=1.0)
    grid_centers = random.uniform(key, (n_grid, 2), minval=-1.0, maxval=1.0)
    l_cell = 0.2
    
    observations = compute_obs_jit(positions, velocities, grid_centers, l_cell)
    
    expected_dim = compute_observation_dim(obs_params)
    assert observations.shape == (n_agents, expected_dim)
    
    print("  ✓ Observations JIT compatibility tests passed")


def test_observations_vmap():
    """Test that observations work with vmap (batched environments)."""
    print("Testing observations vmap compatibility...")
    
    n_envs = 4
    n_agents = 5
    n_grid = 15
    
    obs_params = ObservationParams(
        topo_nei_max=3,
        num_obs_grid_max=8,
        d_sen=2.0,
    )
    
    @jax.jit
    def batch_compute_obs(positions, velocities, grid_centers, l_cell):
        return jax.vmap(
            lambda p, v, g: compute_observations(p, v, g, l_cell, obs_params)
        )(positions, velocities, grid_centers)
    
    key = random.PRNGKey(42)
    positions = random.uniform(key, (n_envs, n_agents, 2), minval=-2.0, maxval=2.0)
    velocities = random.uniform(key, (n_envs, n_agents, 2), minval=-1.0, maxval=1.0)
    grid_centers = random.uniform(key, (n_envs, n_grid, 2), minval=-1.0, maxval=1.0)
    l_cell = 0.3
    
    observations = batch_compute_obs(positions, velocities, grid_centers, l_cell)
    
    expected_dim = compute_observation_dim(obs_params)
    assert observations.shape == (n_envs, n_agents, expected_dim)
    
    print("  ✓ Observations vmap compatibility tests passed")


def test_neighbor_indices():
    """Test neighbor indices utility function."""
    print("Testing neighbor indices utility...")
    
    positions = jnp.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [5.0, 0.0],  # Far away
    ])
    
    k = 2
    d_sen = 2.0
    
    neighbor_idx = get_neighbor_indices(positions, k, d_sen)
    
    assert neighbor_idx.shape == (4, k)
    
    # Agent 0 should see agents 1 and 2
    assert 1 in neighbor_idx[0]
    assert 2 in neighbor_idx[0]
    
    # Agent 3 (far away) should see no one (all -1)
    assert jnp.all(neighbor_idx[3] == -1)
    
    print("  ✓ Neighbor indices utility tests passed")


def test_normalization():
    """Test that normalization produces bounded values."""
    print("Testing observation normalization...")
    
    obs_params = ObservationParams(
        topo_nei_max=3,
        num_obs_grid_max=5,
        d_sen=2.0,
        include_self_state=True,
        normalize_obs=True,
        l_max=2.0,
        vel_max=1.0,
    )
    
    # Agents at boundary
    positions = jnp.array([
        [2.0, 2.0],
        [-2.0, -2.0],
    ])
    velocities = jnp.array([
        [1.0, 1.0],
        [-1.0, -1.0],
    ])
    
    grid_centers = jnp.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    l_cell = 0.5
    
    observations = compute_observations(
        positions, velocities, grid_centers, l_cell, obs_params
    )
    
    # Self-state part should be normalized to [-1, 1]
    # positions / l_max = [1, 1] and [-1, -1]
    # velocities / vel_max = [1, 1] and [-1, -1]
    assert jnp.all(observations[:, :2] >= -1.1) and jnp.all(observations[:, :2] <= 1.1), \
        "Position normalization failed"
    
    print("  ✓ Observation normalization tests passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running observations module tests")
    print("="*60 + "\n")
    
    test_observation_dim()
    test_k_nearest_neighbors_single()
    test_k_nearest_neighbors_all()
    test_k_nearest_neighbors_limited()
    test_grid_observations()
    test_target_state()
    test_compute_observations()
    test_observations_jit()
    test_observations_vmap()
    test_neighbor_indices()
    test_normalization()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")
