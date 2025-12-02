"""Test observation and reward normalization improvements.

This test verifies that:
1. Observations are properly normalized to approximately [-1, 1]
2. Reward clipping is working correctly
3. Layer norm is enabled in networks

Run: python -m pytest jax_cus_gym/tests/test_normalization.py -v
Or:  python jax_cus_gym/tests/test_normalization.py
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "jax_marl"))

import jax
import jax.numpy as jnp
from jax import random
import numpy as np


def test_observation_normalization():
    """Test that observations are properly normalized."""
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    from observations import ObservationParams
    
    print("\n" + "="*60)
    print("Testing Observation Normalization")
    print("="*60)
    
    # Check that l_max is correctly set
    obs_params = ObservationParams()
    print(f"\nObservationParams defaults:")
    print(f"  l_max: {obs_params.l_max} (should be 2.5)")
    print(f"  vel_max: {obs_params.vel_max}")
    print(f"  normalize_obs: {obs_params.normalize_obs}")
    
    assert obs_params.l_max == 2.5, f"l_max should be 2.5, got {obs_params.l_max}"
    assert obs_params.normalize_obs == True, "normalize_obs should be True by default"
    
    # Create environment and run a few steps
    env = AssemblySwarmEnv(n_agents=10)
    params = AssemblyParams(
        arena_size=5.0,  # This means positions in [-2.5, 2.5]
        max_velocity=0.8,
    )
    
    key = random.PRNGKey(42)
    
    # Collect observations from multiple episodes
    all_obs = []
    for ep in range(5):
        key, reset_key = random.split(key)
        obs, state = env.reset(reset_key, params)
        all_obs.append(obs)
        
        # Run a few steps
        for step in range(20):
            key, action_key, step_key = random.split(key, 3)
            actions = random.uniform(action_key, (10, 2), minval=-1, maxval=1)
            obs, state, rewards, dones, info = env.step(step_key, state, actions, params)
            all_obs.append(obs)
    
    # Stack all observations
    all_obs = jnp.stack(all_obs)  # (n_samples, n_agents, obs_dim)
    all_obs_flat = all_obs.reshape(-1, all_obs.shape[-1])  # (n_samples * n_agents, obs_dim)
    
    # Compute statistics
    obs_min = jnp.min(all_obs_flat, axis=0)
    obs_max = jnp.max(all_obs_flat, axis=0)
    obs_mean = jnp.mean(all_obs_flat, axis=0)
    obs_std = jnp.std(all_obs_flat, axis=0)
    
    print(f"\nObservation statistics (across {all_obs_flat.shape[0]} samples):")
    print(f"  Shape: {all_obs_flat.shape}")
    print(f"  Global min: {float(jnp.min(obs_min)):.3f}")
    print(f"  Global max: {float(jnp.max(obs_max)):.3f}")
    print(f"  Mean range: [{float(jnp.min(obs_mean)):.3f}, {float(jnp.max(obs_mean)):.3f}]")
    print(f"  Std range: [{float(jnp.min(obs_std)):.3f}, {float(jnp.max(obs_std)):.3f}]")
    
    # Check that most values are in reasonable range
    # With normalization, we expect roughly [-2, 2] for most values
    in_range = jnp.abs(all_obs_flat) < 3.0  # Allow some margin
    in_range_pct = 100 * jnp.mean(in_range)
    print(f"  Values in [-3, 3]: {float(in_range_pct):.1f}%")
    
    assert in_range_pct > 95, f"Only {in_range_pct:.1f}% of observations in [-3, 3]"
    
    print("\n✓ Observation normalization test PASSED")
    return True


def test_reward_clipping():
    """Test that reward clipping is working."""
    from rewards import RewardParams, compute_rewards
    from observations import get_k_nearest_neighbors_all_agents
    
    print("\n" + "="*60)
    print("Testing Reward Clipping")
    print("="*60)
    
    # Check that clipping params exist
    reward_params = RewardParams()
    print(f"\nRewardParams defaults:")
    print(f"  reward_clip_min: {reward_params.reward_clip_min}")
    print(f"  reward_clip_max: {reward_params.reward_clip_max}")
    
    assert hasattr(reward_params, 'reward_clip_min'), "reward_clip_min not found"
    assert hasattr(reward_params, 'reward_clip_max'), "reward_clip_max not found"
    assert reward_params.reward_clip_min == -10.0, f"Expected -10.0, got {reward_params.reward_clip_min}"
    assert reward_params.reward_clip_max == 10.0, f"Expected 10.0, got {reward_params.reward_clip_max}"
    
    # Test that clipping actually works by creating extreme reward scenario
    n_agents = 10
    key = random.PRNGKey(42)
    
    # Create positions that might generate various rewards
    positions = random.uniform(key, (n_agents, 2), minval=-2, maxval=2)
    velocities = jnp.zeros((n_agents, 2))
    
    # Create a simple grid
    grid_centers = jnp.array([
        [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
        [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
        [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0],
    ])
    l_cell = 0.5
    
    # Get neighbor indices
    _, _, _, neighbor_indices = get_k_nearest_neighbors_all_agents(
        positions, velocities, k=6, d_sen=3.0,
        is_periodic=False, boundary_width=2.5, boundary_height=2.5
    )
    
    # Compute rewards
    rewards, info = compute_rewards(
        positions, velocities, grid_centers, l_cell, neighbor_indices,
        reward_params,
        is_periodic=False, boundary_width=2.5, boundary_height=2.5, d_sen=3.0
    )
    
    print(f"\nReward statistics:")
    print(f"  Min: {float(jnp.min(rewards)):.3f}")
    print(f"  Max: {float(jnp.max(rewards)):.3f}")
    print(f"  Mean: {float(jnp.mean(rewards)):.3f}")
    
    # Verify clipping
    assert jnp.all(rewards >= reward_params.reward_clip_min), "Rewards below clip_min"
    assert jnp.all(rewards <= reward_params.reward_clip_max), "Rewards above clip_max"
    
    # Test with extreme params to verify clipping is applied
    extreme_params = RewardParams(
        reward_entering=100.0,  # Very high reward
        penalty_collision=-100.0,  # Very high penalty
        reward_clip_min=-10.0,
        reward_clip_max=10.0,
    )
    
    rewards_extreme, _ = compute_rewards(
        positions, velocities, grid_centers, l_cell, neighbor_indices,
        extreme_params,
        is_periodic=False, boundary_width=2.5, boundary_height=2.5, d_sen=3.0
    )
    
    print(f"\nWith extreme reward params (±100):")
    print(f"  Min (should be >= -10): {float(jnp.min(rewards_extreme)):.3f}")
    print(f"  Max (should be <= 10): {float(jnp.max(rewards_extreme)):.3f}")
    
    assert jnp.all(rewards_extreme >= -10.0), "Clipping failed for min"
    assert jnp.all(rewards_extreme <= 10.0), "Clipping failed for max"
    
    print("\n✓ Reward clipping test PASSED")
    return True


def test_layer_norm_enabled():
    """Test that layer norm is enabled by default in MADDPG config."""
    print("\n" + "="*60)
    print("Testing Layer Norm Configuration")
    print("="*60)
    
    from algo import MADDPGConfig
    
    config = MADDPGConfig(
        n_agents=3,
        obs_dims=(10, 10, 10),
        action_dims=(2, 2, 2),
    )
    
    print(f"\nMADDPGConfig defaults:")
    print(f"  use_layer_norm: {config.use_layer_norm}")
    
    assert config.use_layer_norm == True, f"use_layer_norm should be True, got {config.use_layer_norm}"
    
    print("\n✓ Layer norm configuration test PASSED")
    return True


def test_full_integration():
    """Test that everything works together in a training-like scenario."""
    print("\n" + "="*60)
    print("Testing Full Integration")
    print("="*60)
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    from algo import MADDPG, MADDPGConfig
    
    # Create environment
    n_agents = 5
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = AssemblyParams(arena_size=5.0, max_steps=50)
    
    # Create MADDPG with layer norm
    key = random.PRNGKey(42)
    key, reset_key = random.split(key)
    obs, state = env.reset(reset_key, params)
    obs_dim = obs.shape[-1]
    
    maddpg_config = MADDPGConfig(
        n_agents=n_agents,
        obs_dims=tuple([obs_dim] * n_agents),
        action_dims=tuple([2] * n_agents),
        hidden_dims=(64, 64),
        buffer_size=1000,
        batch_size=32,
        warmup_steps=50,
        use_layer_norm=True,  # Verify this is used
    )
    
    print(f"\nConfig: use_layer_norm={maddpg_config.use_layer_norm}")
    
    maddpg = MADDPG(maddpg_config)
    key, init_key = random.split(key)
    maddpg_state = maddpg.init(init_key)
    
    # Run a few steps and collect statistics
    obs_list = []
    reward_list = []
    
    for step in range(100):
        key, action_key, step_key = random.split(key, 3)
        
        # Select actions
        obs_batch = obs[None, :, :]  # Add batch dim
        actions_batch, maddpg_state = maddpg.select_actions_batched(
            action_key, maddpg_state, obs_batch, explore=True
        )
        actions = actions_batch[0]  # Remove batch dim
        
        # Step environment
        obs, state, rewards, dones, info = env.step(step_key, state, actions, params)
        
        obs_list.append(obs)
        reward_list.append(rewards)
        
        if state.done:
            key, reset_key = random.split(key)
            obs, state = env.reset(reset_key, params)
    
    # Check observation ranges
    all_obs = jnp.stack(obs_list)
    obs_min = float(jnp.min(all_obs))
    obs_max = float(jnp.max(all_obs))
    
    print(f"\nIntegration test statistics:")
    print(f"  Observation range: [{obs_min:.3f}, {obs_max:.3f}]")
    
    # Check reward ranges
    all_rewards = jnp.stack(reward_list)
    reward_min = float(jnp.min(all_rewards))
    reward_max = float(jnp.max(all_rewards))
    
    print(f"  Reward range: [{reward_min:.3f}, {reward_max:.3f}]")
    
    assert reward_min >= -10.0, f"Rewards below clip: {reward_min}"
    assert reward_max <= 10.0, f"Rewards above clip: {reward_max}"
    
    print("\n✓ Full integration test PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("NORMALIZATION IMPROVEMENT VERIFICATION TESTS")
    print("="*60)
    
    tests = [
        ("Observation Normalization", test_observation_normalization),
        ("Reward Clipping", test_reward_clipping),
        ("Layer Norm Configuration", test_layer_norm_enabled),
        ("Full Integration", test_full_integration),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    for name, passed, error in results:
        status = "✓ PASSED" if passed else f"✗ FAILED: {error}"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All normalization improvements verified!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
