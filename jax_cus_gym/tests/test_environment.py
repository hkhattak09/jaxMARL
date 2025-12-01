"""Tests for the environment module.

Run with: python tests/test_environment.py
"""

import jax
import jax.numpy as jnp
from jax import random

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import (
    EnvState, 
    EnvParams, 
    MultiAgentEnv,
    SimpleSwarmEnv,
    SimpleSwarmState,
    SimpleSwarmParams,
)


def test_simple_swarm_creation():
    """Test that SimpleSwarmEnv can be created."""
    print("Testing SimpleSwarmEnv creation...")
    
    env = SimpleSwarmEnv(n_agents=4)
    params = env.default_params
    
    assert env.n_agents == 4
    assert params.max_steps_in_episode == 100
    
    print("  ✓ SimpleSwarmEnv creation tests passed")


def test_reset():
    """Test environment reset."""
    print("Testing reset...")
    
    env = SimpleSwarmEnv(n_agents=5)
    params = env.default_params
    key = random.PRNGKey(0)
    
    obs, state = env.reset(key, params)
    
    # Check observation shape
    assert obs.shape == (5, 4), f"Expected obs shape (5, 4), got {obs.shape}"
    
    # Check state
    assert state.positions.shape == (5, 2)
    assert state.velocities.shape == (5, 2)
    assert state.time == 0
    
    # Check positions are within bounds
    assert jnp.all(state.positions >= -params.arena_size)
    assert jnp.all(state.positions <= params.arena_size)
    
    # Check velocities are zero initially
    assert jnp.allclose(state.velocities, 0.0)
    
    print("  ✓ Reset tests passed")


def test_step():
    """Test environment step."""
    print("Testing step...")
    
    env = SimpleSwarmEnv(n_agents=3)
    params = env.default_params
    key = random.PRNGKey(42)
    
    # Reset
    key, reset_key = random.split(key)
    obs, state = env.reset(reset_key, params)
    
    # Take a step with random actions
    key, step_key, action_key = random.split(key, 3)
    actions = random.uniform(action_key, shape=(3, 2), minval=-1.0, maxval=1.0)
    
    obs_new, state_new, rewards, done, info = env.step(step_key, state, actions, params)
    
    # Check shapes
    assert obs_new.shape == (3, 4), f"Expected obs shape (3, 4), got {obs_new.shape}"
    assert rewards.shape == (3,), f"Expected rewards shape (3,), got {rewards.shape}"
    assert done.shape == (), f"Expected done to be scalar, got shape {done.shape}"
    
    # Check time incremented
    assert state_new.time == 1
    
    # Check positions changed (since we had non-zero actions)
    assert not jnp.allclose(state_new.positions, state.positions)
    
    print("  ✓ Step tests passed")


def test_episode_termination():
    """Test that episode terminates after max steps."""
    print("Testing episode termination...")
    
    n_agents = 2
    max_steps = 10
    
    env = SimpleSwarmEnv(n_agents=n_agents)
    params = SimpleSwarmParams(
        max_steps_in_episode=max_steps,
    )
    
    key = random.PRNGKey(123)
    
    # Reset
    key, reset_key = random.split(key)
    obs, state = env.reset(reset_key, params)
    
    # Run for max_steps
    for i in range(max_steps + 5):  # Go beyond max_steps to test auto-reset
        key, step_key, action_key = random.split(key, 3)
        actions = random.uniform(action_key, shape=(n_agents, 2), minval=-1.0, maxval=1.0)
        
        obs, state, rewards, done, info = env.step(step_key, state, actions, params)
        
        if i < max_steps - 1:
            assert not done, f"Episode should not be done at step {i}"
        elif i == max_steps - 1:
            assert done, f"Episode should be done at step {i}"
            # After done, state should auto-reset (time should be 0 or 1 on next step)
    
    print("  ✓ Episode termination tests passed")


def test_auto_reset():
    """Test that environment auto-resets when done."""
    print("Testing auto-reset...")
    
    n_agents = 2
    max_steps = 5
    
    env = SimpleSwarmEnv(n_agents=n_agents)
    params = SimpleSwarmParams(
        max_steps_in_episode=max_steps,
    )
    
    key = random.PRNGKey(456)
    
    # Reset
    key, reset_key = random.split(key)
    obs, state = env.reset(reset_key, params)
    
    # Run until done
    for i in range(max_steps):
        key, step_key, action_key = random.split(key, 3)
        actions = jnp.zeros((n_agents, 2))  # Zero actions
        obs, state, rewards, done, info = env.step(step_key, state, actions, params)
    
    # At this point, done should be True and state should be reset
    assert done, "Episode should be done"
    
    # The state should have been auto-reset (time should be 0 after auto-reset)
    # Note: Due to auto-reset, the returned state is the NEW reset state
    assert state.time == 0, f"State should be reset, but time is {state.time}"
    
    print("  ✓ Auto-reset tests passed")


def test_jit_compatibility():
    """Test that environment works with JIT compilation."""
    print("Testing JIT compatibility...")
    
    env = SimpleSwarmEnv(n_agents=4)
    params = env.default_params
    
    # JIT compile reset and step
    reset_jit = jax.jit(lambda key: env.reset(key, params))
    step_jit = jax.jit(lambda key, state, action: env.step(key, state, action, params))
    
    key = random.PRNGKey(0)
    
    # Test JIT reset
    key, reset_key = random.split(key)
    obs, state = reset_jit(reset_key)
    assert obs.shape == (4, 4)
    
    # Test JIT step
    key, step_key, action_key = random.split(key, 3)
    actions = random.uniform(action_key, shape=(4, 2), minval=-1.0, maxval=1.0)
    obs, state, rewards, done, info = step_jit(step_key, state, actions)
    assert obs.shape == (4, 4)
    assert rewards.shape == (4,)
    
    print("  ✓ JIT compatibility tests passed")


def test_vmap_over_envs():
    """Test that we can vmap over multiple environment instances (batched envs)."""
    print("Testing vmap over environments...")
    
    n_envs = 8
    n_agents = 3
    
    env = SimpleSwarmEnv(n_agents=n_agents)
    params = env.default_params
    
    # Vectorized reset
    def batch_reset(keys):
        return jax.vmap(lambda k: env.reset(k, params))(keys)
    
    batch_reset_jit = jax.jit(batch_reset)
    
    key = random.PRNGKey(0)
    keys = random.split(key, n_envs)
    
    obs_batch, state_batch = batch_reset_jit(keys)
    
    # Check shapes
    assert obs_batch.shape == (n_envs, n_agents, 4), f"Expected {(n_envs, n_agents, 4)}, got {obs_batch.shape}"
    assert state_batch.positions.shape == (n_envs, n_agents, 2)
    
    # Vectorized step
    def batch_step(keys, states, actions):
        return jax.vmap(lambda k, s, a: env.step(k, s, a, params))(keys, states, actions)
    
    batch_step_jit = jax.jit(batch_step)
    
    # Random actions for all envs
    key, action_key = random.split(key)
    actions_batch = random.uniform(action_key, shape=(n_envs, n_agents, 2), minval=-1.0, maxval=1.0)
    
    step_keys = random.split(key, n_envs)
    obs_batch, state_batch, rewards_batch, done_batch, info_batch = batch_step_jit(
        step_keys, state_batch, actions_batch
    )
    
    # Check shapes
    assert obs_batch.shape == (n_envs, n_agents, 4)
    assert rewards_batch.shape == (n_envs, n_agents)
    assert done_batch.shape == (n_envs,)
    
    print("  ✓ vmap over environments tests passed")


def test_spaces():
    """Test action and observation spaces."""
    print("Testing spaces...")
    
    env = SimpleSwarmEnv(n_agents=5)
    params = env.default_params
    
    action_space = env.action_space(params)
    obs_space = env.observation_space(params)
    
    assert action_space.n_agents == 5
    assert action_space.action_dim == 2
    assert action_space.shape == (5, 2)
    
    assert obs_space.n_agents == 5
    assert obs_space.obs_dim == 4
    assert obs_space.shape == (5, 4)
    
    # Test sampling from action space
    key = random.PRNGKey(0)
    actions = action_space.sample(key)
    assert actions.shape == (5, 2)
    assert action_space.contains(actions)
    
    print("  ✓ Spaces tests passed")


def test_rewards_are_per_agent():
    """Test that each agent receives its own reward."""
    print("Testing per-agent rewards...")
    
    env = SimpleSwarmEnv(n_agents=4)
    params = env.default_params
    key = random.PRNGKey(789)
    
    # Reset
    key, reset_key = random.split(key)
    obs, state = env.reset(reset_key, params)
    
    # Take a step
    key, step_key = random.split(key)
    actions = jnp.zeros((4, 2))  # Zero actions
    obs, state, rewards, done, info = env.step(step_key, state, actions, params)
    
    # Rewards should be -distance for each agent
    expected_rewards = -jnp.linalg.norm(state.positions, axis=1)
    
    # They might not be exactly equal due to numerical precision
    assert jnp.allclose(rewards, expected_rewards, atol=1e-5), \
        f"Rewards {rewards} don't match expected {expected_rewards}"
    
    # Each agent at different position should have different reward
    # (unless they happen to be at same distance from origin)
    
    print("  ✓ Per-agent rewards tests passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running environment module tests")
    print("="*60 + "\n")
    
    test_simple_swarm_creation()
    test_reset()
    test_step()
    test_episode_termination()
    test_auto_reset()
    test_jit_compatibility()
    test_vmap_over_envs()
    test_spaces()
    test_rewards_are_per_agent()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")
