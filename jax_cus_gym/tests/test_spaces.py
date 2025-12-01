"""Tests for the spaces module.

Run with: python -m pytest tests/test_spaces.py -v
Or directly: python tests/test_spaces.py
"""

import jax
import jax.numpy as jnp
from jax import random

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spaces import Discrete, Box, MultiAgentActionSpace, MultiAgentObservationSpace


def test_discrete_space():
    """Test Discrete space sampling and containment."""
    print("Testing Discrete space...")
    
    space = Discrete(n=5)
    key = random.PRNGKey(0)
    
    # Test sampling
    sample = space.sample(key)
    assert sample.shape == (), f"Expected shape (), got {sample.shape}"
    assert 0 <= sample < 5, f"Sample {sample} out of range [0, 5)"
    
    # Test containment
    assert space.contains(jnp.array(0)), "0 should be in space"
    assert space.contains(jnp.array(4)), "4 should be in space"
    assert not space.contains(jnp.array(5)), "5 should not be in space"
    assert not space.contains(jnp.array(-1)), "-1 should not be in space"
    
    # Test JIT compilation
    sample_jit = jax.jit(space.sample)
    sample_result = sample_jit(key)
    assert 0 <= sample_result < 5
    
    print("  ✓ Discrete space tests passed")


def test_box_space():
    """Test Box space sampling and containment."""
    print("Testing Box space...")
    
    space = Box(low=-1.0, high=1.0, shape=(3,))
    key = random.PRNGKey(42)
    
    # Test sampling
    sample = space.sample(key)
    assert sample.shape == (3,), f"Expected shape (3,), got {sample.shape}"
    assert jnp.all(sample >= -1.0) and jnp.all(sample <= 1.0)
    
    # Test containment
    assert space.contains(jnp.array([0.0, 0.0, 0.0]))
    assert space.contains(jnp.array([-1.0, 1.0, 0.5]))
    assert not space.contains(jnp.array([2.0, 0.0, 0.0]))
    
    # Test JIT compilation
    sample_jit = jax.jit(space.sample)
    sample_result = sample_jit(key)
    assert sample_result.shape == (3,)
    
    print("  ✓ Box space tests passed")


def test_multi_agent_action_space():
    """Test MultiAgentActionSpace for MADDPG compatibility."""
    print("Testing MultiAgentActionSpace...")
    
    n_agents = 10
    action_dim = 2  # 2D velocity control
    space = MultiAgentActionSpace(
        n_agents=n_agents,
        action_dim=action_dim,
        low=-1.0,
        high=1.0
    )
    
    key = random.PRNGKey(123)
    
    # Test full sampling (all agents)
    actions = space.sample(key)
    assert actions.shape == (n_agents, action_dim), f"Expected {(n_agents, action_dim)}, got {actions.shape}"
    assert jnp.all(actions >= -1.0) and jnp.all(actions <= 1.0)
    
    # Test single agent sampling
    single_action = space.sample_single(key)
    assert single_action.shape == (action_dim,)
    
    # Test containment
    valid_actions = jnp.zeros((n_agents, action_dim))
    assert space.contains(valid_actions)
    
    # Test properties
    assert space.shape == (n_agents, action_dim)
    assert space.single_agent_shape == (action_dim,)
    
    # Test JIT compilation
    sample_jit = jax.jit(space.sample)
    actions_jit = sample_jit(key)
    assert actions_jit.shape == (n_agents, action_dim)
    
    # Test vmap over agents (important for MADDPG)
    keys = random.split(key, n_agents)
    sample_single_vmap = jax.vmap(space.sample_single)
    actions_vmap = sample_single_vmap(keys)
    assert actions_vmap.shape == (n_agents, action_dim)
    
    print("  ✓ MultiAgentActionSpace tests passed")


def test_multi_agent_observation_space():
    """Test MultiAgentObservationSpace."""
    print("Testing MultiAgentObservationSpace...")
    
    n_agents = 10
    obs_dim = 32  # Example: neighbors + self + grid info
    space = MultiAgentObservationSpace(
        n_agents=n_agents,
        obs_dim=obs_dim,
        low=-jnp.inf,
        high=jnp.inf
    )
    
    # Test properties
    assert space.shape == (n_agents, obs_dim)
    assert space.single_agent_shape == (obs_dim,)
    
    # Test containment with valid observation
    valid_obs = jnp.zeros((n_agents, obs_dim))
    assert space.contains(valid_obs)
    
    print("  ✓ MultiAgentObservationSpace tests passed")


def test_jit_compatibility():
    """Test that all spaces work correctly under JIT compilation."""
    print("Testing JIT compatibility...")
    
    @jax.jit
    def sample_all_spaces(key):
        keys = random.split(key, 4)
        
        discrete = Discrete(n=5)
        box = Box(low=-1.0, high=1.0, shape=(4,))
        ma_action = MultiAgentActionSpace(n_agents=3, action_dim=2)
        ma_obs = MultiAgentObservationSpace(n_agents=3, obs_dim=10)
        
        return {
            'discrete': discrete.sample(keys[0]),
            'box': box.sample(keys[1]),
            'ma_action': ma_action.sample(keys[2]),
        }
    
    key = random.PRNGKey(0)
    results = sample_all_spaces(key)
    
    assert results['discrete'].shape == ()
    assert results['box'].shape == (4,)
    assert results['ma_action'].shape == (3, 2)
    
    print("  ✓ JIT compatibility tests passed")


def test_vmap_compatibility():
    """Test that spaces work with vmap for batched environments."""
    print("Testing vmap compatibility...")
    
    n_envs = 8
    n_agents = 5
    action_dim = 2
    
    space = MultiAgentActionSpace(n_agents=n_agents, action_dim=action_dim)
    
    # Batch sample over multiple environments
    @jax.jit
    def batch_sample(keys):
        return jax.vmap(space.sample)(keys)
    
    key = random.PRNGKey(0)
    keys = random.split(key, n_envs)
    
    batch_actions = batch_sample(keys)
    assert batch_actions.shape == (n_envs, n_agents, action_dim)
    
    print("  ✓ vmap compatibility tests passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running spaces module tests")
    print("="*60 + "\n")
    
    test_discrete_space()
    test_box_space()
    test_multi_agent_action_space()
    test_multi_agent_observation_space()
    test_jit_compatibility()
    test_vmap_compatibility()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")
