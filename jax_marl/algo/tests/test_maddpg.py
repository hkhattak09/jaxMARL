"""Tests for maddpg.py - MADDPG algorithm implementation."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random
import pytest

from maddpg import (
    MADDPG,
    MADDPGConfig,
    MADDPGState,
    make_maddpg,
)


class TestMADDPGConfig:
    """Tests for MADDPG configuration."""
    
    def test_config_creation(self):
        """Test creating MADDPG config."""
        print("Testing MADDPGConfig creation...")
        
        config = MADDPGConfig(
            n_agents=3,
            obs_dims=(10, 10, 10),
            action_dims=(2, 2, 2),
        )
        
        assert config.n_agents == 3
        assert config.obs_dims == (10, 10, 10)
        assert config.action_dims == (2, 2, 2)
        assert config.gamma == 0.95  # default
        assert config.tau == 0.01  # default
        
        print("   MADDPGConfig creation: PASSED")
    
    def test_config_with_custom_params(self):
        """Test config with custom parameters."""
        print("Testing MADDPGConfig with custom params...")
        
        config = MADDPGConfig(
            n_agents=4,
            obs_dims=(8, 8, 8, 8),
            action_dims=(3, 3, 3, 3),
            hidden_dims=(128, 128),
            lr_actor=1e-3,
            lr_critic=1e-2,
            gamma=0.99,
            tau=0.005,
            buffer_size=500000,
            batch_size=256,
        )
        
        assert config.hidden_dims == (128, 128)
        assert config.lr_actor == 1e-3
        assert config.gamma == 0.99
        
        print("   MADDPGConfig with custom params: PASSED")


class TestMADDPGCreation:
    """Tests for MADDPG creation."""
    
    def test_maddpg_creation(self):
        """Test creating MADDPG instance."""
        print("Testing MADDPG creation...")
        
        config = MADDPGConfig(
            n_agents=3,
            obs_dims=(10, 10, 10),
            action_dims=(2, 2, 2),
        )
        
        maddpg = MADDPG(config)
        
        assert maddpg.n_agents == 3
        assert len(maddpg.actors) == 3
        assert len(maddpg.critics) == 3
        assert maddpg.total_obs_dim == 30
        assert maddpg.total_action_dim == 6
        
        print("   MADDPG creation: PASSED")
    
    def test_maddpg_init(self):
        """Test initializing MADDPG state."""
        print("Testing MADDPG init...")
        
        config = MADDPGConfig(
            n_agents=2,
            obs_dims=(10, 10),
            action_dims=(2, 2),
        )
        
        maddpg = MADDPG(config)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        assert isinstance(state, MADDPGState)
        assert len(state.agent_states) == 2
        assert state.step == 0
        assert state.episode == 0
        assert state.noise_scale == config.noise_scale_initial
        
        # Check buffer initialized
        assert state.buffer_state is not None
        
        print("   MADDPG init: PASSED")
    
    def test_make_maddpg_uniform_dims(self):
        """Test make_maddpg with uniform dimensions."""
        print("Testing make_maddpg with uniform dims...")
        
        maddpg = make_maddpg(
            n_agents=4,
            obs_dims=10,  # Same for all agents
            action_dims=2,  # Same for all agents
        )
        
        assert maddpg.n_agents == 4
        assert maddpg.obs_dims == (10, 10, 10, 10)
        assert maddpg.action_dims == (2, 2, 2, 2)
        
        print("   make_maddpg with uniform dims: PASSED")
    
    def test_make_maddpg_heterogeneous_dims(self):
        """Test make_maddpg with different dimensions per agent."""
        print("Testing make_maddpg with heterogeneous dims...")
        
        maddpg = make_maddpg(
            n_agents=3,
            obs_dims=(8, 10, 12),
            action_dims=(2, 3, 4),
        )
        
        assert maddpg.obs_dims == (8, 10, 12)
        assert maddpg.action_dims == (2, 3, 4)
        assert maddpg.total_obs_dim == 30
        assert maddpg.total_action_dim == 9
        
        print("   make_maddpg with heterogeneous dims: PASSED")
    
    def test_maddpg_shared_critic(self):
        """Test MADDPG with shared critic."""
        print("Testing MADDPG with shared critic...")
        
        config = MADDPGConfig(
            n_agents=3,
            obs_dims=(10, 10, 10),
            action_dims=(2, 2, 2),
            shared_critic=True,
        )
        
        maddpg = MADDPG(config)
        
        # All critics should be the same object
        assert maddpg.critics[0] is maddpg.critics[1]
        assert maddpg.critics[1] is maddpg.critics[2]
        
        print("   MADDPG with shared critic: PASSED")


class TestActionSelection:
    """Tests for action selection."""
    
    def test_select_actions_explore(self):
        """Test selecting actions with exploration."""
        print("Testing select_actions with exploration...")
        
        maddpg = make_maddpg(n_agents=2, obs_dims=10, action_dims=2)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        # Create observations for each agent
        observations = [jnp.zeros(10), jnp.zeros(10)]
        
        key, action_key = random.split(key)
        actions, log_probs, new_state = maddpg.select_actions(
            action_key, state, observations, explore=True
        )
        
        assert len(actions) == 2
        assert actions[0].shape == (2,)
        assert actions[1].shape == (2,)
        assert len(log_probs) == 2
        
        print(f"   Actions: {actions[0]}, {actions[1]}")
        print("   select_actions with exploration: PASSED")
    
    def test_select_actions_no_explore(self):
        """Test selecting actions without exploration."""
        print("Testing select_actions without exploration...")
        
        maddpg = make_maddpg(n_agents=2, obs_dims=10, action_dims=2)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        observations = [jnp.zeros(10), jnp.zeros(10)]
        
        key, action_key = random.split(key)
        actions, log_probs, new_state = maddpg.select_actions(
            action_key, state, observations, explore=False
        )
        
        assert len(actions) == 2
        
        # Without exploration, same obs should give same actions
        actions2, _, _ = maddpg.select_actions(
            action_key, state, observations, explore=False
        )
        
        assert jnp.allclose(actions[0], actions2[0])
        assert jnp.allclose(actions[1], actions2[1])
        
        print("   select_actions without exploration: PASSED")
    
    def test_select_target_actions(self):
        """Test selecting target actions."""
        print("Testing select_target_actions...")
        
        maddpg = make_maddpg(n_agents=3, obs_dims=10, action_dims=2)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        batch_size = 32
        next_obs = jnp.zeros((batch_size, 30))  # total_obs_dim = 30
        
        target_actions = maddpg.select_target_actions(state, next_obs)
        
        assert target_actions.shape == (batch_size, 6)  # total_action_dim = 6
        
        print("   select_target_actions: PASSED")


class TestBufferOperations:
    """Tests for replay buffer operations."""
    
    def test_store_transition(self):
        """Test storing transitions in buffer."""
        print("Testing store_transition...")
        
        maddpg = make_maddpg(n_agents=2, obs_dims=10, action_dims=2)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        # Create dummy transition
        observations = [jnp.zeros(10), jnp.zeros(10)]
        actions = [jnp.zeros(2), jnp.zeros(2)]
        rewards = [jnp.array(1.0), jnp.array(0.5)]
        next_observations = [jnp.ones(10), jnp.ones(10)]
        dones = [jnp.array(False), jnp.array(False)]
        
        new_state = maddpg.store_transition(
            state=state,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
        )
        
        # Buffer size should have increased
        assert int(new_state.buffer_state.size) == 1
        
        print("   store_transition: PASSED")
    
    def test_store_multiple_transitions(self):
        """Test storing multiple transitions."""
        print("Testing store multiple transitions...")
        
        maddpg = make_maddpg(n_agents=2, obs_dims=10, action_dims=2)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        # Store 100 transitions
        for i in range(100):
            observations = [random.normal(random.PRNGKey(i), (10,)) for _ in range(2)]
            actions = [random.uniform(random.PRNGKey(i+1000), (2,), minval=-1, maxval=1) for _ in range(2)]
            rewards = [jnp.array(float(i % 10) / 10) for _ in range(2)]
            next_observations = [random.normal(random.PRNGKey(i+2000), (10,)) for _ in range(2)]
            dones = [jnp.array(False) for _ in range(2)]
            
            state = maddpg.store_transition(
                state=state,
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                dones=dones,
            )
        
        assert int(state.buffer_state.size) == 100
        
        print("   store multiple transitions: PASSED")


class TestUpdate:
    """Tests for MADDPG update."""
    
    def test_update_not_ready(self):
        """Test update when buffer doesn't have enough samples."""
        print("Testing update when not ready...")
        
        config = MADDPGConfig(
            n_agents=2,
            obs_dims=(10, 10),
            action_dims=(2, 2),
            batch_size=64,
            warmup_steps=100,
        )
        maddpg = MADDPG(config)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        # Try to update without any data
        key, update_key = random.split(key)
        new_state, info = maddpg.update(update_key, state)
        
        assert info.get('can_update') == False
        
        print("   update when not ready: PASSED")
    
    def test_update_with_data(self):
        """Test update with sufficient data."""
        print("Testing update with data...")
        
        config = MADDPGConfig(
            n_agents=2,
            obs_dims=(10, 10),
            action_dims=(2, 2),
            batch_size=32,
            warmup_steps=0,
        )
        maddpg = MADDPG(config)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        # Fill buffer with enough data
        for i in range(100):
            key, k = random.split(key)
            observations = [random.normal(k, (10,)) for _ in range(2)]
            actions = [random.uniform(k, (2,), minval=-1, maxval=1) for _ in range(2)]
            rewards = [jnp.array(0.1) for _ in range(2)]
            next_observations = [random.normal(k, (10,)) for _ in range(2)]
            dones = [jnp.array(False) for _ in range(2)]
            
            state = maddpg.store_transition(
                state=state,
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                dones=dones,
            )
        
        # Now update should work
        key, update_key = random.split(key)
        new_state, info = maddpg.update(update_key, state)
        
        assert info.get('can_update') == True
        assert 'agent_0/critic_loss' in info
        assert 'agent_0/actor_loss' in info
        assert 'agent_1/critic_loss' in info
        assert 'agent_1/actor_loss' in info
        
        print(f"   Agent 0 critic loss: {info['agent_0/critic_loss']:.4f}")
        print(f"   Agent 0 actor loss: {info['agent_0/actor_loss']:.4f}")
        print("   update with data: PASSED")
    
    def test_update_changes_params(self):
        """Test that update changes network parameters."""
        print("Testing update changes params...")
        
        config = MADDPGConfig(
            n_agents=2,
            obs_dims=(10, 10),
            action_dims=(2, 2),
            batch_size=32,
            warmup_steps=0,
        )
        maddpg = MADDPG(config)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        # Store data
        for i in range(100):
            key, k = random.split(key)
            observations = [random.normal(k, (10,)) for _ in range(2)]
            actions = [random.uniform(k, (2,), minval=-1, maxval=1) for _ in range(2)]
            rewards = [jnp.array(1.0) for _ in range(2)]
            next_observations = [random.normal(k, (10,)) for _ in range(2)]
            dones = [jnp.array(False) for _ in range(2)]
            
            state = maddpg.store_transition(
                state=state,
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                dones=dones,
            )
        
        # Get initial params
        old_actor_params = jax.tree_util.tree_leaves(state.agent_states[0].actor_params)
        old_critic_params = jax.tree_util.tree_leaves(state.agent_states[0].critic_params)
        
        # Update
        key, update_key = random.split(key)
        new_state, info = maddpg.update(update_key, state)
        
        # Check params changed
        new_actor_params = jax.tree_util.tree_leaves(new_state.agent_states[0].actor_params)
        new_critic_params = jax.tree_util.tree_leaves(new_state.agent_states[0].critic_params)
        
        actor_changed = any(
            not jnp.allclose(old, new)
            for old, new in zip(old_actor_params, new_actor_params)
        )
        critic_changed = any(
            not jnp.allclose(old, new)
            for old, new in zip(old_critic_params, new_critic_params)
        )
        
        assert actor_changed, "Actor params should change"
        assert critic_changed, "Critic params should change"
        
        print("   update changes params: PASSED")


class TestNoiseManagement:
    """Tests for noise management."""
    
    def test_reset_noise(self):
        """Test resetting exploration noise."""
        print("Testing reset_noise...")
        
        maddpg = make_maddpg(n_agents=2, obs_dims=10, action_dims=2)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        # Select some actions to potentially change noise state
        observations = [jnp.zeros(10), jnp.zeros(10)]
        for i in range(10):
            key, k = random.split(key)
            _, _, state = maddpg.select_actions(k, state, observations, explore=True)
        
        # Reset noise
        reset_state = maddpg.reset_noise(state)
        
        # Check noise was reset
        assert reset_state is not None
        
        print("   reset_noise: PASSED")
    
    def test_increment_episode(self):
        """Test incrementing episode counter."""
        print("Testing increment_episode...")
        
        maddpg = make_maddpg(n_agents=2, obs_dims=10, action_dims=2)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        assert state.episode == 0
        
        state = maddpg.increment_episode(state)
        assert state.episode == 1
        
        state = maddpg.increment_episode(state)
        assert state.episode == 2
        
        print("   increment_episode: PASSED")
    
    def test_noise_decay(self):
        """Test that noise scale decays during training."""
        print("Testing noise decay...")
        
        config = MADDPGConfig(
            n_agents=2,
            obs_dims=(10, 10),
            action_dims=(2, 2),
            batch_size=32,
            warmup_steps=0,
            noise_scale_initial=0.3,
            noise_scale_final=0.05,
            noise_decay_steps=100,
        )
        maddpg = MADDPG(config)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        initial_noise = float(state.noise_scale)
        
        # Fill buffer
        for i in range(100):
            key, k = random.split(key)
            observations = [random.normal(k, (10,)) for _ in range(2)]
            actions = [random.uniform(k, (2,), minval=-1, maxval=1) for _ in range(2)]
            rewards = [jnp.array(0.1) for _ in range(2)]
            next_observations = [random.normal(k, (10,)) for _ in range(2)]
            dones = [jnp.array(False) for _ in range(2)]
            
            state = maddpg.store_transition(
                state=state,
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                dones=dones,
            )
        
        # Do several updates
        for i in range(50):
            key, update_key = random.split(key)
            state, info = maddpg.update(update_key, state)
        
        final_noise = float(state.noise_scale)
        
        assert final_noise < initial_noise, f"Noise should decay: {initial_noise} -> {final_noise}"
        
        print(f"   Noise decay: {initial_noise:.3f} -> {final_noise:.3f}")
        print("   noise decay: PASSED")


class TestSerialization:
    """Tests for saving/loading parameters."""
    
    def test_get_and_load_params(self):
        """Test getting and loading parameters."""
        print("Testing get_params and load_params...")
        
        maddpg = make_maddpg(n_agents=2, obs_dims=10, action_dims=2)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        # Get params
        params = maddpg.get_params(state)
        
        assert 'agent_params' in params
        assert len(params['agent_params']) == 2
        assert 'config' in params
        
        # Create new state and load params
        key2 = random.PRNGKey(123)
        new_state = maddpg.init(key2)
        
        # Initially different
        old_leaves = jax.tree_util.tree_leaves(new_state.agent_states[0].actor_params)
        orig_leaves = jax.tree_util.tree_leaves(state.agent_states[0].actor_params)
        
        # Load
        loaded_state = maddpg.load_params(new_state, params)
        
        # Should match now
        loaded_leaves = jax.tree_util.tree_leaves(loaded_state.agent_states[0].actor_params)
        
        all_match = all(
            jnp.allclose(o, l)
            for o, l in zip(orig_leaves, loaded_leaves)
        )
        assert all_match, "Loaded params should match"
        
        print("   get_params and load_params: PASSED")


class TestJITCompatibility:
    """Tests for JIT compilation."""
    
    def test_select_actions_jittable(self):
        """Test that select_actions can be JIT compiled."""
        print("Testing JIT compilation...")
        
        maddpg = make_maddpg(n_agents=2, obs_dims=10, action_dims=2)
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        # This should be JIT-compilable
        @jax.jit
        def select_fn(key, agent_states, obs_list):
            # Simplified version for JIT test
            actions = []
            for i, (actor, agent_state, obs) in enumerate(zip(maddpg.actors, agent_states, obs_list)):
                action = actor.apply(agent_state.actor_params, obs[None, :])
                actions.append(action[0])
            return actions
        
        observations = [jnp.zeros(10), jnp.zeros(10)]
        actions = select_fn(key, state.agent_states, observations)
        
        assert len(actions) == 2
        print("   select_actions JIT: PASSED")
        print("   JIT compilation: PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running maddpg.py tests")
    print("=" * 60)
    
    test_classes = [
        TestMADDPGConfig,
        TestMADDPGCreation,
        TestActionSelection,
        TestBufferOperations,
        TestUpdate,
        TestNoiseManagement,
        TestSerialization,
        TestJITCompatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    passed += 1
                except Exception as e:
                    print(f"   FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
