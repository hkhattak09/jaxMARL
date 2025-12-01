"""Tests for buffers.py module.

Tests cover:
1. ReplayBuffer - Multi-agent replay buffer
2. PerAgentReplayBuffer - Single agent buffer
3. Utility functions for data manipulation
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random

import sys
sys.path.insert(0, '/home/hassan/MARL_jax/jaxMARL/jax_marl/algo')

from buffers import (
    Transition,
    BatchTransition,
    ReplayBuffer,
    ReplayBufferState,
    PerAgentReplayBuffer,
    PerAgentBufferState,
    PerAgentTransition,
    PerAgentBatch,
    create_replay_buffer,
    flatten_transition_for_agent,
    get_all_actions_flat,
    get_all_obs_flat,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def key():
    """Random key for testing."""
    return random.PRNGKey(42)


@pytest.fixture
def buffer_config():
    """Default buffer configuration."""
    return {
        'capacity': 1000,
        'n_agents': 5,
        'obs_dim': 10,
        'action_dim': 2,
        'global_state_dim': 50,
    }


@pytest.fixture
def replay_buffer(buffer_config):
    """Create replay buffer instance."""
    return ReplayBuffer(**buffer_config)


@pytest.fixture
def buffer_state(replay_buffer):
    """Initialize buffer state."""
    return replay_buffer.init()


@pytest.fixture
def sample_transition(buffer_config, key):
    """Create a sample transition."""
    n_agents = buffer_config['n_agents']
    obs_dim = buffer_config['obs_dim']
    action_dim = buffer_config['action_dim']
    global_state_dim = buffer_config['global_state_dim']
    
    keys = random.split(key, 6)
    
    return Transition(
        obs=random.normal(keys[0], (n_agents, obs_dim)),
        actions=random.uniform(keys[1], (n_agents, action_dim), minval=-1, maxval=1),
        rewards=random.normal(keys[2], (n_agents,)),
        next_obs=random.normal(keys[3], (n_agents, obs_dim)),
        dones=jnp.zeros((n_agents,)),
        global_state=random.normal(keys[4], (global_state_dim,)),
        next_global_state=random.normal(keys[5], (global_state_dim,)),
    )


# ============================================================================
# ReplayBuffer Tests
# ============================================================================

class TestReplayBuffer:
    """Tests for ReplayBuffer class."""
    
    def test_init_creates_empty_buffer(self, replay_buffer, buffer_config):
        """Test buffer initialization creates correct shapes."""
        state = replay_buffer.init()
        
        assert state.obs.shape == (buffer_config['capacity'], buffer_config['n_agents'], buffer_config['obs_dim'])
        assert state.actions.shape == (buffer_config['capacity'], buffer_config['n_agents'], buffer_config['action_dim'])
        assert state.rewards.shape == (buffer_config['capacity'], buffer_config['n_agents'])
        assert state.next_obs.shape == (buffer_config['capacity'], buffer_config['n_agents'], buffer_config['obs_dim'])
        assert state.dones.shape == (buffer_config['capacity'], buffer_config['n_agents'])
        assert state.global_state.shape == (buffer_config['capacity'], buffer_config['global_state_dim'])
        assert int(state.position) == 0
        assert int(state.size) == 0
        # New fields should be None by default
        assert state.log_probs is None
        assert state.action_priors is None
    
    def test_init_without_global_state(self, buffer_config):
        """Test buffer initialization without global state."""
        config = buffer_config.copy()
        config['use_global_state'] = False
        buffer = ReplayBuffer(**config)
        state = buffer.init()
        
        assert state.global_state is None
        assert state.next_global_state is None
    
    def test_add_single_transition(self, replay_buffer, buffer_state, sample_transition):
        """Test adding a single transition."""
        new_state = replay_buffer.add(buffer_state, sample_transition)
        
        assert new_state.position == 1
        assert new_state.size == 1
        assert jnp.allclose(new_state.obs[0], sample_transition.obs)
        assert jnp.allclose(new_state.actions[0], sample_transition.actions)
        assert jnp.allclose(new_state.rewards[0], sample_transition.rewards)
    
    def test_add_multiple_transitions(self, replay_buffer, buffer_state, sample_transition, key):
        """Test adding multiple transitions."""
        state = buffer_state
        
        for i in range(10):
            key, subkey = random.split(key)
            transition = Transition(
                obs=random.normal(subkey, sample_transition.obs.shape),
                actions=random.uniform(subkey, sample_transition.actions.shape),
                rewards=random.normal(subkey, sample_transition.rewards.shape),
                next_obs=random.normal(subkey, sample_transition.next_obs.shape),
                dones=jnp.zeros_like(sample_transition.dones),
                global_state=random.normal(subkey, sample_transition.global_state.shape),
                next_global_state=random.normal(subkey, sample_transition.next_global_state.shape),
            )
            state = replay_buffer.add(state, transition)
        
        assert state.position == 10
        assert state.size == 10
    
    def test_circular_buffer_wrapping(self, buffer_config, sample_transition, key):
        """Test buffer wraps around when full."""
        small_config = buffer_config.copy()
        small_config['capacity'] = 5
        buffer = ReplayBuffer(**small_config)
        state = buffer.init()
        
        # Add more transitions than capacity
        for i in range(8):
            key, subkey = random.split(key)
            transition = Transition(
                obs=jnp.ones_like(sample_transition.obs) * i,
                actions=random.uniform(subkey, sample_transition.actions.shape),
                rewards=jnp.ones_like(sample_transition.rewards) * i,
                next_obs=random.normal(subkey, sample_transition.next_obs.shape),
                dones=jnp.zeros_like(sample_transition.dones),
                global_state=random.normal(subkey, sample_transition.global_state.shape),
                next_global_state=random.normal(subkey, sample_transition.next_global_state.shape),
            )
            state = buffer.add(state, transition)
        
        # Position should wrap around: 8 % 5 = 3
        assert state.position == 3
        # Size should be capped at capacity
        assert state.size == 5
    
    def test_sample_batch(self, replay_buffer, buffer_state, sample_transition, key):
        """Test sampling a batch from buffer."""
        state = buffer_state
        
        # Add enough transitions
        for i in range(50):
            key, subkey = random.split(key)
            transition = Transition(
                obs=random.normal(subkey, sample_transition.obs.shape),
                actions=random.uniform(subkey, sample_transition.actions.shape),
                rewards=random.normal(subkey, sample_transition.rewards.shape),
                next_obs=random.normal(subkey, sample_transition.next_obs.shape),
                dones=jnp.zeros_like(sample_transition.dones),
                global_state=random.normal(subkey, sample_transition.global_state.shape),
                next_global_state=random.normal(subkey, sample_transition.next_global_state.shape),
            )
            state = replay_buffer.add(state, transition)
        
        # Sample batch
        key, sample_key = random.split(key)
        batch_size = 32
        batch = replay_buffer.sample(state, sample_key, batch_size)
        
        assert batch.obs.shape == (batch_size, 5, 10)  # (batch, agents, obs_dim)
        assert batch.actions.shape == (batch_size, 5, 2)
        assert batch.rewards.shape == (batch_size, 5)
        assert batch.next_obs.shape == (batch_size, 5, 10)
        assert batch.global_state.shape == (batch_size, 50)
    
    def test_sample_returns_different_batches(self, replay_buffer, buffer_state, sample_transition, key):
        """Test that different keys produce different samples."""
        state = buffer_state
        
        # Fill buffer
        for i in range(100):
            key, subkey = random.split(key)
            transition = Transition(
                obs=random.normal(subkey, sample_transition.obs.shape) * i,
                actions=random.uniform(subkey, sample_transition.actions.shape),
                rewards=random.normal(subkey, sample_transition.rewards.shape),
                next_obs=random.normal(subkey, sample_transition.next_obs.shape),
                dones=jnp.zeros_like(sample_transition.dones),
                global_state=random.normal(subkey, sample_transition.global_state.shape),
                next_global_state=random.normal(subkey, sample_transition.next_global_state.shape),
            )
            state = replay_buffer.add(state, transition)
        
        # Sample with different keys
        key1, key2 = random.split(key)
        batch1 = replay_buffer.sample(state, key1, 16)
        batch2 = replay_buffer.sample(state, key2, 16)
        
        # Batches should be different
        assert not jnp.allclose(batch1.obs, batch2.obs)
    
    def test_can_sample(self, replay_buffer, buffer_state, sample_transition, key):
        """Test can_sample check."""
        state = buffer_state
        
        # Empty buffer
        assert not replay_buffer.can_sample(state, 10)
        
        # Add some transitions
        for i in range(5):
            key, subkey = random.split(key)
            transition = Transition(
                obs=random.normal(subkey, sample_transition.obs.shape),
                actions=random.uniform(subkey, sample_transition.actions.shape),
                rewards=random.normal(subkey, sample_transition.rewards.shape),
                next_obs=random.normal(subkey, sample_transition.next_obs.shape),
                dones=jnp.zeros_like(sample_transition.dones),
                global_state=random.normal(subkey, sample_transition.global_state.shape),
                next_global_state=random.normal(subkey, sample_transition.next_global_state.shape),
            )
            state = replay_buffer.add(state, transition)
        
        assert replay_buffer.can_sample(state, 5)
        assert not replay_buffer.can_sample(state, 10)
    
    def test_is_full(self, buffer_config, sample_transition, key):
        """Test is_full check."""
        small_config = buffer_config.copy()
        small_config['capacity'] = 10
        buffer = ReplayBuffer(**small_config)
        state = buffer.init()
        
        # Not full initially
        assert not buffer.is_full(state)
        
        # Fill it up
        for i in range(10):
            key, subkey = random.split(key)
            transition = Transition(
                obs=random.normal(subkey, sample_transition.obs.shape),
                actions=random.uniform(subkey, sample_transition.actions.shape),
                rewards=random.normal(subkey, sample_transition.rewards.shape),
                next_obs=random.normal(subkey, sample_transition.next_obs.shape),
                dones=jnp.zeros_like(sample_transition.dones),
                global_state=random.normal(subkey, sample_transition.global_state.shape),
                next_global_state=random.normal(subkey, sample_transition.next_global_state.shape),
            )
            state = buffer.add(state, transition)
        
        assert buffer.is_full(state)
    
    def test_add_batch(self, replay_buffer, buffer_state, buffer_config, key):
        """Test batch add functionality."""
        batch_size = 16
        n_agents = buffer_config['n_agents']
        obs_dim = buffer_config['obs_dim']
        action_dim = buffer_config['action_dim']
        global_state_dim = buffer_config['global_state_dim']
        
        keys = random.split(key, 6)
        
        batch_transition = Transition(
            obs=random.normal(keys[0], (batch_size, n_agents, obs_dim)),
            actions=random.uniform(keys[1], (batch_size, n_agents, action_dim)),
            rewards=random.normal(keys[2], (batch_size, n_agents)),
            next_obs=random.normal(keys[3], (batch_size, n_agents, obs_dim)),
            dones=jnp.zeros((batch_size, n_agents)),
            global_state=random.normal(keys[4], (batch_size, global_state_dim)),
            next_global_state=random.normal(keys[5], (batch_size, global_state_dim)),
        )
        
        new_state = replay_buffer.add_batch(buffer_state, batch_transition)
        
        assert new_state.position == batch_size
        assert new_state.size == batch_size


# ============================================================================
# PerAgentReplayBuffer Tests
# ============================================================================

class TestPerAgentReplayBuffer:
    """Tests for single-agent replay buffer."""
    
    @pytest.fixture
    def per_agent_buffer(self):
        return PerAgentReplayBuffer(
            capacity=100,
            obs_dim=10,
            action_dim=2,
        )
    
    @pytest.fixture
    def per_agent_transition(self, key):
        keys = random.split(key, 4)
        return PerAgentTransition(
            obs=random.normal(keys[0], (10,)),
            action=random.uniform(keys[1], (2,)),
            reward=0.5,
            next_obs=random.normal(keys[2], (10,)),
            done=False,
        )
    
    def test_init(self, per_agent_buffer):
        """Test initialization."""
        state = per_agent_buffer.init()
        
        assert state.obs.shape == (100, 10)
        assert state.actions.shape == (100, 2)
        assert state.rewards.shape == (100,)
        assert state.position == 0
        assert state.size == 0
    
    def test_add_transition(self, per_agent_buffer, per_agent_transition):
        """Test adding transition."""
        state = per_agent_buffer.init()
        new_state = per_agent_buffer.add(state, per_agent_transition)
        
        assert new_state.position == 1
        assert new_state.size == 1
        assert jnp.allclose(new_state.obs[0], per_agent_transition.obs)
    
    def test_sample(self, per_agent_buffer, per_agent_transition, key):
        """Test sampling."""
        state = per_agent_buffer.init()
        
        # Fill buffer
        for i in range(50):
            key, subkey = random.split(key)
            transition = PerAgentTransition(
                obs=random.normal(subkey, (10,)),
                action=random.uniform(subkey, (2,)),
                reward=float(i),
                next_obs=random.normal(subkey, (10,)),
                done=False,
            )
            state = per_agent_buffer.add(state, transition)
        
        # Sample
        key, sample_key = random.split(key)
        batch = per_agent_buffer.sample(state, sample_key, 16)
        
        assert batch.obs.shape == (16, 10)
        assert batch.actions.shape == (16, 2)
        assert batch.rewards.shape == (16,)


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_create_replay_buffer(self):
        """Test convenience function."""
        buffer, state = create_replay_buffer(
            capacity=500,
            n_agents=3,
            obs_dim=8,
            action_dim=4,
            global_state_dim=24,
        )
        
        assert isinstance(buffer, ReplayBuffer)
        assert isinstance(state, ReplayBufferState)
        assert state.size == 0
    
    def test_flatten_transition_for_agent(self, key):
        """Test extracting single agent data."""
        batch_size = 32
        n_agents = 5
        obs_dim = 10
        action_dim = 2
        
        keys = random.split(key, 5)
        
        batch = BatchTransition(
            obs=random.normal(keys[0], (batch_size, n_agents, obs_dim)),
            actions=random.uniform(keys[1], (batch_size, n_agents, action_dim)),
            rewards=random.normal(keys[2], (batch_size, n_agents)),
            next_obs=random.normal(keys[3], (batch_size, n_agents, obs_dim)),
            dones=jnp.zeros((batch_size, n_agents)),
        )
        
        agent_batch = flatten_transition_for_agent(batch, agent_idx=2)
        
        assert agent_batch.obs.shape == (batch_size, obs_dim)
        assert agent_batch.actions.shape == (batch_size, action_dim)
        assert jnp.allclose(agent_batch.obs, batch.obs[:, 2])
    
    def test_get_all_actions_flat(self, key):
        """Test flattening all actions."""
        batch_size = 32
        n_agents = 5
        action_dim = 2
        
        batch = BatchTransition(
            obs=jnp.zeros((batch_size, n_agents, 10)),
            actions=random.normal(key, (batch_size, n_agents, action_dim)),
            rewards=jnp.zeros((batch_size, n_agents)),
            next_obs=jnp.zeros((batch_size, n_agents, 10)),
            dones=jnp.zeros((batch_size, n_agents)),
        )
        
        flat_actions = get_all_actions_flat(batch)
        
        assert flat_actions.shape == (batch_size, n_agents * action_dim)
        assert flat_actions.shape == (32, 10)
    
    def test_get_all_obs_flat(self, key):
        """Test flattening all observations."""
        batch_size = 32
        n_agents = 5
        obs_dim = 10
        
        batch = BatchTransition(
            obs=random.normal(key, (batch_size, n_agents, obs_dim)),
            actions=jnp.zeros((batch_size, n_agents, 2)),
            rewards=jnp.zeros((batch_size, n_agents)),
            next_obs=jnp.zeros((batch_size, n_agents, obs_dim)),
            dones=jnp.zeros((batch_size, n_agents)),
        )
        
        flat_obs = get_all_obs_flat(batch)
        
        assert flat_obs.shape == (batch_size, n_agents * obs_dim)
        assert flat_obs.shape == (32, 50)


# ============================================================================
# JIT Compatibility Tests
# ============================================================================

class TestJITCompatibility:
    """Test that operations can be JIT compiled."""
    
    def test_add_jittable(self, replay_buffer, buffer_state, sample_transition):
        """Test add can be JIT compiled."""
        @jax.jit
        def add_jit(state, transition):
            return replay_buffer.add(state, transition)
        
        new_state = add_jit(buffer_state, sample_transition)
        assert new_state.size == 1
    
    def test_sample_jittable(self, replay_buffer, buffer_state, sample_transition, key):
        """Test sample can be JIT compiled."""
        # Fill buffer first
        state = buffer_state
        for i in range(50):
            key, subkey = random.split(key)
            transition = Transition(
                obs=random.normal(subkey, sample_transition.obs.shape),
                actions=random.uniform(subkey, sample_transition.actions.shape),
                rewards=random.normal(subkey, sample_transition.rewards.shape),
                next_obs=random.normal(subkey, sample_transition.next_obs.shape),
                dones=jnp.zeros_like(sample_transition.dones),
                global_state=random.normal(subkey, sample_transition.global_state.shape),
                next_global_state=random.normal(subkey, sample_transition.next_global_state.shape),
            )
            state = replay_buffer.add(state, transition)
        
        @jax.jit
        def sample_jit(state, key):
            return replay_buffer.sample(state, key, 16)
        
        key, sample_key = random.split(key)
        batch = sample_jit(state, sample_key)
        assert batch.obs.shape == (16, 5, 10)
    
    def test_can_sample_jittable(self, replay_buffer, buffer_state, sample_transition):
        """Test can_sample can be used in JIT context."""
        @jax.jit
        def check_can_sample(state):
            return replay_buffer.can_sample(state, 10)
        
        # Empty buffer - should return False
        result = check_can_sample(buffer_state)
        assert not bool(result)
        
        # Fill buffer
        state = buffer_state
        for i in range(15):
            state = replay_buffer.add(state, sample_transition)
        
        # Now should return True
        result = check_can_sample(state)
        assert bool(result)


# ============================================================================
# New Feature Tests
# ============================================================================

class TestNewFeatures:
    """Tests for new buffer features."""
    
    def test_log_probs_storage(self, buffer_config, key):
        """Test storing and retrieving log probabilities."""
        config = buffer_config.copy()
        config['store_log_probs'] = True
        buffer = ReplayBuffer(**config)
        state = buffer.init()
        
        # Check log_probs array is initialized
        assert state.log_probs is not None
        assert state.log_probs.shape == (config['capacity'], config['n_agents'])
        
        # Add transition with log_probs
        keys = random.split(key, 7)
        transition = Transition(
            obs=random.normal(keys[0], (config['n_agents'], config['obs_dim'])),
            actions=random.uniform(keys[1], (config['n_agents'], config['action_dim'])),
            rewards=random.normal(keys[2], (config['n_agents'],)),
            next_obs=random.normal(keys[3], (config['n_agents'], config['obs_dim'])),
            dones=jnp.zeros((config['n_agents'],)),
            global_state=random.normal(keys[4], (config['global_state_dim'],)),
            next_global_state=random.normal(keys[5], (config['global_state_dim'],)),
            log_probs=random.normal(keys[6], (config['n_agents'],)),
        )
        
        state = buffer.add(state, transition)
        assert jnp.allclose(state.log_probs[0], transition.log_probs)
        
        # Sample and check log_probs returned
        batch = buffer.sample(state, key, 1)
        assert batch.log_probs is not None
        assert batch.log_probs.shape == (1, config['n_agents'])
    
    def test_action_priors_storage(self, buffer_config, key):
        """Test storing and retrieving action priors."""
        config = buffer_config.copy()
        config['store_action_priors'] = True
        buffer = ReplayBuffer(**config)
        state = buffer.init()
        
        # Check action_priors array is initialized
        assert state.action_priors is not None
        assert state.action_priors.shape == (config['capacity'], config['n_agents'], config['action_dim'])
        
        # Add transition with action_priors
        keys = random.split(key, 7)
        transition = Transition(
            obs=random.normal(keys[0], (config['n_agents'], config['obs_dim'])),
            actions=random.uniform(keys[1], (config['n_agents'], config['action_dim'])),
            rewards=random.normal(keys[2], (config['n_agents'],)),
            next_obs=random.normal(keys[3], (config['n_agents'], config['obs_dim'])),
            dones=jnp.zeros((config['n_agents'],)),
            global_state=random.normal(keys[4], (config['global_state_dim'],)),
            next_global_state=random.normal(keys[5], (config['global_state_dim'],)),
            action_priors=random.uniform(keys[6], (config['n_agents'], config['action_dim'])),
        )
        
        state = buffer.add(state, transition)
        assert jnp.allclose(state.action_priors[0], transition.action_priors)
        
        # Sample and check action_priors returned
        batch = buffer.sample(state, key, 1)
        assert batch.action_priors is not None
        assert batch.action_priors.shape == (1, config['n_agents'], config['action_dim'])
    
    def test_sample_without_replacement(self, replay_buffer, buffer_state, sample_transition, key):
        """Test sampling without replacement."""
        state = buffer_state
        
        # Fill buffer with unique values
        for i in range(100):
            key, subkey = random.split(key)
            transition = Transition(
                obs=jnp.ones_like(sample_transition.obs) * i,  # Unique identifier
                actions=random.uniform(subkey, sample_transition.actions.shape),
                rewards=random.normal(subkey, sample_transition.rewards.shape),
                next_obs=random.normal(subkey, sample_transition.next_obs.shape),
                dones=jnp.zeros_like(sample_transition.dones),
                global_state=random.normal(subkey, sample_transition.global_state.shape),
                next_global_state=random.normal(subkey, sample_transition.next_global_state.shape),
            )
            state = replay_buffer.add(state, transition)
        
        # Sample without replacement
        key, sample_key = random.split(key)
        batch = replay_buffer.sample_without_replacement(state, sample_key, 50)
        
        # Check no duplicates by looking at obs values (each is unique)
        obs_sums = batch.obs.sum(axis=(1, 2))  # Should all be different
        unique_count = len(jnp.unique(obs_sums))
        assert unique_count == 50  # All samples should be unique
    
    def test_reset(self, replay_buffer, buffer_state, sample_transition):
        """Test buffer reset."""
        state = buffer_state
        
        # Fill buffer
        for i in range(50):
            state = replay_buffer.add(state, sample_transition)
        
        assert int(state.size) == 50
        assert int(state.position) == 50
        
        # Reset
        reset_state = replay_buffer.reset(state)
        
        assert int(reset_state.size) == 0
        assert int(reset_state.position) == 0
        # Data arrays should still exist (not reallocated)
        assert reset_state.obs.shape == state.obs.shape
    
    def test_get_average_rewards(self, replay_buffer, buffer_state, sample_transition, key):
        """Test average rewards calculation."""
        state = buffer_state
        
        # Add transitions with known rewards
        for i in range(20):
            key, subkey = random.split(key)
            transition = Transition(
                obs=random.normal(subkey, sample_transition.obs.shape),
                actions=random.uniform(subkey, sample_transition.actions.shape),
                rewards=jnp.ones_like(sample_transition.rewards) * i,  # rewards = i
                next_obs=random.normal(subkey, sample_transition.next_obs.shape),
                dones=jnp.zeros_like(sample_transition.dones),
                global_state=random.normal(subkey, sample_transition.global_state.shape),
                next_global_state=random.normal(subkey, sample_transition.next_global_state.shape),
            )
            state = replay_buffer.add(state, transition)
        
        avg_rewards = replay_buffer.get_average_rewards(state)
        
        # Average of 0..19 is 9.5
        assert avg_rewards.shape == (5,)  # n_agents
        assert jnp.allclose(avg_rewards, 9.5)
    
    def test_dtype_specification(self, buffer_config):
        """Test custom dtype."""
        config = buffer_config.copy()
        config['dtype'] = jnp.float16
        buffer = ReplayBuffer(**config)
        state = buffer.init()
        
        assert state.obs.dtype == jnp.float16
        assert state.actions.dtype == jnp.float16
        assert state.rewards.dtype == jnp.float16
    
    def test_per_agent_buffer_can_sample(self, key):
        """Test PerAgentReplayBuffer can_sample method."""
        buffer = PerAgentReplayBuffer(capacity=100, obs_dim=10, action_dim=2)
        state = buffer.init()
        
        # Empty buffer
        assert not bool(buffer.can_sample(state, 10))
        
        # Add some transitions
        for i in range(15):
            key, subkey = random.split(key)
            transition = PerAgentTransition(
                obs=random.normal(subkey, (10,)),
                action=random.uniform(subkey, (2,)),
                reward=float(i),
                next_obs=random.normal(subkey, (10,)),
                done=False,
            )
            state = buffer.add(state, transition)
        
        assert bool(buffer.can_sample(state, 10))
        assert bool(buffer.can_sample(state, 15))
        assert not bool(buffer.can_sample(state, 20))
    
    def test_per_agent_buffer_reset(self, key):
        """Test PerAgentReplayBuffer reset method."""
        buffer = PerAgentReplayBuffer(capacity=100, obs_dim=10, action_dim=2)
        state = buffer.init()
        
        # Add some transitions
        for i in range(15):
            key, subkey = random.split(key)
            transition = PerAgentTransition(
                obs=random.normal(subkey, (10,)),
                action=random.uniform(subkey, (2,)),
                reward=float(i),
                next_obs=random.normal(subkey, (10,)),
                done=False,
            )
            state = buffer.add(state, transition)
        
        assert int(state.size) == 15
        
        reset_state = buffer.reset(state)
        assert int(reset_state.size) == 0
        assert int(reset_state.position) == 0


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x'])
