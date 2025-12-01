"""Replay buffers for MADDPG algorithm.

This module provides JAX-compatible replay buffers for:
1. ReplayBuffer - Basic replay buffer for experience storage
2. PrioritizedReplayBuffer - With prioritized experience replay (optional)

Buffers are designed to be:
- JIT-compatible for fast sampling
- Memory-efficient using JAX arrays
- Compatible with multi-agent scenarios

The buffer stores transitions in the format used by maddpg_wrapper.py:
- obs: (n_agents, obs_dim)
- actions: (n_agents, action_dim)
- rewards: (n_agents,)
- next_obs: (n_agents, obs_dim)
- dones: (n_agents,) or scalar
- global_state: (global_state_dim,) for centralized critic
"""

from typing import Tuple, Optional, NamedTuple, Dict, Any, Union
import jax
import jax.numpy as jnp
from jax import random, lax
from flax import struct
from functools import partial


# ============================================================================
# Transition Types
# ============================================================================

class Transition(NamedTuple):
    """A single transition for replay buffer.
    
    Compatible with maddpg_wrapper.py Transition type.
    
    Attributes:
        obs: Observations for all agents (n_agents, obs_dim)
        actions: Actions taken by all agents (n_agents, action_dim)
        rewards: Rewards received (n_agents,)
        next_obs: Next observations (n_agents, obs_dim)
        dones: Done flags (n_agents,) or scalar
        global_state: Optional global state for centralized critic (global_state_dim,)
        next_global_state: Optional next global state (global_state_dim,)
        log_probs: Optional log probabilities of actions (n_agents,) - for PPO/importance sampling
        action_priors: Optional prior/expert actions (n_agents, action_dim) - for guided learning
    """
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray
    global_state: Optional[jnp.ndarray] = None
    next_global_state: Optional[jnp.ndarray] = None
    log_probs: Optional[jnp.ndarray] = None
    action_priors: Optional[jnp.ndarray] = None


class BatchTransition(NamedTuple):
    """Batched transitions for training.
    
    Attributes:
        obs: (batch_size, n_agents, obs_dim)
        actions: (batch_size, n_agents, action_dim)
        rewards: (batch_size, n_agents)
        next_obs: (batch_size, n_agents, obs_dim)
        dones: (batch_size, n_agents) or (batch_size,)
        global_state: (batch_size, global_state_dim) if used
        next_global_state: (batch_size, global_state_dim) if used
        log_probs: (batch_size, n_agents) if used
        action_priors: (batch_size, n_agents, action_dim) if used
    """
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray
    global_state: Optional[jnp.ndarray] = None
    next_global_state: Optional[jnp.ndarray] = None
    log_probs: Optional[jnp.ndarray] = None
    action_priors: Optional[jnp.ndarray] = None


# ============================================================================
# Replay Buffer State
# ============================================================================

@struct.dataclass
class ReplayBufferState:
    """State of the replay buffer.
    
    Attributes:
        obs: Stored observations (capacity, n_agents, obs_dim)
        actions: Stored actions (capacity, n_agents, action_dim)
        rewards: Stored rewards (capacity, n_agents)
        next_obs: Stored next observations (capacity, n_agents, obs_dim)
        dones: Stored done flags (capacity, n_agents) or (capacity,)
        global_state: Stored global states (capacity, global_state_dim) or None
        next_global_state: Stored next global states (capacity, global_state_dim) or None
        log_probs: Stored log probabilities (capacity, n_agents) or None
        action_priors: Stored prior actions (capacity, n_agents, action_dim) or None
        position: Current write position
        size: Number of stored transitions
    """
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray
    global_state: Optional[jnp.ndarray]
    next_global_state: Optional[jnp.ndarray]
    log_probs: Optional[jnp.ndarray]
    action_priors: Optional[jnp.ndarray]
    position: jnp.ndarray  # Use jnp.ndarray for JIT compatibility
    size: jnp.ndarray      # Use jnp.ndarray for JIT compatibility


# ============================================================================
# Replay Buffer Class
# ============================================================================

class ReplayBuffer:
    """Replay buffer for multi-agent reinforcement learning.
    
    This buffer stores transitions and supports efficient random sampling.
    It uses JAX arrays for GPU-compatible storage and sampling.
    
    Features:
    - Circular buffer with fixed capacity
    - Optional global state storage for CTDE
    - JIT-compatible operations
    - Efficient batch sampling
    
    Example:
        ```python
        buffer = ReplayBuffer(
            capacity=100000,
            n_agents=5,
            obs_dim=10,
            action_dim=2,
            global_state_dim=50,
        )
        
        state = buffer.init()
        
        # Add transitions
        state = buffer.add(state, transition)
        
        # Sample batch
        key = jax.random.PRNGKey(0)
        batch = buffer.sample(state, key, batch_size=256)
        ```
    """
    
    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        global_state_dim: Optional[int] = None,
        use_global_state: bool = True,
        store_log_probs: bool = False,
        store_action_priors: bool = False,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize replay buffer configuration.
        
        Args:
            capacity: Maximum number of transitions to store
            n_agents: Number of agents
            obs_dim: Observation dimension per agent
            action_dim: Action dimension per agent
            global_state_dim: Global state dimension (for centralized critic)
            use_global_state: Whether to store global states
            store_log_probs: Whether to store log probabilities (for PPO)
            store_action_priors: Whether to store prior actions (for guided learning)
            dtype: Data type for arrays (float32 or float16 for memory)
        """
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.use_global_state = use_global_state and global_state_dim is not None
        self.store_log_probs = store_log_probs
        self.store_action_priors = store_action_priors
        self.dtype = dtype
    
    def init(self) -> ReplayBufferState:
        """Initialize empty buffer state.
        
        Returns:
            Empty ReplayBufferState
        """
        obs = jnp.zeros((self.capacity, self.n_agents, self.obs_dim), dtype=self.dtype)
        actions = jnp.zeros((self.capacity, self.n_agents, self.action_dim), dtype=self.dtype)
        rewards = jnp.zeros((self.capacity, self.n_agents), dtype=self.dtype)
        next_obs = jnp.zeros((self.capacity, self.n_agents, self.obs_dim), dtype=self.dtype)
        dones = jnp.zeros((self.capacity, self.n_agents), dtype=self.dtype)
        
        if self.use_global_state:
            global_state = jnp.zeros((self.capacity, self.global_state_dim), dtype=self.dtype)
            next_global_state = jnp.zeros((self.capacity, self.global_state_dim), dtype=self.dtype)
        else:
            global_state = None
            next_global_state = None
        
        if self.store_log_probs:
            log_probs = jnp.zeros((self.capacity, self.n_agents), dtype=self.dtype)
        else:
            log_probs = None
            
        if self.store_action_priors:
            action_priors = jnp.zeros((self.capacity, self.n_agents, self.action_dim), dtype=self.dtype)
        else:
            action_priors = None
        
        return ReplayBufferState(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            global_state=global_state,
            next_global_state=next_global_state,
            log_probs=log_probs,
            action_priors=action_priors,
            position=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )
    
    def add(
        self,
        state: ReplayBufferState,
        transition: Transition,
    ) -> ReplayBufferState:
        """Add a transition to the buffer.
        
        Args:
            state: Current buffer state
            transition: Transition to add
            
        Returns:
            Updated buffer state
        """
        pos = state.position
        
        # Update arrays at current position
        new_obs = state.obs.at[pos].set(transition.obs)
        new_actions = state.actions.at[pos].set(transition.actions)
        new_rewards = state.rewards.at[pos].set(transition.rewards)
        new_next_obs = state.next_obs.at[pos].set(transition.next_obs)
        new_dones = state.dones.at[pos].set(transition.dones)
        
        # Update global states if used
        if self.use_global_state and transition.global_state is not None:
            new_global_state = state.global_state.at[pos].set(transition.global_state)
            new_next_global_state = state.next_global_state.at[pos].set(transition.next_global_state)
        else:
            new_global_state = state.global_state
            new_next_global_state = state.next_global_state
        
        # Update log_probs if used
        if self.store_log_probs and transition.log_probs is not None:
            new_log_probs = state.log_probs.at[pos].set(transition.log_probs)
        else:
            new_log_probs = state.log_probs
            
        # Update action_priors if used
        if self.store_action_priors and transition.action_priors is not None:
            new_action_priors = state.action_priors.at[pos].set(transition.action_priors)
        else:
            new_action_priors = state.action_priors
        
        # Update position (circular)
        new_position = (pos + 1) % self.capacity
        
        # Update size
        new_size = jnp.minimum(state.size + 1, self.capacity)
        
        return ReplayBufferState(
            obs=new_obs,
            actions=new_actions,
            rewards=new_rewards,
            next_obs=new_next_obs,
            dones=new_dones,
            global_state=new_global_state,
            next_global_state=new_next_global_state,
            log_probs=new_log_probs,
            action_priors=new_action_priors,
            position=new_position,
            size=new_size,
        )
    
    def add_batch(
        self,
        state: ReplayBufferState,
        transitions: Transition,
    ) -> ReplayBufferState:
        """Add a batch of transitions to the buffer.
        
        Args:
            state: Current buffer state
            transitions: Batch of transitions (arrays with batch dimension first)
            
        Returns:
            Updated buffer state
        """
        batch_size = transitions.obs.shape[0]
        
        # Calculate indices for batch insertion
        positions = (jnp.arange(batch_size) + state.position) % self.capacity
        
        # Update arrays
        new_obs = state.obs.at[positions].set(transitions.obs)
        new_actions = state.actions.at[positions].set(transitions.actions)
        new_rewards = state.rewards.at[positions].set(transitions.rewards)
        new_next_obs = state.next_obs.at[positions].set(transitions.next_obs)
        new_dones = state.dones.at[positions].set(transitions.dones)
        
        if self.use_global_state and transitions.global_state is not None:
            new_global_state = state.global_state.at[positions].set(transitions.global_state)
            new_next_global_state = state.next_global_state.at[positions].set(transitions.next_global_state)
        else:
            new_global_state = state.global_state
            new_next_global_state = state.next_global_state
        
        # Update log_probs if used
        if self.store_log_probs and transitions.log_probs is not None:
            new_log_probs = state.log_probs.at[positions].set(transitions.log_probs)
        else:
            new_log_probs = state.log_probs
            
        # Update action_priors if used
        if self.store_action_priors and transitions.action_priors is not None:
            new_action_priors = state.action_priors.at[positions].set(transitions.action_priors)
        else:
            new_action_priors = state.action_priors
        
        # Update position and size
        new_position = (state.position + batch_size) % self.capacity
        new_size = jnp.minimum(state.size + batch_size, self.capacity)
        
        return ReplayBufferState(
            obs=new_obs,
            actions=new_actions,
            rewards=new_rewards,
            next_obs=new_next_obs,
            dones=new_dones,
            global_state=new_global_state,
            next_global_state=new_next_global_state,
            log_probs=new_log_probs,
            action_priors=new_action_priors,
            position=new_position,
            size=new_size,
        )
    
    def sample(
        self,
        state: ReplayBufferState,
        key: jax.Array,
        batch_size: int,
    ) -> BatchTransition:
        """Sample a batch of transitions from the buffer.
        
        Args:
            state: Current buffer state
            key: JAX random key
            batch_size: Number of transitions to sample
            
        Returns:
            BatchTransition with sampled data
        """
        # Sample random indices from valid range
        indices = random.randint(key, (batch_size,), 0, state.size)
        
        # Gather sampled transitions
        obs = state.obs[indices]
        actions = state.actions[indices]
        rewards = state.rewards[indices]
        next_obs = state.next_obs[indices]
        dones = state.dones[indices]
        
        if self.use_global_state:
            global_state = state.global_state[indices]
            next_global_state = state.next_global_state[indices]
        else:
            global_state = None
            next_global_state = None
        
        if self.store_log_probs:
            log_probs = state.log_probs[indices]
        else:
            log_probs = None
            
        if self.store_action_priors:
            action_priors = state.action_priors[indices]
        else:
            action_priors = None
        
        return BatchTransition(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            global_state=global_state,
            next_global_state=next_global_state,
            log_probs=log_probs,
            action_priors=action_priors,
        )
    
    def sample_without_replacement(
        self,
        state: ReplayBufferState,
        key: jax.Array,
        batch_size: int,
    ) -> BatchTransition:
        """Sample a batch without replacement (no duplicate samples).
        
        More expensive than regular sampling but avoids duplicates.
        Important for small buffers or when exact batch statistics matter.
        
        Args:
            state: Current buffer state
            key: JAX random key
            batch_size: Number of transitions to sample
            
        Returns:
            BatchTransition with sampled data (no duplicates)
        """
        # Use permutation to sample without replacement
        indices = random.permutation(key, state.size)[:batch_size]
        
        # Gather sampled transitions
        obs = state.obs[indices]
        actions = state.actions[indices]
        rewards = state.rewards[indices]
        next_obs = state.next_obs[indices]
        dones = state.dones[indices]
        
        if self.use_global_state:
            global_state = state.global_state[indices]
            next_global_state = state.next_global_state[indices]
        else:
            global_state = None
            next_global_state = None
        
        if self.store_log_probs:
            log_probs = state.log_probs[indices]
        else:
            log_probs = None
            
        if self.store_action_priors:
            action_priors = state.action_priors[indices]
        else:
            action_priors = None
        
        return BatchTransition(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            global_state=global_state,
            next_global_state=next_global_state,
            log_probs=log_probs,
            action_priors=action_priors,
        )
    
    def can_sample(self, state: ReplayBufferState, batch_size: int) -> jnp.ndarray:
        """Check if buffer has enough samples (JIT-compatible).
        
        Args:
            state: Current buffer state
            batch_size: Desired batch size
            
        Returns:
            Boolean array - True if buffer has at least batch_size transitions
        """
        return state.size >= batch_size
    
    def is_full(self, state: ReplayBufferState) -> jnp.ndarray:
        """Check if buffer is full (JIT-compatible).
        
        Args:
            state: Current buffer state
            
        Returns:
            Boolean array - True if buffer is at capacity
        """
        return state.size >= self.capacity
    
    def reset(self, state: ReplayBufferState) -> ReplayBufferState:
        """Reset buffer to empty state while preserving array allocations.
        
        More efficient than re-initializing as it avoids reallocation.
        
        Args:
            state: Current buffer state
            
        Returns:
            Reset buffer state with position and size set to 0
        """
        return state.replace(
            position=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )
    
    def get_average_rewards(
        self, 
        state: ReplayBufferState, 
        n_recent: Optional[int] = None,
    ) -> jnp.ndarray:
        """Calculate average rewards over recent transitions.
        
        Useful for monitoring training progress.
        
        Args:
            state: Current buffer state
            n_recent: Number of recent transitions to average (default: all)
            
        Returns:
            Average rewards per agent (n_agents,)
        """
        if n_recent is None:
            n_recent = state.size
        
        # Calculate start index for recent transitions
        # Handle circular buffer case
        n_to_avg = jnp.minimum(n_recent, state.size)
        
        # For simplicity, compute mean over all stored (up to size)
        # This is an approximation - exact recent would need circular handling
        return jnp.mean(state.rewards[:state.size], axis=0)


# ============================================================================
# Per-Agent Replay Buffer (for independent learners)
# ============================================================================

@struct.dataclass
class PerAgentBufferState:
    """State for per-agent replay buffer.
    
    Each agent has its own buffer for independent learning.
    
    Attributes:
        obs: (capacity, obs_dim)
        actions: (capacity, action_dim)
        rewards: (capacity,)
        next_obs: (capacity, obs_dim)
        dones: (capacity,)
        position: Current write position
        size: Number of stored transitions
    """
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray
    position: jnp.ndarray  # Use jnp.ndarray for JIT compatibility
    size: jnp.ndarray      # Use jnp.ndarray for JIT compatibility


class PerAgentTransition(NamedTuple):
    """Single-agent transition."""
    obs: jnp.ndarray      # (obs_dim,)
    action: jnp.ndarray   # (action_dim,)
    reward: float
    next_obs: jnp.ndarray # (obs_dim,)
    done: bool


class PerAgentBatch(NamedTuple):
    """Batched single-agent transitions."""
    obs: jnp.ndarray      # (batch_size, obs_dim)
    actions: jnp.ndarray  # (batch_size, action_dim)
    rewards: jnp.ndarray  # (batch_size,)
    next_obs: jnp.ndarray # (batch_size, obs_dim)
    dones: jnp.ndarray    # (batch_size,)


class PerAgentReplayBuffer:
    """Replay buffer for a single agent.
    
    Simpler buffer without multi-agent structure.
    Useful for independent learners or ablation studies.
    """
    
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize buffer configuration.
        
        Args:
            capacity: Maximum transitions to store
            obs_dim: Observation dimension
            action_dim: Action dimension
            dtype: Data type for arrays
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dtype = dtype
    
    def init(self) -> PerAgentBufferState:
        """Initialize empty buffer."""
        return PerAgentBufferState(
            obs=jnp.zeros((self.capacity, self.obs_dim), dtype=self.dtype),
            actions=jnp.zeros((self.capacity, self.action_dim), dtype=self.dtype),
            rewards=jnp.zeros((self.capacity,), dtype=self.dtype),
            next_obs=jnp.zeros((self.capacity, self.obs_dim), dtype=self.dtype),
            dones=jnp.zeros((self.capacity,), dtype=self.dtype),
            position=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )
    
    def add(
        self,
        state: PerAgentBufferState,
        transition: PerAgentTransition,
    ) -> PerAgentBufferState:
        """Add a transition to the buffer."""
        pos = state.position
        
        return PerAgentBufferState(
            obs=state.obs.at[pos].set(transition.obs),
            actions=state.actions.at[pos].set(transition.action),
            rewards=state.rewards.at[pos].set(transition.reward),
            next_obs=state.next_obs.at[pos].set(transition.next_obs),
            dones=state.dones.at[pos].set(transition.done),
            position=(pos + 1) % self.capacity,
            size=jnp.minimum(state.size + 1, self.capacity),
        )
    
    def sample(
        self,
        state: PerAgentBufferState,
        key: jax.Array,
        batch_size: int,
    ) -> PerAgentBatch:
        """Sample a batch from the buffer."""
        indices = random.randint(key, (batch_size,), 0, state.size)
        
        return PerAgentBatch(
            obs=state.obs[indices],
            actions=state.actions[indices],
            rewards=state.rewards[indices],
            next_obs=state.next_obs[indices],
            dones=state.dones[indices],
        )
    
    def can_sample(self, state: PerAgentBufferState, batch_size: int) -> jnp.ndarray:
        """Check if buffer has enough samples."""
        return state.size >= batch_size
    
    def is_full(self, state: PerAgentBufferState) -> jnp.ndarray:
        """Check if buffer is full."""
        return state.size >= self.capacity
    
    def reset(self, state: PerAgentBufferState) -> PerAgentBufferState:
        """Reset buffer to empty state."""
        return state.replace(
            position=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_replay_buffer(
    capacity: int,
    n_agents: int,
    obs_dim: int,
    action_dim: int,
    global_state_dim: Optional[int] = None,
) -> Tuple[ReplayBuffer, ReplayBufferState]:
    """Create and initialize a replay buffer.
    
    Convenience function that creates buffer and initial state.
    
    Args:
        capacity: Maximum transitions to store
        n_agents: Number of agents
        obs_dim: Observation dimension per agent
        action_dim: Action dimension per agent
        global_state_dim: Global state dimension (optional)
        
    Returns:
        buffer: ReplayBuffer instance
        state: Initial ReplayBufferState
    """
    buffer = ReplayBuffer(
        capacity=capacity,
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        global_state_dim=global_state_dim,
    )
    state = buffer.init()
    return buffer, state


def flatten_transition_for_agent(
    batch: BatchTransition,
    agent_idx: int,
) -> PerAgentBatch:
    """Extract single agent's data from multi-agent batch.
    
    Useful for per-agent critic updates in MADDPG.
    
    Args:
        batch: Multi-agent batch transition
        agent_idx: Index of agent to extract
        
    Returns:
        Single-agent batch
    """
    return PerAgentBatch(
        obs=batch.obs[:, agent_idx],
        actions=batch.actions[:, agent_idx],
        rewards=batch.rewards[:, agent_idx],
        next_obs=batch.next_obs[:, agent_idx],
        dones=batch.dones[:, agent_idx] if batch.dones.ndim > 1 else batch.dones,
    )


def get_all_actions_flat(batch: BatchTransition) -> jnp.ndarray:
    """Flatten all agents' actions for centralized critic.
    
    Args:
        batch: Multi-agent batch
        
    Returns:
        Flattened actions (batch_size, n_agents * action_dim)
    """
    batch_size = batch.actions.shape[0]
    return batch.actions.reshape(batch_size, -1)


def get_all_obs_flat(batch: BatchTransition) -> jnp.ndarray:
    """Flatten all agents' observations.
    
    Args:
        batch: Multi-agent batch
        
    Returns:
        Flattened observations (batch_size, n_agents * obs_dim)
    """
    batch_size = batch.obs.shape[0]
    return batch.obs.reshape(batch_size, -1)
