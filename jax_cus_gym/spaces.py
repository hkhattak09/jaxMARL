"""JAX-compatible space definitions for multi-agent environments.

These spaces are designed to be:
1. Fully compatible with JAX transformations (jit, vmap, etc.)
2. Immutable (using frozen dataclasses)
3. Support multi-agent scenarios with batched sampling

Key differences from gymnax spaces:
- Added MultiAgentActionSpace and MultiAgentObservationSpace for MADDPG compatibility
- All shapes and bounds are stored as static values for JIT compilation
"""

from typing import Tuple, Union, Any
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Space:
    """Base class for all spaces. Uses flax.struct for immutability."""
    
    def sample(self, key: jax.Array) -> jax.Array:
        """Sample a random element from this space."""
        raise NotImplementedError
    
    def contains(self, x: jax.Array) -> jax.Array:
        """Check if x is a valid member of this space."""
        raise NotImplementedError


@struct.dataclass
class Discrete(Space):
    """Discrete space {0, 1, ..., n-1}.
    
    Attributes:
        n: Number of discrete elements in the space.
    """
    n: int
    
    def sample(self, key: jax.Array) -> jax.Array:
        """Sample a random integer in [0, n)."""
        return jax.random.randint(key, shape=(), minval=0, maxval=self.n, dtype=jnp.int32)
    
    def contains(self, x: jax.Array) -> jax.Array:
        """Check if x is in [0, n)."""
        x_int = x.astype(jnp.int32)
        return jnp.logical_and(x_int >= 0, x_int < self.n)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return ()
    
    @property
    def dtype(self):
        return jnp.int32


@struct.dataclass 
class Box(Space):
    """Continuous box space in R^n.
    
    Represents a bounded or unbounded continuous space.
    
    Attributes:
        low: Lower bound (scalar or array)
        high: Upper bound (scalar or array)  
        shape: Shape of the space
    """
    low: Union[float, jax.Array]
    high: Union[float, jax.Array]
    shape: Tuple[int, ...]
    
    def sample(self, key: jax.Array) -> jax.Array:
        """Sample uniformly from the box."""
        return jax.random.uniform(
            key, 
            shape=self.shape, 
            minval=self.low, 
            maxval=self.high,
            dtype=jnp.float32
        )
    
    def contains(self, x: jax.Array) -> jax.Array:
        """Check if all elements of x are within bounds."""
        above_low = jnp.all(x >= self.low)
        below_high = jnp.all(x <= self.high)
        return jnp.logical_and(above_low, below_high)
    
    @property
    def dtype(self):
        return jnp.float32


@struct.dataclass
class MultiAgentActionSpace(Space):
    """Action space for multiple agents.
    
    Each agent has the same action space (Box with shape (action_dim,)).
    The combined action shape is (n_agents, action_dim).
    
    Attributes:
        n_agents: Number of agents
        action_dim: Dimension of each agent's action
        low: Lower bound for actions
        high: Upper bound for actions
    """
    n_agents: int
    action_dim: int
    low: float = -1.0
    high: float = 1.0
    
    def sample(self, key: jax.Array) -> jax.Array:
        """Sample actions for all agents. Shape: (n_agents, action_dim)"""
        return jax.random.uniform(
            key,
            shape=(self.n_agents, self.action_dim),
            minval=self.low,
            maxval=self.high,
            dtype=jnp.float32
        )
    
    def sample_single(self, key: jax.Array) -> jax.Array:
        """Sample action for a single agent. Shape: (action_dim,)"""
        return jax.random.uniform(
            key,
            shape=(self.action_dim,),
            minval=self.low,
            maxval=self.high,
            dtype=jnp.float32
        )
    
    def contains(self, x: jax.Array) -> jax.Array:
        """Check if actions are valid for all agents."""
        above_low = jnp.all(x >= self.low)
        below_high = jnp.all(x <= self.high)
        correct_shape = x.shape == (self.n_agents, self.action_dim)
        return jnp.logical_and(jnp.logical_and(above_low, below_high), correct_shape)
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_agents, self.action_dim)
    
    @property
    def single_agent_shape(self) -> Tuple[int]:
        return (self.action_dim,)
    
    @property
    def dtype(self):
        return jnp.float32


@struct.dataclass
class MultiAgentObservationSpace(Space):
    """Observation space for multiple agents.
    
    Each agent has its own observation of shape (obs_dim,).
    The combined observation shape is (n_agents, obs_dim).
    
    Attributes:
        n_agents: Number of agents
        obs_dim: Dimension of each agent's observation
        low: Lower bound for observations
        high: Upper bound for observations
    """
    n_agents: int
    obs_dim: int
    low: float = -jnp.inf
    high: float = jnp.inf
    
    def sample(self, key: jax.Array) -> jax.Array:
        """Sample observations for all agents (typically not used)."""
        # Clip to reasonable range for sampling
        low_clip = jnp.maximum(self.low, -1e6)
        high_clip = jnp.minimum(self.high, 1e6)
        return jax.random.uniform(
            key,
            shape=(self.n_agents, self.obs_dim),
            minval=low_clip,
            maxval=high_clip,
            dtype=jnp.float32
        )
    
    def contains(self, x: jax.Array) -> jax.Array:
        """Check if observations are valid."""
        above_low = jnp.all(x >= self.low)
        below_high = jnp.all(x <= self.high)
        correct_shape = x.shape == (self.n_agents, self.obs_dim)
        return jnp.logical_and(jnp.logical_and(above_low, below_high), correct_shape)
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_agents, self.obs_dim)
    
    @property
    def single_agent_shape(self) -> Tuple[int]:
        return (self.obs_dim,)
    
    @property
    def dtype(self):
        return jnp.float32


# Utility functions for space conversion (useful for interfacing with other libraries)
def get_agent_action_space(multi_space: MultiAgentActionSpace, agent_idx: int) -> Box:
    """Get the action space for a single agent."""
    return Box(
        low=multi_space.low,
        high=multi_space.high,
        shape=(multi_space.action_dim,)
    )


def get_agent_obs_space(multi_space: MultiAgentObservationSpace, agent_idx: int) -> Box:
    """Get the observation space for a single agent."""
    return Box(
        low=multi_space.low,
        high=multi_space.high,
        shape=(multi_space.obs_dim,)
    )
