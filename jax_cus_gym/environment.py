"""Abstract base class for multi-agent JAX environments.

This module defines the core interface for multi-agent environments that are:
1. Fully compatible with JAX transformations (jit, vmap, grad)
2. Stateless - all state is passed explicitly
3. Compatible with MADDPG and other MARL algorithms

Key design principles:
- EnvState contains ALL mutable state (positions, velocities, time, etc.)
- EnvParams contains ALL configurable parameters (immutable during episode)
- All methods are pure functions (no side effects)
- The step() function handles auto-reset internally
"""

from functools import partial
from typing import Any, Generic, TypeVar, Tuple, Dict
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import struct

from spaces import MultiAgentActionSpace, MultiAgentObservationSpace


# Type variables for generic typing
TEnvState = TypeVar("TEnvState", bound="EnvState")
TEnvParams = TypeVar("TEnvParams", bound="EnvParams")


@struct.dataclass
class EnvState:
    """Base class for environment state.
    
    All environment states must inherit from this and include a `time` field.
    States are immutable - updates create new state objects.
    
    Attributes:
        time: Current timestep in the episode
    """
    time: int


@struct.dataclass
class EnvParams:
    """Base class for environment parameters.
    
    Parameters are fixed for the duration of an episode.
    They define the environment configuration.
    
    Attributes:
        max_steps_in_episode: Maximum steps before episode terminates
    """
    max_steps_in_episode: int = 500


class MultiAgentEnv(Generic[TEnvState, TEnvParams]):
    """Abstract base class for multi-agent environments.
    
    This class defines the interface that all multi-agent environments must implement.
    It follows the gymnax pattern but extends it for multi-agent scenarios.
    
    Key differences from single-agent environments:
    - Observations have shape (n_agents, obs_dim)
    - Actions have shape (n_agents, action_dim)
    - Rewards have shape (n_agents,) - each agent gets its own reward
    - Done is a scalar (episode ends for all agents simultaneously)
    
    Subclasses must implement:
    - step_env(): Environment-specific step logic
    - reset_env(): Environment-specific reset logic
    - get_obs(): Compute observations from state
    - is_terminal(): Check if episode should end
    - action_space(): Return the action space
    - observation_space(): Return the observation space
    """
    
    def __init__(self):
        """Initialize the environment."""
        pass
    
    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters."""
        return EnvParams()
    
    @property
    @abstractmethod
    def n_agents(self) -> int:
        """Number of agents in the environment."""
        raise NotImplementedError
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: jax.Array,
        params: TEnvParams | None = None,
    ) -> Tuple[jax.Array, TEnvState, jax.Array, jax.Array, Dict[str, Any]]:
        """Perform a step in the environment.
        
        This method handles the step transition and auto-reset logic.
        When done=True, the environment automatically resets.
        
        Args:
            key: JAX random key
            state: Current environment state
            action: Actions for all agents, shape (n_agents, action_dim)
            params: Environment parameters (uses default if None)
            
        Returns:
            obs: Observations for all agents, shape (n_agents, obs_dim)
            state: New environment state
            reward: Rewards for all agents, shape (n_agents,)
            done: Whether episode is done (scalar bool)
            info: Additional information dictionary
        """
        if params is None:
            params = self.default_params
        
        # Split key for step and potential reset
        key_step, key_reset = jax.random.split(key)
        
        # Perform environment step
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        
        # Prepare reset state (in case we need it)
        obs_re, state_re = self.reset_env(key_reset, params)
        
        # Auto-reset: select reset state if done, otherwise keep stepped state
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        
        return obs, state, reward, done, info
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: jax.Array,
        params: TEnvParams | None = None,
    ) -> Tuple[jax.Array, TEnvState]:
        """Reset the environment.
        
        Args:
            key: JAX random key
            params: Environment parameters (uses default if None)
            
        Returns:
            obs: Initial observations for all agents, shape (n_agents, obs_dim)
            state: Initial environment state
        """
        if params is None:
            params = self.default_params
        
        return self.reset_env(key, params)
    
    @abstractmethod
    def step_env(
        self,
        key: jax.Array,
        state: TEnvState,
        action: jax.Array,
        params: TEnvParams,
    ) -> Tuple[jax.Array, TEnvState, jax.Array, jax.Array, Dict[str, Any]]:
        """Environment-specific step logic.
        
        This method should implement the actual environment dynamics.
        
        Args:
            key: JAX random key
            state: Current environment state
            action: Actions for all agents, shape (n_agents, action_dim)
            params: Environment parameters
            
        Returns:
            obs: Observations for all agents, shape (n_agents, obs_dim)
            state: New environment state
            reward: Rewards for all agents, shape (n_agents,)
            done: Whether episode is done (scalar bool)
            info: Additional information dictionary
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset_env(
        self,
        key: jax.Array,
        params: TEnvParams,
    ) -> Tuple[jax.Array, TEnvState]:
        """Environment-specific reset logic.
        
        This method should initialize the environment state.
        
        Args:
            key: JAX random key
            params: Environment parameters
            
        Returns:
            obs: Initial observations for all agents, shape (n_agents, obs_dim)
            state: Initial environment state
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_obs(
        self,
        state: TEnvState,
        params: TEnvParams,
    ) -> jax.Array:
        """Compute observations from state.
        
        Args:
            state: Current environment state
            params: Environment parameters
            
        Returns:
            obs: Observations for all agents, shape (n_agents, obs_dim)
        """
        raise NotImplementedError
    
    @abstractmethod
    def is_terminal(
        self,
        state: TEnvState,
        params: TEnvParams,
    ) -> jax.Array:
        """Check if the episode should terminate.
        
        Args:
            state: Current environment state
            params: Environment parameters
            
        Returns:
            done: Boolean scalar indicating if episode is done
        """
        raise NotImplementedError
    
    def discount(
        self,
        state: TEnvState,
        params: TEnvParams,
    ) -> jax.Array:
        """Return discount factor (0 if terminal, 1 otherwise).
        
        Useful for computing returns in RL algorithms.
        
        Args:
            state: Current environment state
            params: Environment parameters
            
        Returns:
            discount: 0.0 if terminal, 1.0 otherwise
        """
        return jax.lax.select(
            self.is_terminal(state, params),
            jnp.array(0.0),
            jnp.array(1.0)
        )
    
    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__
    
    @abstractmethod
    def action_space(self, params: TEnvParams) -> MultiAgentActionSpace:
        """Return the action space for all agents.
        
        Args:
            params: Environment parameters
            
        Returns:
            MultiAgentActionSpace with shape (n_agents, action_dim)
        """
        raise NotImplementedError
    
    @abstractmethod
    def observation_space(self, params: TEnvParams) -> MultiAgentObservationSpace:
        """Return the observation space for all agents.
        
        Args:
            params: Environment parameters
            
        Returns:
            MultiAgentObservationSpace with shape (n_agents, obs_dim)
        """
        raise NotImplementedError


# ============================================================================
# Simple example environment for testing the base class
# ============================================================================

@struct.dataclass
class SimpleSwarmState(EnvState):
    """State for a simple test swarm environment."""
    positions: jax.Array  # (n_agents, 2)
    velocities: jax.Array  # (n_agents, 2)
    time: int


@struct.dataclass
class SimpleSwarmParams(EnvParams):
    """Parameters for a simple test swarm environment.
    
    Note: n_agents is NOT here because it affects array shapes.
    Array shapes must be static (known at JIT compile time) in JAX.
    n_agents is instead a property of the environment class.
    """
    max_steps_in_episode: int = 100
    arena_size: float = 2.0
    max_velocity: float = 1.0
    dt: float = 0.1


class SimpleSwarmEnv(MultiAgentEnv[SimpleSwarmState, SimpleSwarmParams]):
    """A simple multi-agent environment for testing.
    
    Agents move in a 2D arena. Actions are 2D velocity commands.
    Reward is -distance to origin for each agent.
    Episode ends when max steps reached.
    
    Note: n_agents is fixed at construction time because JAX requires
    static array shapes for JIT compilation.
    """
    
    def __init__(self, n_agents: int = 4):
        super().__init__()
        self._n_agents = n_agents
    
    @property
    def n_agents(self) -> int:
        return self._n_agents
    
    @property
    def default_params(self) -> SimpleSwarmParams:
        return SimpleSwarmParams()
    
    def step_env(
        self,
        key: jax.Array,
        state: SimpleSwarmState,
        action: jax.Array,
        params: SimpleSwarmParams,
    ) -> Tuple[jax.Array, SimpleSwarmState, jax.Array, jax.Array, Dict[str, Any]]:
        """Simple dynamics: velocity control with clipping."""
        
        # Clip actions to valid range
        action = jnp.clip(action, -1.0, 1.0)
        
        # Update velocities (actions are velocity commands)
        velocities = action * params.max_velocity
        
        # Update positions
        new_positions = state.positions + velocities * params.dt
        
        # Clip to arena bounds
        new_positions = jnp.clip(
            new_positions, 
            -params.arena_size, 
            params.arena_size
        )
        
        # Create new state
        new_state = SimpleSwarmState(
            positions=new_positions,
            velocities=velocities,
            time=state.time + 1,
        )
        
        # Compute observations
        obs = self.get_obs(new_state, params)
        
        # Compute rewards (negative distance to origin)
        distances = jnp.linalg.norm(new_positions, axis=1)
        rewards = -distances
        
        # Check if done
        done = self.is_terminal(new_state, params)
        
        info = {"distances": distances}
        
        return obs, new_state, rewards, done, info
    
    def reset_env(
        self,
        key: jax.Array,
        params: SimpleSwarmParams,
    ) -> Tuple[jax.Array, SimpleSwarmState]:
        """Reset agents to random positions."""
        
        key_pos, key_vel = jax.random.split(key)
        
        # Random initial positions
        # Use self._n_agents (static) instead of params.n_agents (traced)
        positions = jax.random.uniform(
            key_pos,
            shape=(self._n_agents, 2),
            minval=-params.arena_size,
            maxval=params.arena_size,
        )
        
        # Zero initial velocities
        velocities = jnp.zeros((self._n_agents, 2))
        
        state = SimpleSwarmState(
            positions=positions,
            velocities=velocities,
            time=0,
        )
        
        obs = self.get_obs(state, params)
        
        return obs, state
    
    def get_obs(
        self,
        state: SimpleSwarmState,
        params: SimpleSwarmParams,
    ) -> jax.Array:
        """Observation is [position, velocity] for each agent."""
        return jnp.concatenate([state.positions, state.velocities], axis=1)
    
    def is_terminal(
        self,
        state: SimpleSwarmState,
        params: SimpleSwarmParams,
    ) -> jax.Array:
        """Episode ends when max steps reached."""
        return state.time >= params.max_steps_in_episode
    
    def action_space(self, params: SimpleSwarmParams) -> MultiAgentActionSpace:
        """2D velocity control for each agent."""
        return MultiAgentActionSpace(
            n_agents=self._n_agents,
            action_dim=2,
            low=-1.0,
            high=1.0,
        )
    
    def observation_space(self, params: SimpleSwarmParams) -> MultiAgentObservationSpace:
        """Observation is [x, y, vx, vy] for each agent."""
        return MultiAgentObservationSpace(
            n_agents=self._n_agents,
            obs_dim=4,  # position (2) + velocity (2)
            low=-jnp.inf,
            high=jnp.inf,
        )
