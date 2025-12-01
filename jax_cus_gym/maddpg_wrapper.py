"""MADDPG-compatible wrapper for the Assembly Swarm Environment.

This module provides a wrapper that makes the environment compatible with
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) training.

Key features:
- Separates per-agent observations and actions
- Provides centralized critic interface (global state)
- Handles experience collection for replay buffers
- Compatible with common MARL libraries

Usage:
    from maddpg_wrapper import MADDPGWrapper
    
    env = MADDPGWrapper(n_agents=10)
    obs = env.reset(key)
    
    # Training loop
    actions = [agent.act(obs[i]) for i, agent in enumerate(agents)]
    next_obs, rewards, dones, info = env.step(key, actions)
"""

from typing import Tuple, Dict, Any, List, NamedTuple, Optional
import jax
import jax.numpy as jnp
from jax import random
import flax.struct as struct

from assembly_env import (
    AssemblySwarmEnv,
    AssemblyParams,
    AssemblyState,
    make_assembly_env,
)


class Transition(NamedTuple):
    """A single transition for replay buffer.
    
    Attributes:
        obs: Observations for all agents (n_agents, obs_dim)
        actions: Actions taken by all agents (n_agents, action_dim)
        rewards: Rewards received (n_agents,)
        next_obs: Next observations (n_agents, obs_dim)
        dones: Done flags (n_agents,)
        global_state: Optional global state for centralized critic
    """
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray
    global_state: Optional[jnp.ndarray] = None
    next_global_state: Optional[jnp.ndarray] = None


@struct.dataclass
class MADDPGState:
    """State wrapper for MADDPG training.
    
    Extends AssemblyState with additional training info.
    """
    env_state: AssemblyState
    episode_returns: jnp.ndarray  # (n_agents,)
    episode_length: int


class MADDPGWrapper:
    """MADDPG-compatible wrapper for AssemblySwarmEnv.
    
    This wrapper provides:
    1. Per-agent observation/action interface
    2. Global state for centralized critic (CTDE)
    3. Experience collection utilities
    4. Vectorized environment support
    
    Example:
        ```python
        wrapper = MADDPGWrapper(n_agents=10, arena_size=5.0)
        
        key = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(key)
        
        # obs is (n_agents, obs_dim) - each agent gets its observation
        # For MADDPG, you typically have separate actor networks per agent
        
        actions = jnp.zeros((10, 2))  # Each agent outputs 2D action
        next_obs, state, rewards, dones, info = wrapper.step(key, state, actions)
        
        # Get global state for centralized critic
        global_state = wrapper.get_global_state(state)
        ```
    """
    
    def __init__(
        self,
        n_agents: int = 10,
        **env_kwargs,
    ):
        """Initialize the MADDPG wrapper.
        
        Args:
            n_agents: Number of agents
            **env_kwargs: Additional arguments passed to AssemblyParams
        """
        self.env, self.params = make_assembly_env(n_agents=n_agents, **env_kwargs)
        self._n_agents = n_agents
    
    @property
    def n_agents(self) -> int:
        return self._n_agents
    
    @property
    def obs_dim(self) -> int:
        """Observation dimension per agent."""
        return self.env.get_obs_dim(self.params)
    
    @property
    def action_dim(self) -> int:
        """Action dimension per agent."""
        return self.env.get_action_dim(self.params)
    
    @property
    def global_state_dim(self) -> int:
        """Dimension of global state for centralized critic.
        
        Global state includes:
        - All agent positions and velocities: n_agents * 4
        - Grid center positions: max_n_grid * 2
        - Time: 1
        """
        max_n_grid = self.env.shape_library.max_n_grid
        return self.n_agents * 4 + max_n_grid * 2 + 1
    
    def reset(
        self,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, MADDPGState]:
        """Reset the environment.
        
        Args:
            key: JAX random key
            
        Returns:
            obs: Initial observations (n_agents, obs_dim)
            state: MADDPG state wrapper
        """
        obs, env_state = self.env.reset(key, self.params)
        
        state = MADDPGState(
            env_state=env_state,
            episode_returns=jnp.zeros(self.n_agents),
            episode_length=0,
        )
        
        return obs, state
    
    def step(
        self,
        key: jnp.ndarray,
        state: MADDPGState,
        actions: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, MADDPGState, jnp.ndarray, jnp.ndarray, Dict]:
        """Take a step in the environment.
        
        Args:
            key: JAX random key
            state: Current MADDPG state
            actions: Actions for all agents (n_agents, action_dim)
            
        Returns:
            obs: New observations (n_agents, obs_dim)
            new_state: Updated MADDPG state
            rewards: Rewards for all agents (n_agents,)
            dones: Done flags (n_agents,)
            info: Additional information
        """
        obs, env_state, rewards, dones, info = self.env.step(
            key, state.env_state, actions, self.params
        )
        
        new_state = MADDPGState(
            env_state=env_state,
            episode_returns=state.episode_returns + rewards,
            episode_length=state.episode_length + 1,
        )
        
        # Add episode info
        info["episode_returns"] = new_state.episode_returns
        info["episode_length"] = new_state.episode_length
        
        return obs, new_state, rewards, dones, info
    
    def get_global_state(self, state: MADDPGState) -> jnp.ndarray:
        """Get global state for centralized critic.
        
        The global state contains information about all agents and the
        environment, useful for the centralized critic in CTDE.
        
        Args:
            state: Current MADDPG state
            
        Returns:
            global_state: (global_state_dim,) array
        """
        env_state = state.env_state
        
        # Flatten all agent positions and velocities
        agent_states = jnp.concatenate([
            env_state.positions.flatten(),  # (n_agents * 2,)
            env_state.velocities.flatten(),  # (n_agents * 2,)
        ])
        
        # Flatten grid centers
        grid_states = env_state.grid_centers.flatten()  # (n_grid * 2,)
        
        # Normalized time
        time_state = jnp.array([env_state.time / (self.params.max_steps * self.params.dt)])
        
        return jnp.concatenate([agent_states, grid_states, time_state])
    
    def get_agent_obs(
        self, 
        obs: jnp.ndarray, 
        agent_idx: int
    ) -> jnp.ndarray:
        """Get observation for a specific agent.
        
        Args:
            obs: Full observations (n_agents, obs_dim)
            agent_idx: Index of agent
            
        Returns:
            Agent's observation (obs_dim,)
        """
        return obs[agent_idx]
    
    def collect_transition(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_obs: jnp.ndarray,
        dones: jnp.ndarray,
        state: Optional[MADDPGState] = None,
        next_state: Optional[MADDPGState] = None,
    ) -> Transition:
        """Collect a transition for the replay buffer.
        
        Args:
            obs: Current observations
            actions: Actions taken
            rewards: Rewards received
            next_obs: Next observations
            dones: Done flags
            state: Current state (for global state)
            next_state: Next state (for global state)
            
        Returns:
            Transition tuple
        """
        global_state = None
        next_global_state = None
        
        if state is not None:
            global_state = self.get_global_state(state)
        if next_state is not None:
            next_global_state = self.get_global_state(next_state)
        
        return Transition(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            global_state=global_state,
            next_global_state=next_global_state,
        )


class VectorizedMADDPGWrapper:
    """Vectorized MADDPG wrapper for parallel environment execution.
    
    Uses JAX vmap for efficient parallel environment execution.
    
    Example:
        ```python
        vec_wrapper = VectorizedMADDPGWrapper(n_envs=8, n_agents=10)
        
        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        obs, states = vec_wrapper.reset(keys)
        
        # obs is (n_envs, n_agents, obs_dim)
        actions = jnp.zeros((8, 10, 2))
        next_obs, states, rewards, dones, info = vec_wrapper.step(keys, states, actions)
        ```
    """
    
    def __init__(
        self,
        n_envs: int,
        n_agents: int = 10,
        **env_kwargs,
    ):
        """Initialize vectorized wrapper.
        
        Args:
            n_envs: Number of parallel environments
            n_agents: Number of agents per environment
            **env_kwargs: Additional arguments passed to AssemblyParams
        """
        self.n_envs = n_envs
        self.wrapper = MADDPGWrapper(n_agents=n_agents, **env_kwargs)
        
        # Create vectorized functions
        self._vec_reset = jax.jit(jax.vmap(self.wrapper.reset))
        self._vec_step = jax.jit(jax.vmap(self.wrapper.step))
        self._vec_global_state = jax.vmap(self.wrapper.get_global_state)
    
    @property
    def n_agents(self) -> int:
        return self.wrapper.n_agents
    
    @property
    def obs_dim(self) -> int:
        return self.wrapper.obs_dim
    
    @property
    def action_dim(self) -> int:
        return self.wrapper.action_dim
    
    @property
    def global_state_dim(self) -> int:
        return self.wrapper.global_state_dim
    
    def reset(
        self,
        keys: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, MADDPGState]:
        """Reset all environments.
        
        Args:
            keys: Random keys (n_envs,)
            
        Returns:
            obs: (n_envs, n_agents, obs_dim)
            states: Batched MADDPGState
        """
        return self._vec_reset(keys)
    
    def step(
        self,
        keys: jnp.ndarray,
        states: MADDPGState,
        actions: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, MADDPGState, jnp.ndarray, jnp.ndarray, Dict]:
        """Step all environments.
        
        Args:
            keys: Random keys (n_envs,)
            states: Batched states
            actions: (n_envs, n_agents, action_dim)
            
        Returns:
            obs: (n_envs, n_agents, obs_dim)
            states: Updated batched states
            rewards: (n_envs, n_agents)
            dones: (n_envs, n_agents)
            info: Batched info dict
        """
        return self._vec_step(keys, states, actions)
    
    def get_global_states(self, states: MADDPGState) -> jnp.ndarray:
        """Get global states for all environments.
        
        Args:
            states: Batched states
            
        Returns:
            global_states: (n_envs, global_state_dim)
        """
        return self._vec_global_state(states)


# Convenience functions for MADDPG training

def create_maddpg_env(
    n_agents: int = 10,
    **kwargs,
) -> MADDPGWrapper:
    """Create a MADDPG-compatible environment.
    
    Args:
        n_agents: Number of agents
        **kwargs: Additional environment parameters
        
    Returns:
        MADDPGWrapper instance
    """
    return MADDPGWrapper(n_agents=n_agents, **kwargs)


def create_vec_maddpg_env(
    n_envs: int,
    n_agents: int = 10,
    **kwargs,
) -> VectorizedMADDPGWrapper:
    """Create a vectorized MADDPG-compatible environment.
    
    Args:
        n_envs: Number of parallel environments
        n_agents: Number of agents per environment
        **kwargs: Additional environment parameters
        
    Returns:
        VectorizedMADDPGWrapper instance
    """
    return VectorizedMADDPGWrapper(n_envs=n_envs, n_agents=n_agents, **kwargs)


def rollout_episode(
    wrapper: MADDPGWrapper,
    key: jnp.ndarray,
    policy_fn,
    max_steps: Optional[int] = None,
) -> Tuple[List[Transition], Dict]:
    """Rollout a full episode and collect transitions.
    
    Args:
        wrapper: MADDPG wrapper
        key: Random key
        policy_fn: Function that takes (key, obs) and returns actions
        max_steps: Maximum steps (default: from params)
        
    Returns:
        transitions: List of Transition tuples
        info: Episode info (returns, length, etc.)
    """
    if max_steps is None:
        max_steps = wrapper.params.max_steps
    
    key, reset_key = random.split(key)
    obs, state = wrapper.reset(reset_key)
    
    transitions = []
    
    for _ in range(max_steps):
        key, action_key, step_key = random.split(key, 3)
        
        # Get actions from policy
        actions = policy_fn(action_key, obs)
        
        # Step environment
        next_obs, next_state, rewards, dones, info = wrapper.step(
            step_key, state, actions
        )
        
        # Collect transition
        transition = wrapper.collect_transition(
            obs, actions, rewards, next_obs, dones,
            state, next_state
        )
        transitions.append(transition)
        
        # Check if done
        if next_state.env_state.done:
            break
        
        obs = next_obs
        state = next_state
    
    episode_info = {
        "episode_return": float(jnp.sum(state.episode_returns)),
        "episode_length": state.episode_length,
        "mean_agent_return": float(jnp.mean(state.episode_returns)),
        "final_coverage": float(info.get("coverage_rate", 0.0)),
    }
    
    return transitions, episode_info


def stack_transitions(transitions: List[Transition]) -> Transition:
    """Stack a list of transitions into batched arrays.
    
    Useful for adding to replay buffer.
    
    Args:
        transitions: List of Transition tuples
        
    Returns:
        Batched Transition with shape (T, ...)
    """
    return Transition(
        obs=jnp.stack([t.obs for t in transitions]),
        actions=jnp.stack([t.actions for t in transitions]),
        rewards=jnp.stack([t.rewards for t in transitions]),
        next_obs=jnp.stack([t.next_obs for t in transitions]),
        dones=jnp.stack([t.dones for t in transitions]),
        global_state=jnp.stack([t.global_state for t in transitions]) 
            if transitions[0].global_state is not None else None,
        next_global_state=jnp.stack([t.next_global_state for t in transitions])
            if transitions[0].next_global_state is not None else None,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing MADDPGWrapper...")
    
    wrapper = MADDPGWrapper(n_agents=5, arena_size=4.0)
    print(f"n_agents: {wrapper.n_agents}")
    print(f"obs_dim: {wrapper.obs_dim}")
    print(f"action_dim: {wrapper.action_dim}")
    print(f"global_state_dim: {wrapper.global_state_dim}")
    
    key = random.PRNGKey(0)
    obs, state = wrapper.reset(key)
    print(f"Obs shape: {obs.shape}")
    
    # Take a step
    key, step_key = random.split(key)
    actions = random.uniform(key, (5, 2), minval=-1, maxval=1)
    next_obs, next_state, rewards, dones, info = wrapper.step(step_key, state, actions)
    
    print(f"Rewards: {rewards}")
    print(f"Global state shape: {wrapper.get_global_state(state).shape}")
    
    # Test transition collection
    transition = wrapper.collect_transition(
        obs, actions, rewards, next_obs, dones, state, next_state
    )
    print(f"Transition obs shape: {transition.obs.shape}")
    print(f"Transition global_state shape: {transition.global_state.shape}")
    
    print("\nTesting VectorizedMADDPGWrapper...")
    vec_wrapper = VectorizedMADDPGWrapper(n_envs=4, n_agents=5)
    
    keys = random.split(random.PRNGKey(1), 4)
    obs, states = vec_wrapper.reset(keys)
    print(f"Vectorized obs shape: {obs.shape}")
    
    actions = random.uniform(random.PRNGKey(2), (4, 5, 2), minval=-1, maxval=1)
    step_keys = random.split(random.PRNGKey(3), 4)
    next_obs, next_states, rewards, dones, info = vec_wrapper.step(step_keys, states, actions)
    print(f"Vectorized rewards shape: {rewards.shape}")
    
    global_states = vec_wrapper.get_global_states(states)
    print(f"Global states shape: {global_states.shape}")
    
    print("\nMADDPGWrapper tests passed!")
