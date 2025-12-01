"""MADDPG (Multi-Agent Deep Deterministic Policy Gradient) Algorithm.

This module provides the main MADDPG coordinator class that manages multiple
DDPG agents in a multi-agent environment.

Key features:
- Centralized training with decentralized execution (CTDE)
- Each agent has its own actor (policy) but critics use global information
- Support for action priors and regularization
- Noise scheduling for exploration
- JIT-compatible training loop

Reference:
    Lowe et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
    https://arxiv.org/abs/1706.02275
"""

from typing import Tuple, NamedTuple, Optional, Dict, Any, List, Callable, Union
import jax
import jax.numpy as jnp
from jax import random
from flax import struct
import optax
from functools import partial

try:
    from .agents import (
        DDPGAgentState,
        AgentConfig,
        create_agent,
        create_agent_with_networks,
        select_action,
        select_action_with_noise,
        select_target_action,
        update_critic,
        update_actor,
        update_targets,
        reset_noise,
        get_noise_scale,
        get_agent_params,
        load_agent_params,
    )
    from .networks import Actor, ActorDiscrete, Critic, CriticTwin
    from .buffers import ReplayBuffer, ReplayBufferState, PerAgentReplayBuffer, Transition
    from .noise import OUNoiseState, ou_noise_init
    from .utils import soft_update
except ImportError:
    from agents import (
        DDPGAgentState,
        AgentConfig,
        create_agent,
        create_agent_with_networks,
        select_action,
        select_action_with_noise,
        select_target_action,
        update_critic,
        update_actor,
        update_targets,
        reset_noise,
        get_noise_scale,
        get_agent_params,
        load_agent_params,
    )
    from networks import Actor, ActorDiscrete, Critic, CriticTwin
    from buffers import ReplayBuffer, ReplayBufferState, PerAgentReplayBuffer, Transition
    from noise import OUNoiseState, ou_noise_init
    from utils import soft_update


# ============================================================================
# MADDPG State
# ============================================================================

@struct.dataclass
class MADDPGState:
    """Complete state for MADDPG training.
    
    Attributes:
        agent_states: List of DDPGAgentState for each agent
        buffer_state: Replay buffer state
        step: Global training step counter
        episode: Episode counter
        noise_scale: Current exploration noise scale
    """
    agent_states: List[DDPGAgentState]
    buffer_state: ReplayBufferState
    step: jnp.ndarray
    episode: jnp.ndarray
    noise_scale: jnp.ndarray


class MADDPGConfig(NamedTuple):
    """Configuration for MADDPG algorithm.
    
    Attributes:
        n_agents: Number of agents
        obs_dims: Observation dimension for each agent
        action_dims: Action dimension for each agent
        global_state_dim: Dimension of global state (optional)
        hidden_dims: Hidden layer dimensions for networks
        lr_actor: Learning rate for actors
        lr_critic: Learning rate for critics
        gamma: Discount factor
        tau: Soft update coefficient
        buffer_size: Replay buffer capacity
        batch_size: Batch size for updates
        warmup_steps: Steps before training starts
        noise_scale_initial: Initial exploration noise
        noise_scale_final: Final exploration noise
        noise_decay_steps: Steps to decay noise over
        noise_schedule: 'linear', 'exponential', or 'cosine'
        update_every: Steps between updates
        updates_per_step: Number of gradient updates per step
        discrete_action: Whether actions are discrete
        use_layer_norm: Whether to use layer normalization
        max_grad_norm: Maximum gradient norm (None = no clipping)
        prior_weight: Weight for action prior regularization
        shared_critic: Whether to share critic across agents
    """
    n_agents: int
    obs_dims: Tuple[int, ...]
    action_dims: Tuple[int, ...]
    global_state_dim: Optional[int] = None
    hidden_dims: Tuple[int, ...] = (64, 64)
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    gamma: float = 0.95
    tau: float = 0.01
    buffer_size: int = 1000000
    batch_size: int = 1024
    warmup_steps: int = 10000
    noise_scale_initial: float = 0.3
    noise_scale_final: float = 0.05
    noise_decay_steps: int = 100000
    noise_schedule: str = 'linear'
    update_every: int = 100
    updates_per_step: int = 1
    discrete_action: bool = False
    use_layer_norm: bool = False
    max_grad_norm: Optional[float] = None
    prior_weight: float = 0.0
    shared_critic: bool = False


# ============================================================================
# MADDPG Class
# ============================================================================

class MADDPG:
    """Multi-Agent Deep Deterministic Policy Gradient.
    
    This class coordinates multiple DDPG agents for multi-agent RL.
    Each agent has its own actor network but critics observe the full state.
    
    Example:
        ```python
        config = MADDPGConfig(
            n_agents=3,
            obs_dims=(10, 10, 10),
            action_dims=(2, 2, 2),
        )
        
        maddpg = MADDPG(config)
        state = maddpg.init(key)
        
        # During rollout
        actions, log_probs, state = maddpg.select_actions(key, state, observations)
        
        # Store transition
        state = maddpg.store_transition(state, obs, actions, rewards, next_obs, dones)
        
        # Update
        state, info = maddpg.update(key, state)
        ```
    """
    
    def __init__(self, config: MADDPGConfig):
        """Initialize MADDPG.
        
        Args:
            config: MADDPG configuration
        """
        self.config = config
        self.n_agents = config.n_agents
        
        # Compute dimensions
        self.obs_dims = config.obs_dims
        self.action_dims = config.action_dims
        self.total_obs_dim = sum(config.obs_dims)
        self.total_action_dim = sum(config.action_dims)
        
        # Global state dimension for critic
        if config.global_state_dim is not None:
            self.global_state_dim = config.global_state_dim
        else:
            self.global_state_dim = self.total_obs_dim
        
        # Critic input: global_state + all_actions
        self.critic_input_dim = self.global_state_dim + self.total_action_dim
        
        # Create networks (shared across all agents if shared_critic)
        self.actors = []
        self.critics = []
        
        for i in range(self.n_agents):
            if config.discrete_action:
                actor = ActorDiscrete(
                    n_actions=config.action_dims[i],
                    hidden_dims=config.hidden_dims,
                    use_layer_norm=config.use_layer_norm,
                )
            else:
                actor = Actor(
                    action_dim=config.action_dims[i],
                    hidden_dims=config.hidden_dims,
                    use_layer_norm=config.use_layer_norm,
                )
            self.actors.append(actor)
            
            if not config.shared_critic or i == 0:
                critic = Critic(
                    hidden_dims=config.hidden_dims,
                    use_layer_norm=config.use_layer_norm,
                )
                self.critics.append(critic)
        
        if config.shared_critic:
            # All agents share the same critic
            self.critics = [self.critics[0]] * self.n_agents
        
        # Create optimizers
        if config.max_grad_norm is not None:
            self.actor_optimizer = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.lr_actor),
            )
            self.critic_optimizer = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.lr_critic),
            )
        else:
            self.actor_optimizer = optax.adam(config.lr_actor)
            self.critic_optimizer = optax.adam(config.lr_critic)
        
        # Create replay buffer
        # Note: For MADDPG, we store flattened obs/actions since agents may have different dims
        # The buffer is configured for the total dimensions
        self.buffer = ReplayBuffer(
            capacity=config.buffer_size,
            n_agents=config.n_agents,
            obs_dim=max(config.obs_dims),  # Use max dim, we'll handle padding
            action_dim=max(config.action_dims),  # Use max dim
            global_state_dim=self.global_state_dim if config.global_state_dim else None,
            use_global_state=config.global_state_dim is not None,
            store_log_probs=True,
            store_action_priors=config.prior_weight > 0,
        )
    
    def init(self, key: jax.Array) -> MADDPGState:
        """Initialize MADDPG state.
        
        Args:
            key: JAX random key
            
        Returns:
            Initial MADDPGState
        """
        keys = random.split(key, self.n_agents + 1)
        buffer_key = keys[0]
        agent_keys = keys[1:]
        
        # Initialize agent states
        agent_states = []
        for i, (k, actor, critic) in enumerate(zip(agent_keys, self.actors, self.critics)):
            agent_config = AgentConfig(
                obs_dim=self.obs_dims[i],
                action_dim=self.action_dims[i],
                critic_input_dim=self.critic_input_dim,
                hidden_dims=self.config.hidden_dims,
                lr_actor=self.config.lr_actor,
                lr_critic=self.config.lr_critic,
                gamma=self.config.gamma,
                tau=self.config.tau,
                discrete_action=self.config.discrete_action,
                noise_type='gaussian',
                noise_scale=self.config.noise_scale_initial,
                use_layer_norm=self.config.use_layer_norm,
                max_grad_norm=self.config.max_grad_norm,
            )
            
            state = create_agent_with_networks(k, agent_config, actor, critic)
            agent_states.append(state)
        
        # Initialize buffer
        buffer_state = self.buffer.init()
        
        return MADDPGState(
            agent_states=agent_states,
            buffer_state=buffer_state,
            step=jnp.array(0, dtype=jnp.int32),
            episode=jnp.array(0, dtype=jnp.int32),
            noise_scale=jnp.array(self.config.noise_scale_initial, dtype=jnp.float32),
        )
    
    def select_actions(
        self,
        key: jax.Array,
        state: MADDPGState,
        observations: List[jax.Array],
        explore: bool = True,
    ) -> Tuple[List[jax.Array], List[jax.Array], MADDPGState]:
        """Select actions for all agents.
        
        Args:
            key: JAX random key
            state: Current MADDPG state
            observations: List of observations for each agent
            explore: Whether to add exploration noise
            
        Returns:
            actions: List of actions for each agent
            log_probs: List of log probabilities
            new_state: Updated state (noise state may change)
        """
        keys = random.split(key, self.n_agents)
        
        actions = []
        log_probs = []
        new_agent_states = []
        
        for i, (k, actor, agent_state, obs) in enumerate(
            zip(keys, self.actors, state.agent_states, observations)
        ):
            if explore:
                action, log_prob, new_noise_state = select_action_with_noise(
                    key=k,
                    actor=actor,
                    actor_params=agent_state.actor_params,
                    obs=obs,
                    noise_scale=float(state.noise_scale),
                    noise_type='gaussian',
                    noise_state=agent_state.noise_state,
                    epsilon=0.0,
                    discrete_action=self.config.discrete_action,
                )
                
                if new_noise_state is not None:
                    new_agent_state = agent_state.replace(noise_state=new_noise_state)
                else:
                    new_agent_state = agent_state
            else:
                action = select_action(actor, agent_state.actor_params, obs)
                log_prob = jnp.zeros((1,))
                new_agent_state = agent_state
            
            actions.append(action)
            log_probs.append(log_prob)
            new_agent_states.append(new_agent_state)
        
        new_state = state.replace(agent_states=new_agent_states)
        
        return actions, log_probs, new_state
    
    def select_target_actions(
        self,
        state: MADDPGState,
        next_observations: jax.Array,
    ) -> jax.Array:
        """Select target actions for all agents (for computing targets).
        
        Args:
            state: Current MADDPG state
            next_observations: Next observations, shape (batch, total_obs_dim)
            
        Returns:
            target_actions: All target actions concatenated, shape (batch, total_action_dim)
        """
        batch_size = next_observations.shape[0]
        target_actions = []
        
        obs_start = 0
        for i, (actor, agent_state) in enumerate(zip(self.actors, state.agent_states)):
            obs_end = obs_start + self.obs_dims[i]
            agent_obs = next_observations[:, obs_start:obs_end]
            
            target_action = select_target_action(
                actor, agent_state.target_actor_params, agent_obs
            )
            target_actions.append(target_action)
            
            obs_start = obs_end
        
        return jnp.concatenate(target_actions, axis=-1)
    
    def store_transition(
        self,
        state: MADDPGState,
        observations: List[jax.Array],
        actions: List[jax.Array],
        rewards: List[jax.Array],
        next_observations: List[jax.Array],
        dones: List[jax.Array],
        global_state: Optional[jax.Array] = None,
        next_global_state: Optional[jax.Array] = None,
        log_probs: Optional[List[jax.Array]] = None,
        action_priors: Optional[List[jax.Array]] = None,
    ) -> MADDPGState:
        """Store a transition in the replay buffer.
        
        Args:
            state: Current MADDPG state
            observations: List of observations for each agent
            actions: List of actions for each agent
            rewards: List of rewards for each agent
            next_observations: List of next observations
            dones: List of done flags
            global_state: Optional global state
            next_global_state: Optional next global state
            log_probs: Optional log probabilities
            action_priors: Optional prior actions
            
        Returns:
            Updated MADDPGState
        """
        # Stack observations and actions per agent
        # Need to pad to max dimensions if heterogeneous
        max_obs_dim = self.buffer.obs_dim
        max_action_dim = self.buffer.action_dim
        
        # Pad observations to max dim
        obs_padded = []
        for i, obs in enumerate(observations):
            obs_flat = obs.flatten()
            if len(obs_flat) < max_obs_dim:
                obs_flat = jnp.pad(obs_flat, (0, max_obs_dim - len(obs_flat)))
            obs_padded.append(obs_flat)
        obs_stacked = jnp.stack(obs_padded)  # (n_agents, max_obs_dim)
        
        # Pad actions to max dim
        actions_padded = []
        for i, action in enumerate(actions):
            action_flat = action.flatten()
            if len(action_flat) < max_action_dim:
                action_flat = jnp.pad(action_flat, (0, max_action_dim - len(action_flat)))
            actions_padded.append(action_flat)
        actions_stacked = jnp.stack(actions_padded)  # (n_agents, max_action_dim)
        
        # Pad next observations
        next_obs_padded = []
        for i, obs in enumerate(next_observations):
            obs_flat = obs.flatten()
            if len(obs_flat) < max_obs_dim:
                obs_flat = jnp.pad(obs_flat, (0, max_obs_dim - len(obs_flat)))
            next_obs_padded.append(obs_flat)
        next_obs_stacked = jnp.stack(next_obs_padded)  # (n_agents, max_obs_dim)
        
        # Stack rewards and dones per agent
        rewards_stacked = jnp.stack([
            r.flatten()[0] if r.ndim > 0 else r for r in rewards
        ])  # (n_agents,)
        dones_stacked = jnp.stack([
            d.flatten()[0] if d.ndim > 0 else d for d in dones
        ])  # (n_agents,)
        
        # Handle optional log_probs
        if log_probs is not None:
            log_probs_stacked = jnp.stack([
                lp.flatten()[0] if lp.ndim > 0 else lp for lp in log_probs
            ])  # (n_agents,)
        else:
            log_probs_stacked = None
        
        # Handle optional action_priors
        if action_priors is not None:
            priors_padded = []
            for i, prior in enumerate(action_priors):
                prior_flat = prior.flatten()
                if len(prior_flat) < max_action_dim:
                    prior_flat = jnp.pad(prior_flat, (0, max_action_dim - len(prior_flat)))
                priors_padded.append(prior_flat)
            action_priors_stacked = jnp.stack(priors_padded)  # (n_agents, max_action_dim)
        else:
            action_priors_stacked = None
        
        # Create transition
        transition = Transition(
            obs=obs_stacked,
            actions=actions_stacked,
            rewards=rewards_stacked,
            next_obs=next_obs_stacked,
            dones=dones_stacked,
            global_state=global_state,
            next_global_state=next_global_state,
            log_probs=log_probs_stacked,
            action_priors=action_priors_stacked,
        )
        
        # Add to buffer
        new_buffer_state = self.buffer.add(state.buffer_state, transition)
        
        return state.replace(buffer_state=new_buffer_state)
    
    def select_actions_batched(
        self,
        key: jax.Array,
        state: MADDPGState,
        obs_batch: jax.Array,
        explore: bool = True,
    ) -> Tuple[jax.Array, MADDPGState]:
        """Select actions for all agents across multiple parallel environments.
        
        This is a vectorized version that avoids Python loops for efficiency.
        
        Args:
            key: JAX random key
            state: Current MADDPG state
            obs_batch: Observations, shape (n_envs, n_agents, obs_dim)
            explore: Whether to add exploration noise
            
        Returns:
            actions: Actions, shape (n_envs, n_agents, action_dim)
            new_state: Updated state
        """
        n_envs = obs_batch.shape[0]
        n_agents = self.n_agents
        
        # Generate keys for all envs and agents
        keys = random.split(key, n_envs * n_agents).reshape(n_envs, n_agents, 2)
        
        # Process each agent (still loop over agents since they have different networks,
        # but vectorize over environments)
        all_actions = []
        new_agent_states = []
        
        # Extract noise_scale once (outside of traced functions to avoid float() issues)
        noise_scale_value = state.noise_scale
        
        for i, (actor, agent_state) in enumerate(zip(self.actors, state.agent_states)):
            # obs for this agent across all envs: (n_envs, obs_dim)
            agent_obs = obs_batch[:, i, :]
            agent_keys = keys[:, i, :]  # (n_envs, 2) but we just need n_envs keys
            
            if explore:
                # Vectorized action selection with noise across all envs
                def select_single(key_obs):
                    k, obs = key_obs
                    action, _, _ = select_action_with_noise(
                        key=k,
                        actor=actor,
                        actor_params=agent_state.actor_params,
                        obs=obs,
                        noise_scale=noise_scale_value,  # Pass JAX array directly
                        noise_type='gaussian',
                        noise_state=None,  # Gaussian doesn't need state
                        epsilon=0.0,
                        discrete_action=self.config.discrete_action,
                    )
                    return action
                
                # vmap over environments
                agent_keys_flat = random.split(random.fold_in(key, i), n_envs)
                actions = jax.vmap(select_single)((agent_keys_flat, agent_obs))
            else:
                # No noise - just apply actor to batch
                actions = actor.apply(agent_state.actor_params, agent_obs, training=False)
            
            all_actions.append(actions)
            new_agent_states.append(agent_state)
        
        # Stack: list of (n_envs, action_dim) -> (n_envs, n_agents, action_dim)
        actions_batch = jnp.stack(all_actions, axis=1)
        
        new_state = state.replace(agent_states=new_agent_states)
        return actions_batch, new_state
    
    def store_transitions_batched(
        self,
        state: MADDPGState,
        obs_batch: jax.Array,
        actions_batch: jax.Array,
        rewards_batch: jax.Array,
        next_obs_batch: jax.Array,
        dones_batch: jax.Array,
        action_priors_batch: Optional[jax.Array] = None,
    ) -> MADDPGState:
        """Store transitions from multiple parallel environments at once.
        
        This is much more efficient than storing one at a time.
        
        Args:
            state: Current MADDPG state
            obs_batch: Observations, shape (n_envs, n_agents, obs_dim)
            actions_batch: Actions, shape (n_envs, n_agents, action_dim)
            rewards_batch: Rewards, shape (n_envs, n_agents)
            next_obs_batch: Next observations, shape (n_envs, n_agents, obs_dim)
            dones_batch: Done flags, shape (n_envs, n_agents)
            action_priors_batch: Optional prior actions, shape (n_envs, n_agents, action_dim)
            
        Returns:
            Updated MADDPGState
        """
        # Create batched transition - shapes already match buffer format
        transitions = Transition(
            obs=obs_batch,
            actions=actions_batch,
            rewards=rewards_batch,
            next_obs=next_obs_batch,
            dones=dones_batch,
            global_state=None,
            next_global_state=None,
            log_probs=None,
            action_priors=action_priors_batch,
        )
        
        # Add batch to buffer
        new_buffer_state = self.buffer.add_batch(state.buffer_state, transitions)
        
        return state.replace(buffer_state=new_buffer_state)

    def update(
        self,
        key: jax.Array,
        state: MADDPGState,
        action_priors: Optional[jax.Array] = None,
    ) -> Tuple[MADDPGState, Dict[str, Any]]:
        """Perform one update step for all agents.
        
        Args:
            key: JAX random key
            state: Current MADDPG state
            action_priors: Optional action priors for regularization
            
        Returns:
            new_state: Updated state
            info: Dictionary with training metrics
        """
        config = self.config
        
        # Check if we can update
        can_update = self.buffer.can_sample(state.buffer_state, config.batch_size)
        
        if not can_update:
            return state, {'can_update': False}
        
        # Sample from buffer - returns BatchTransition namedtuple
        key, sample_key = random.split(key)
        batch = self.buffer.sample(state.buffer_state, sample_key, config.batch_size)
        
        # batch.obs has shape (batch_size, n_agents, obs_dim)
        # batch.actions has shape (batch_size, n_agents, action_dim)
        # batch.rewards has shape (batch_size, n_agents)
        # batch.dones has shape (batch_size, n_agents)
        
        # Flatten observations and actions for critic input
        batch_size = batch.obs.shape[0]
        
        # Flatten per-agent obs to total_obs: (batch, n_agents * obs_dim)
        obs_flat = batch.obs.reshape(batch_size, -1)  # (batch, n_agents * max_obs_dim)
        next_obs_flat = batch.next_obs.reshape(batch_size, -1)
        
        # Flatten per-agent actions: (batch, n_agents * action_dim)
        actions_flat = batch.actions.reshape(batch_size, -1)
        
        # Use global_state if available, else use flattened obs
        if batch.global_state is not None:
            global_states = batch.global_state
            next_global_states = batch.next_global_state
        else:
            global_states = obs_flat
            next_global_states = next_obs_flat
        
        # Compute target actions for all agents
        # For each agent, extract their observation and compute target action
        target_actions_list = []
        for i, (actor, agent_state) in enumerate(zip(self.actors, state.agent_states)):
            # Extract this agent's obs from batch: (batch, obs_dim)
            agent_next_obs = batch.next_obs[:, i, :self.obs_dims[i]]
            target_action = select_target_action(
                actor, agent_state.target_actor_params, agent_next_obs
            )
            # Pad if needed
            if target_action.shape[-1] < self.buffer.action_dim:
                target_action = jnp.pad(
                    target_action, 
                    ((0, 0), (0, self.buffer.action_dim - target_action.shape[-1]))
                )
            target_actions_list.append(target_action)
        
        # Stack and flatten target actions
        target_actions = jnp.stack(target_actions_list, axis=1).reshape(batch_size, -1)
        
        # Update each agent
        new_agent_states = list(state.agent_states)
        info = {'can_update': True}
        
        for agent_i in range(self.n_agents):
            key, update_key = random.split(key)
            
            # Get this agent's data
            agent_state = new_agent_states[agent_i]
            actor = self.actors[agent_i]
            critic = self.critics[agent_i]
            
            # Agent's observation from batch: (batch, obs_dim)
            agent_obs = batch.obs[:, agent_i, :self.obs_dims[agent_i]]
            
            # Agent's reward: (batch, 1)
            agent_reward = batch.rewards[:, agent_i:agent_i+1]
            
            # Construct critic inputs
            # Current: global_state + current_actions (flattened)
            critic_input = jnp.concatenate([global_states, actions_flat], axis=-1)
            
            # Next: next_global_state + target_actions
            next_critic_input = jnp.concatenate([next_global_states, target_actions], axis=-1)
            
            # Done flag (use this agent's done)
            agent_dones = batch.dones[:, agent_i:agent_i+1]
            
            # Update critic
            key, critic_key = random.split(key)
            agent_state, critic_info = update_critic(
                agent_state=agent_state,
                critic=critic,
                actor=actor,
                optimizer=self.critic_optimizer,
                global_obs=global_states,
                all_actions=actions_flat,
                rewards=agent_reward,
                next_global_obs=next_global_states,
                next_all_actions=target_actions,
                dones=agent_dones,
                gamma=config.gamma,
            )
            
            # Update actor
            # Need to construct other agents' actions
            # actions_flat has shape (batch, n_agents * action_dim)
            action_dim = self.buffer.action_dim
            action_start = agent_i * action_dim
            action_end = action_start + action_dim
            
            other_actions_before = actions_flat[:, :action_start]
            other_actions_after = actions_flat[:, action_end:]
            all_actions_except_agent = jnp.concatenate(
                [other_actions_before, other_actions_after], axis=-1
            )
            
            # Get action prior for this agent if provided
            agent_prior = None
            if action_priors is not None:
                agent_prior = action_priors[:, action_start:action_end]
            
            key, actor_key = random.split(key)
            agent_state, actor_info = update_actor(
                agent_state=agent_state,
                actor=actor,
                critic=critic,
                optimizer=self.actor_optimizer,
                global_obs=global_states,
                agent_obs=agent_obs,
                all_actions_except_agent=all_actions_except_agent,
                agent_action_idx=agent_i,
                action_dim=self.action_dims[agent_i],
                action_prior=agent_prior,
                prior_weight=config.prior_weight,
            )
            
            new_agent_states[agent_i] = agent_state
            
            # Record info
            info[f'agent_{agent_i}/critic_loss'] = critic_info['critic_loss']
            info[f'agent_{agent_i}/actor_loss'] = actor_info['actor_loss']
            info[f'agent_{agent_i}/q_value'] = critic_info['q_values']
        
        # Update all target networks
        for i in range(self.n_agents):
            new_agent_states[i] = update_targets(new_agent_states[i], config.tau)
        
        # Update noise scale
        new_step = state.step + 1
        new_noise_scale = get_noise_scale(
            int(new_step),
            config.noise_scale_initial,
            config.noise_scale_final,
            config.noise_decay_steps,
            config.noise_schedule,
        )
        
        new_state = state.replace(
            agent_states=new_agent_states,
            step=new_step,
            noise_scale=jnp.array(new_noise_scale, dtype=jnp.float32),
        )
        
        info['step'] = new_step
        info['noise_scale'] = new_noise_scale
        
        return new_state, info
    
    def reset_noise(self, state: MADDPGState) -> MADDPGState:
        """Reset exploration noise for all agents.
        
        Args:
            state: Current state
            
        Returns:
            State with reset noise
        """
        new_agent_states = []
        for i, agent_state in enumerate(state.agent_states):
            new_state = reset_noise(agent_state, self.action_dims[i])
            new_agent_states.append(new_state)
        
        return state.replace(agent_states=new_agent_states)
    
    def increment_episode(self, state: MADDPGState) -> MADDPGState:
        """Increment episode counter and reset noise.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        state = self.reset_noise(state)
        return state.replace(episode=state.episode + 1)
    
    def get_params(self, state: MADDPGState) -> Dict[str, Any]:
        """Get all parameters for saving.
        
        Args:
            state: Current state
            
        Returns:
            Dictionary of parameters
        """
        agent_params = [get_agent_params(s) for s in state.agent_states]
        
        return {
            'agent_params': agent_params,
            'step': state.step,
            'episode': state.episode,
            'noise_scale': state.noise_scale,
            'config': self.config._asdict(),
        }
    
    def load_params(
        self,
        state: MADDPGState,
        params: Dict[str, Any],
    ) -> MADDPGState:
        """Load parameters from a saved dict.
        
        Args:
            state: Current state (for structure)
            params: Parameters to load
            
        Returns:
            State with loaded parameters
        """
        new_agent_states = []
        for agent_state, agent_params in zip(state.agent_states, params['agent_params']):
            new_state = load_agent_params(agent_state, agent_params)
            new_agent_states.append(new_state)
        
        return state.replace(
            agent_states=new_agent_states,
            step=params.get('step', state.step),
            episode=params.get('episode', state.episode),
            noise_scale=params.get('noise_scale', state.noise_scale),
        )


# ============================================================================
# Factory Functions
# ============================================================================

def make_maddpg(
    n_agents: int,
    obs_dims: Union[int, Tuple[int, ...]],
    action_dims: Union[int, Tuple[int, ...]],
    **kwargs,
) -> MADDPG:
    """Create a MADDPG instance with simplified interface.
    
    Args:
        n_agents: Number of agents
        obs_dims: Observation dimension(s). If int, same for all agents.
        action_dims: Action dimension(s). If int, same for all agents.
        **kwargs: Additional config parameters
        
    Returns:
        MADDPG instance
    """
    # Handle uniform dimensions
    if isinstance(obs_dims, int):
        obs_dims = tuple([obs_dims] * n_agents)
    if isinstance(action_dims, int):
        action_dims = tuple([action_dims] * n_agents)
    
    config = MADDPGConfig(
        n_agents=n_agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        **kwargs,
    )
    
    return MADDPG(config)


def make_maddpg_from_env(
    env,
    **kwargs,
) -> MADDPG:
    """Create MADDPG from environment interface.
    
    Expects env to have:
    - n_agents: int
    - observation_spaces: list of spaces with .shape
    - action_spaces: list of spaces with .shape
    
    Args:
        env: Environment instance
        **kwargs: Additional config parameters
        
    Returns:
        MADDPG instance
    """
    n_agents = env.n_agents
    
    # Get dimensions from spaces
    obs_dims = tuple(space.shape[0] for space in env.observation_spaces)
    action_dims = tuple(space.shape[0] for space in env.action_spaces)
    
    # Check for global state
    global_state_dim = getattr(env, 'global_state_dim', None)
    
    return make_maddpg(
        n_agents=n_agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        global_state_dim=global_state_dim,
        **kwargs,
    )


# ============================================================================
# Training Loop Utilities  
# ============================================================================

def train_step(
    key: jax.Array,
    maddpg: MADDPG,
    state: MADDPGState,
    env_state: Any,
    env_step_fn: Callable,
    env_params: Any,
) -> Tuple[MADDPGState, Any, Dict[str, Any]]:
    """Perform one training step.
    
    This is a utility function for a common training pattern.
    
    Args:
        key: JAX random key
        maddpg: MADDPG instance
        state: MADDPG state
        env_state: Environment state
        env_step_fn: Function to step environment
        env_params: Environment parameters
        
    Returns:
        new_maddpg_state: Updated MADDPG state
        new_env_state: Updated environment state
        info: Training info
    """
    key, action_key, step_key, update_key = random.split(key, 4)
    
    # Get observations from env_state (environment-specific)
    # This assumes env_state has an 'obs' attribute
    observations = env_state.obs if hasattr(env_state, 'obs') else env_state
    
    # Select actions
    actions, log_probs, state = maddpg.select_actions(
        action_key, state, observations, explore=True
    )
    
    # Step environment
    next_obs, new_env_state, rewards, dones, env_info = env_step_fn(
        step_key, env_state, actions, env_params
    )
    
    # Store transition
    state = maddpg.store_transition(
        state=state,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        dones=dones,
        log_probs=log_probs,
    )
    
    # Update if ready
    info = {}
    if state.step >= maddpg.config.warmup_steps:
        if state.step % maddpg.config.update_every == 0:
            for _ in range(maddpg.config.updates_per_step):
                update_key, key = random.split(update_key)
                state, update_info = maddpg.update(update_key, state)
                info.update(update_info)
    
    # Handle episode end
    if any(d for d in dones):
        state = maddpg.increment_episode(state)
    
    info['step'] = state.step
    info['episode'] = state.episode
    
    return state, new_env_state, info
