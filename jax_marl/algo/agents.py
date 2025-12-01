"""DDPG Agent for MADDPG algorithm.

This module provides the DDPGAgent class that encapsulates:
- Actor (policy) and Critic networks with their target networks
- Optimizer states for both networks
- Exploration noise handling
- Single-step and batched action selection
- Update functions for actor and critic

All operations are designed to be:
- Stateless (state passed explicitly)
- JIT-compilable
- vmap-compatible for multi-agent scenarios
"""

from typing import Tuple, NamedTuple, Optional, Dict, Any, Callable, Union
import jax
import jax.numpy as jnp
from jax import random
from flax import struct
from flax.training import train_state
import flax.linen as nn
import optax
from functools import partial

from networks import Actor, ActorDiscrete, Critic, CriticTwin, create_maddpg_networks
from utils import soft_update, hard_update, gumbel_softmax, onehot_from_logits
from noise import (
    gaussian_noise, add_gaussian_noise, GaussianNoiseParams,
    OUNoiseState, OUNoiseParams, ou_noise_init, ou_noise_step, ou_noise_reset, add_ou_noise,
    NoiseScheduler, NoiseSchedulerState,
)


# ============================================================================
# Agent State
# ============================================================================

@struct.dataclass
class DDPGAgentState:
    """Complete state for a DDPG agent.
    
    Contains all trainable parameters, target parameters, and optimizer states.
    This is an immutable dataclass that gets updated functionally.
    
    Attributes:
        actor_params: Parameters for the actor (policy) network
        critic_params: Parameters for the critic (value) network
        target_actor_params: Parameters for the target actor network
        target_critic_params: Parameters for the target critic network
        actor_opt_state: Optimizer state for actor
        critic_opt_state: Optimizer state for critic
        noise_state: Optional noise state for OU noise
        step: Training step counter
    """
    actor_params: Any
    critic_params: Any
    target_actor_params: Any
    target_critic_params: Any
    actor_opt_state: Any
    critic_opt_state: Any
    noise_state: Optional[Any] = None
    step: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))


class AgentConfig(NamedTuple):
    """Configuration for a DDPG agent.
    
    Attributes:
        obs_dim: Observation dimension for this agent
        action_dim: Action dimension for this agent
        critic_input_dim: Input dimension for critic (global obs + all actions)
        hidden_dims: Hidden layer dimensions
        lr_actor: Learning rate for actor
        lr_critic: Learning rate for critic
        gamma: Discount factor
        tau: Soft update coefficient
        discrete_action: Whether actions are discrete
        noise_type: 'gaussian' or 'ou'
        noise_scale: Initial noise scale
        noise_final_scale: Final noise scale (for scheduling)
        use_layer_norm: Whether to use layer normalization
        max_grad_norm: Maximum gradient norm for clipping (None = no clipping)
    """
    obs_dim: int
    action_dim: int
    critic_input_dim: int
    hidden_dims: Tuple[int, ...] = (64, 64)
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    gamma: float = 0.95
    tau: float = 0.01
    discrete_action: bool = False
    noise_type: str = 'gaussian'
    noise_scale: float = 0.1
    noise_final_scale: float = 0.01
    use_layer_norm: bool = False
    max_grad_norm: Optional[float] = None


# ============================================================================
# Agent Initialization
# ============================================================================

def create_agent(
    key: jax.Array,
    config: AgentConfig,
) -> Tuple[DDPGAgentState, Actor, Critic]:
    """Create a DDPG agent with initialized networks and optimizer states.
    
    Args:
        key: JAX random key
        config: Agent configuration
        
    Returns:
        agent_state: Initial agent state
        actor: Actor network (for applying to get actions)
        critic: Critic network (for computing Q-values)
    """
    key_actor, key_critic = random.split(key)
    
    # Create networks
    if config.discrete_action:
        actor = ActorDiscrete(
            n_actions=config.action_dim,  # ActorDiscrete uses n_actions
            hidden_dims=config.hidden_dims,
            use_layer_norm=config.use_layer_norm,
        )
    else:
        actor = Actor(
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            use_layer_norm=config.use_layer_norm,
        )
    
    critic = Critic(
        hidden_dims=config.hidden_dims,
        use_layer_norm=config.use_layer_norm,
    )
    
    # Initialize parameters
    # critic_input_dim = obs_dim + total_action_dim for MADDPG
    # We split it for the critic's (obs, action) interface
    dummy_obs = jnp.zeros((1, config.obs_dim))
    dummy_action = jnp.zeros((1, config.critic_input_dim - config.obs_dim))
    # For critic, we need dummy obs and action separately
    # critic_input_dim should be obs_dim (for global state) + total_actions
    # But if critic_input_dim already includes both, we need to be careful
    # Let's use a simpler approach: assume critic_input_dim = global_obs + all_actions
    # and we'll create dummy inputs for init
    dummy_critic_obs = jnp.zeros((1, config.obs_dim))
    dummy_critic_action = jnp.zeros((1, config.critic_input_dim - config.obs_dim))
    
    actor_params = actor.init(key_actor, dummy_obs)
    critic_params = critic.init(key_critic, dummy_critic_obs, dummy_critic_action)
    
    # Initialize target networks with same parameters
    target_actor_params = actor_params
    target_critic_params = critic_params
    
    # Create optimizers
    if config.max_grad_norm is not None:
        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.lr_actor),
        )
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.lr_critic),
        )
    else:
        actor_optimizer = optax.adam(config.lr_actor)
        critic_optimizer = optax.adam(config.lr_critic)
    
    actor_opt_state = actor_optimizer.init(actor_params)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # Initialize noise state if using OU noise
    noise_state = None
    if config.noise_type == 'ou':
        noise_state = ou_noise_init(config.action_dim)
    
    agent_state = DDPGAgentState(
        actor_params=actor_params,
        critic_params=critic_params,
        target_actor_params=target_actor_params,
        target_critic_params=target_critic_params,
        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,
        noise_state=noise_state,
        step=jnp.array(0, dtype=jnp.int32),
    )
    
    return agent_state, actor, critic


def create_agent_with_networks(
    key: jax.Array,
    config: AgentConfig,
    actor: nn.Module,
    critic: nn.Module,
) -> DDPGAgentState:
    """Create agent state with pre-defined network architectures.
    
    Useful when you want to share network definitions across agents.
    
    Args:
        key: JAX random key
        config: Agent configuration
        actor: Actor network module
        critic: Critic network module
        
    Returns:
        agent_state: Initial agent state
    """
    key_actor, key_critic = random.split(key)
    
    # Initialize parameters
    dummy_obs = jnp.zeros((1, config.obs_dim))
    dummy_critic_obs = jnp.zeros((1, config.obs_dim))
    dummy_critic_action = jnp.zeros((1, config.critic_input_dim - config.obs_dim))
    
    actor_params = actor.init(key_actor, dummy_obs)
    critic_params = critic.init(key_critic, dummy_critic_obs, dummy_critic_action)
    
    # Initialize target networks with same parameters
    target_actor_params = actor_params
    target_critic_params = critic_params
    
    # Create optimizers
    if config.max_grad_norm is not None:
        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.lr_actor),
        )
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.lr_critic),
        )
    else:
        actor_optimizer = optax.adam(config.lr_actor)
        critic_optimizer = optax.adam(config.lr_critic)
    
    actor_opt_state = actor_optimizer.init(actor_params)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # Initialize noise state if using OU noise
    noise_state = None
    if config.noise_type == 'ou':
        noise_state = ou_noise_init(config.action_dim)
    
    return DDPGAgentState(
        actor_params=actor_params,
        critic_params=critic_params,
        target_actor_params=target_actor_params,
        target_critic_params=target_critic_params,
        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,
        noise_state=noise_state,
        step=jnp.array(0, dtype=jnp.int32),
    )


# ============================================================================
# Action Selection
# ============================================================================

def select_action(
    actor: nn.Module,
    actor_params: Any,
    obs: jax.Array,
    training: bool = False,
) -> jax.Array:
    """Select action using the actor network (no noise).
    
    Args:
        actor: Actor network
        actor_params: Actor parameters
        obs: Observation(s), shape (obs_dim,) or (batch, obs_dim)
        training: Whether in training mode (affects dropout if used)
        
    Returns:
        action: Action(s), shape (action_dim,) or (batch, action_dim)
    """
    # Handle single observation
    single_obs = obs.ndim == 1
    if single_obs:
        obs = obs[None, :]
    
    action = actor.apply(actor_params, obs, training=training)
    
    if single_obs:
        action = action[0]
    
    return action


def select_action_with_noise(
    key: jax.Array,
    actor: nn.Module,
    actor_params: Any,
    obs: jax.Array,
    noise_scale: float,
    noise_type: str = 'gaussian',
    noise_state: Optional[OUNoiseState] = None,
    ou_params: Optional[OUNoiseParams] = None,
    epsilon: float = 0.0,
    discrete_action: bool = False,
    clip_low: float = -1.0,
    clip_high: float = 1.0,
) -> Tuple[jax.Array, jax.Array, Optional[OUNoiseState]]:
    """Select action with exploration noise.
    
    For continuous actions:
    - With probability epsilon, sample random uniform action
    - Otherwise, add noise (Gaussian or OU) to policy action
    
    For discrete actions:
    - Use Gumbel-softmax with temperature
    
    Args:
        key: JAX random key
        actor: Actor network
        actor_params: Actor parameters
        obs: Observation(s)
        noise_scale: Scale of exploration noise
        noise_type: 'gaussian' or 'ou'
        noise_state: OU noise state (if noise_type='ou')
        ou_params: OU noise parameters (if noise_type='ou')
        epsilon: Probability of random action (continuous only)
        discrete_action: Whether actions are discrete
        clip_low: Lower bound for action clipping
        clip_high: Upper bound for action clipping
        
    Returns:
        action: Action(s) with exploration
        log_prob: Log probability of action (approximate)
        new_noise_state: Updated noise state (None if Gaussian)
    """
    key_eps, key_random, key_noise = random.split(key, 3)
    
    # Handle single observation
    single_obs = obs.ndim == 1
    if single_obs:
        obs = obs[None, :]
    
    batch_size = obs.shape[0]
    
    # Get policy action
    policy_action = actor.apply(actor_params, obs, training=False)
    action_dim = policy_action.shape[-1]
    
    if discrete_action:
        # For discrete actions, use Gumbel-softmax
        # Temperature inversely related to noise_scale (higher noise = higher temp)
        temperature = jnp.maximum(noise_scale, 0.1)
        action = gumbel_softmax(key_noise, policy_action, temperature, hard=True)
        log_prob = jnp.full((batch_size, 1), -action_dim * jnp.log(action_dim))
        new_noise_state = None
    else:
        # For continuous actions
        # Random action with probability epsilon
        random_action = random.uniform(
            key_random, 
            policy_action.shape, 
            minval=clip_low, 
            maxval=clip_high
        )
        
        # Add noise to policy action
        if noise_type == 'gaussian':
            noise = gaussian_noise(key_noise, policy_action.shape, noise_scale)
            noisy_action = jnp.clip(policy_action + noise, clip_low, clip_high)
            new_noise_state = None
            
            # Log prob of Gaussian noise
            log_prob_noise = -0.5 * jnp.sum((noise / noise_scale) ** 2, axis=-1, keepdims=True)
            log_prob_noise = log_prob_noise - action_dim * jnp.log(noise_scale * jnp.sqrt(2 * jnp.pi))
        else:  # OU noise
            if noise_state is None:
                noise_state = ou_noise_init(action_dim)
            if ou_params is None:
                ou_params = OUNoiseParams(scale=noise_scale, action_dim=action_dim)
            
            # Handle batched OU noise
            if batch_size > 1:
                # For batched case, generate same noise for all (simpler)
                # Or could use vmap for per-sample noise
                new_noise_state, noise = ou_noise_step(key_noise, noise_state, ou_params)
                noise = jnp.broadcast_to(noise, policy_action.shape)
            else:
                new_noise_state, noise = ou_noise_step(key_noise, noise_state, ou_params)
            
            noisy_action = jnp.clip(policy_action + noise, clip_low, clip_high)
            log_prob_noise = jnp.full((batch_size, 1), -action_dim * jnp.log(2.0))  # Approximate
        
        # Epsilon-greedy: random action vs noisy policy action
        use_random = random.uniform(key_eps, (batch_size, 1)) < epsilon
        action = jnp.where(use_random, random_action, noisy_action)
        
        # Log prob: uniform for random, Gaussian for noisy
        log_prob_uniform = -action_dim * jnp.log(clip_high - clip_low)
        log_prob = jnp.where(use_random, log_prob_uniform, log_prob_noise)
    
    if single_obs:
        action = action[0]
        log_prob = log_prob[0]
    
    return action, log_prob, new_noise_state


def select_target_action(
    actor: nn.Module,
    target_actor_params: Any,
    obs: jax.Array,
) -> jax.Array:
    """Select action using the target actor network.
    
    Args:
        actor: Actor network
        target_actor_params: Target actor parameters
        obs: Observation(s)
        
    Returns:
        action: Target action(s)
    """
    return select_action(actor, target_actor_params, obs, training=False)


# ============================================================================
# Update Functions
# ============================================================================

def compute_critic_loss(
    critic: nn.Module,
    critic_params: Any,
    target_critic_params: Any,
    actor: nn.Module,
    target_actor_params: Any,
    global_obs: jax.Array,
    all_actions: jax.Array,
    rewards: jax.Array,
    next_global_obs: jax.Array,
    next_all_actions: jax.Array,
    dones: jax.Array,
    gamma: float,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute critic (TD) loss.
    
    Loss = MSE(Q(s, a), r + gamma * Q_target(s', a'))
    where a' = target_actor(s')
    
    Args:
        critic: Critic network
        critic_params: Current critic parameters
        target_critic_params: Target critic parameters
        actor: Actor network
        target_actor_params: Target actor parameters
        global_obs: Current global observations (batch, global_obs_dim)
        all_actions: All agents' actions (batch, total_action_dim)
        rewards: Rewards received (batch, 1)
        next_global_obs: Next global observations (batch, global_obs_dim)
        next_all_actions: Next actions from all agents (batch, total_action_dim)
        dones: Done flags (batch, 1)
        gamma: Discount factor
        
    Returns:
        loss: Scalar loss value
        info: Dict with additional info (q_values, targets, etc.)
    """
    # Compute current Q-value
    # Critic takes (obs, action) separately
    q_values = critic.apply(critic_params, global_obs, all_actions)
    
    # Compute target Q-value
    next_q_values = critic.apply(target_critic_params, next_global_obs, next_all_actions)
    
    # Bellman target
    targets = rewards + gamma * next_q_values * (1 - dones)
    targets = jax.lax.stop_gradient(targets)
    
    # MSE loss
    loss = jnp.mean((q_values - targets) ** 2)
    
    info = {
        'q_values': jnp.mean(q_values),
        'target_q_values': jnp.mean(targets),
        'td_error': jnp.mean(jnp.abs(q_values - targets)),
    }
    
    return loss, info


def compute_actor_loss(
    actor: nn.Module,
    actor_params: Any,
    critic: nn.Module,
    critic_params: Any,
    global_obs: jax.Array,
    agent_obs: jax.Array,
    all_actions_except_agent: jax.Array,
    agent_action_idx: int,
    action_dim: int,
    action_prior: Optional[jax.Array] = None,
    prior_weight: float = 0.0,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute actor (policy gradient) loss.
    
    Loss = -E[Q(s, a)] + prior_weight * MSE(a, a_prior)
    
    Args:
        actor: Actor network
        actor_params: Actor parameters
        critic: Critic network  
        critic_params: Critic parameters (use current, not target)
        global_obs: Global observations for critic (batch, global_obs_dim)
        agent_obs: This agent's observations (batch, obs_dim)
        all_actions_except_agent: Actions from all other agents (batch, total - action_dim)
        agent_action_idx: Index to insert this agent's action
        action_dim: This agent's action dimension
        action_prior: Prior actions for regularization (optional)
        prior_weight: Weight for prior regularization
        
    Returns:
        loss: Scalar loss value
        info: Dict with additional info
    """
    # Get current policy action
    policy_action = actor.apply(actor_params, agent_obs, training=True)
    
    # Construct full action vector for critic
    # Insert this agent's action at the correct position
    batch_size = agent_obs.shape[0]
    
    # Build all_actions by inserting this agent's action at correct position
    # all_actions_except_agent has shape (batch, total_action_dim - action_dim)
    # We need to insert policy_action at position agent_action_idx * action_dim
    
    if all_actions_except_agent is not None and all_actions_except_agent.shape[-1] > 0:
        # Split and insert agent's action at correct position
        insert_idx = agent_action_idx * action_dim
        before = all_actions_except_agent[:, :insert_idx]
        after = all_actions_except_agent[:, insert_idx:]
        all_actions = jnp.concatenate([before, policy_action, after], axis=-1)
    else:
        all_actions = policy_action
    
    # Q-value (we want to maximize, so negate for loss)
    # Critic takes (obs, action) separately
    q_value = critic.apply(critic_params, global_obs, all_actions)
    policy_loss = -jnp.mean(q_value)
    
    # Prior regularization
    reg_loss = jnp.array(0.0)
    if action_prior is not None and prior_weight > 0:
        # Only compute for valid priors (non-zero)
        valid_mask = jnp.any(jnp.abs(action_prior) > 1e-6, axis=-1, keepdims=True)
        mse = jnp.sum((policy_action - action_prior) ** 2, axis=-1, keepdims=True)
        reg_loss = jnp.sum(mse * valid_mask) / (jnp.sum(valid_mask) + 1e-8)
    
    loss = policy_loss + prior_weight * reg_loss
    
    info = {
        'policy_loss': policy_loss,
        'reg_loss': reg_loss,
        'q_value_mean': jnp.mean(q_value),
        'action_mean': jnp.mean(policy_action),
        'action_std': jnp.std(policy_action),
    }
    
    return loss, info


def update_critic(
    agent_state: DDPGAgentState,
    critic: nn.Module,
    actor: nn.Module,
    optimizer: optax.GradientTransformation,
    global_obs: jax.Array,
    all_actions: jax.Array,
    rewards: jax.Array,
    next_global_obs: jax.Array,
    next_all_actions: jax.Array,
    dones: jax.Array,
    gamma: float,
) -> Tuple[DDPGAgentState, Dict[str, jax.Array]]:
    """Update critic network.
    
    Args:
        agent_state: Current agent state
        critic: Critic network
        actor: Actor network
        optimizer: Critic optimizer
        global_obs: Current global observations (batch, global_obs_dim)
        all_actions: All agents' current actions (batch, total_action_dim)
        rewards: Rewards
        next_global_obs: Next global observations (batch, global_obs_dim)
        next_all_actions: All agents' next actions (batch, total_action_dim)
        dones: Done flags
        gamma: Discount factor
        
    Returns:
        new_agent_state: Updated agent state
        info: Training info dict
    """
    def loss_fn(critic_params):
        return compute_critic_loss(
            critic=critic,
            critic_params=critic_params,
            target_critic_params=agent_state.target_critic_params,
            actor=actor,
            target_actor_params=agent_state.target_actor_params,
            global_obs=global_obs,
            all_actions=all_actions,
            rewards=rewards,
            next_global_obs=next_global_obs,
            next_all_actions=next_all_actions,
            dones=dones,
            gamma=gamma,
        )
    
    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(agent_state.critic_params)
    
    updates, new_opt_state = optimizer.update(grads, agent_state.critic_opt_state, agent_state.critic_params)
    new_critic_params = optax.apply_updates(agent_state.critic_params, updates)
    
    new_agent_state = agent_state.replace(
        critic_params=new_critic_params,
        critic_opt_state=new_opt_state,
    )
    
    info['critic_loss'] = loss
    info['critic_grad_norm'] = optax.global_norm(grads)
    
    return new_agent_state, info


def update_actor(
    agent_state: DDPGAgentState,
    actor: nn.Module,
    critic: nn.Module,
    optimizer: optax.GradientTransformation,
    global_obs: jax.Array,
    agent_obs: jax.Array,
    all_actions_except_agent: jax.Array,
    agent_action_idx: int,
    action_dim: int,
    action_prior: Optional[jax.Array] = None,
    prior_weight: float = 0.0,
) -> Tuple[DDPGAgentState, Dict[str, jax.Array]]:
    """Update actor network.
    
    Args:
        agent_state: Current agent state
        actor: Actor network
        critic: Critic network
        optimizer: Actor optimizer
        global_obs: Global observations for critic
        agent_obs: This agent's observations
        all_actions_except_agent: Actions from other agents
        agent_action_idx: Index for this agent's action
        action_dim: This agent's action dimension
        action_prior: Optional prior actions
        prior_weight: Weight for prior regularization
        
    Returns:
        new_agent_state: Updated agent state
        info: Training info dict
    """
    def loss_fn(actor_params):
        return compute_actor_loss(
            actor=actor,
            actor_params=actor_params,
            critic=critic,
            critic_params=agent_state.critic_params,  # Use current critic
            global_obs=global_obs,
            agent_obs=agent_obs,
            all_actions_except_agent=all_actions_except_agent,
            agent_action_idx=agent_action_idx,
            action_dim=action_dim,
            action_prior=action_prior,
            prior_weight=prior_weight,
        )
    
    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(agent_state.actor_params)
    
    updates, new_opt_state = optimizer.update(grads, agent_state.actor_opt_state, agent_state.actor_params)
    new_actor_params = optax.apply_updates(agent_state.actor_params, updates)
    
    new_agent_state = agent_state.replace(
        actor_params=new_actor_params,
        actor_opt_state=new_opt_state,
    )
    
    info['actor_loss'] = loss
    info['actor_grad_norm'] = optax.global_norm(grads)
    
    return new_agent_state, info


def update_targets(
    agent_state: DDPGAgentState,
    tau: float,
) -> DDPGAgentState:
    """Soft update target networks.
    
    Args:
        agent_state: Current agent state
        tau: Soft update coefficient
        
    Returns:
        new_agent_state: State with updated targets
    """
    new_target_actor_params = soft_update(
        agent_state.target_actor_params,
        agent_state.actor_params,
        tau,
    )
    new_target_critic_params = soft_update(
        agent_state.target_critic_params,
        agent_state.critic_params,
        tau,
    )
    
    return agent_state.replace(
        target_actor_params=new_target_actor_params,
        target_critic_params=new_target_critic_params,
        step=agent_state.step + 1,
    )


# ============================================================================
# Noise Management
# ============================================================================

def reset_noise(
    agent_state: DDPGAgentState,
    action_dim: int,
) -> DDPGAgentState:
    """Reset the agent's exploration noise state.
    
    Args:
        agent_state: Current agent state
        action_dim: Action dimension
        
    Returns:
        new_agent_state: State with reset noise
    """
    if agent_state.noise_state is not None:
        new_noise_state = ou_noise_init(action_dim)
        return agent_state.replace(noise_state=new_noise_state)
    return agent_state


def get_noise_scale(
    step: int,
    initial_scale: float,
    final_scale: float,
    total_steps: int,
    schedule: str = 'linear',
) -> float:
    """Get noise scale based on training progress.
    
    Args:
        step: Current training step
        initial_scale: Initial noise scale
        final_scale: Final noise scale
        total_steps: Total training steps
        schedule: 'linear', 'exponential', or 'cosine'
        
    Returns:
        Current noise scale
    """
    from noise import linear_schedule, exponential_schedule, cosine_schedule
    
    if schedule == 'linear':
        return linear_schedule(initial_scale, final_scale, step, total_steps)
    elif schedule == 'exponential':
        decay_rate = -jnp.log(final_scale / initial_scale) / total_steps
        return exponential_schedule(initial_scale, final_scale, step, decay_rate)
    elif schedule == 'cosine':
        return cosine_schedule(initial_scale, final_scale, step, total_steps)
    else:
        return initial_scale


# ============================================================================
# Multi-Agent Utilities
# ============================================================================

def create_all_agents(
    key: jax.Array,
    configs: list,
) -> Tuple[list, list, list]:
    """Create multiple DDPG agents.
    
    Args:
        key: JAX random key
        configs: List of AgentConfig for each agent
        
    Returns:
        agent_states: List of DDPGAgentState
        actors: List of Actor networks
        critics: List of Critic networks
    """
    n_agents = len(configs)
    keys = random.split(key, n_agents)
    
    agent_states = []
    actors = []
    critics = []
    
    for i, (k, config) in enumerate(zip(keys, configs)):
        state, actor, critic = create_agent(k, config)
        agent_states.append(state)
        actors.append(actor)
        critics.append(critic)
    
    return agent_states, actors, critics


def select_all_actions(
    actors: list,
    actor_params_list: list,
    obs_list: list,
    training: bool = False,
) -> list:
    """Select actions for all agents.
    
    Args:
        actors: List of Actor networks
        actor_params_list: List of actor parameters
        obs_list: List of observations for each agent
        training: Whether in training mode
        
    Returns:
        actions: List of actions for each agent
    """
    actions = []
    for actor, params, obs in zip(actors, actor_params_list, obs_list):
        action = select_action(actor, params, obs, training)
        actions.append(action)
    return actions


def select_all_actions_with_noise(
    key: jax.Array,
    actors: list,
    agent_states: list,
    obs_list: list,
    noise_scale: float,
    noise_type: str = 'gaussian',
    epsilon: float = 0.0,
    discrete_action: bool = False,
) -> Tuple[list, list, list]:
    """Select actions with noise for all agents.
    
    Args:
        key: JAX random key
        actors: List of Actor networks
        agent_states: List of DDPGAgentState
        obs_list: List of observations
        noise_scale: Current noise scale
        noise_type: 'gaussian' or 'ou'
        epsilon: Random action probability
        discrete_action: Whether discrete actions
        
    Returns:
        actions: List of actions
        log_probs: List of log probabilities
        new_agent_states: List of updated states (for OU noise)
    """
    n_agents = len(actors)
    keys = random.split(key, n_agents)
    
    actions = []
    log_probs = []
    new_agent_states = []
    
    for i, (k, actor, state, obs) in enumerate(zip(keys, actors, agent_states, obs_list)):
        action, log_prob, new_noise_state = select_action_with_noise(
            key=k,
            actor=actor,
            actor_params=state.actor_params,
            obs=obs,
            noise_scale=noise_scale,
            noise_type=noise_type,
            noise_state=state.noise_state,
            epsilon=epsilon,
            discrete_action=discrete_action,
        )
        
        actions.append(action)
        log_probs.append(log_prob)
        
        if new_noise_state is not None:
            new_state = state.replace(noise_state=new_noise_state)
        else:
            new_state = state
        new_agent_states.append(new_state)
    
    return actions, log_probs, new_agent_states


# ============================================================================
# Serialization
# ============================================================================

def get_agent_params(agent_state: DDPGAgentState) -> Dict[str, Any]:
    """Get all parameters from agent state for saving.
    
    Args:
        agent_state: Agent state
        
    Returns:
        params_dict: Dictionary of all parameters
    """
    return {
        'actor_params': agent_state.actor_params,
        'critic_params': agent_state.critic_params,
        'target_actor_params': agent_state.target_actor_params,
        'target_critic_params': agent_state.target_critic_params,
        'actor_opt_state': agent_state.actor_opt_state,
        'critic_opt_state': agent_state.critic_opt_state,
        'step': agent_state.step,
    }


def load_agent_params(
    agent_state: DDPGAgentState,
    params_dict: Dict[str, Any],
) -> DDPGAgentState:
    """Load parameters into agent state.
    
    Args:
        agent_state: Current agent state (for structure)
        params_dict: Dictionary of parameters to load
        
    Returns:
        new_agent_state: State with loaded parameters
    """
    return agent_state.replace(
        actor_params=params_dict['actor_params'],
        critic_params=params_dict['critic_params'],
        target_actor_params=params_dict['target_actor_params'],
        target_critic_params=params_dict['target_critic_params'],
        actor_opt_state=params_dict['actor_opt_state'],
        critic_opt_state=params_dict['critic_opt_state'],
        step=params_dict.get('step', jnp.array(0, dtype=jnp.int32)),
    )
