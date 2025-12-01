"""MADDPG Algorithm - JAX Implementation.

Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation
using JAX, Flax, and Optax. Supports centralized training with 
decentralized execution (CTDE).

Quick Start:
    from jax_marl.algo import make_maddpg
    
    # Create MADDPG instance
    maddpg = make_maddpg(n_agents=3, obs_dims=10, action_dims=2)
    state = maddpg.init(key)
    
    # Select actions (with exploration)
    actions, log_probs, state = maddpg.select_actions(key, state, observations)
    
    # Store transition
    state = maddpg.store_transition(state, obs, actions, rewards, next_obs, dones)
    
    # Update networks
    state, info = maddpg.update(key, state)

Modules:
    - maddpg: Main MADDPG coordinator class
    - agents: DDPG agent implementation  
    - networks: Actor and Critic neural networks
    - buffers: Replay buffer implementations
    - noise: Exploration noise (Gaussian, OU)
    - utils: Utility functions
"""

# Version
__version__ = "0.1.0"

# Main MADDPG exports
from .maddpg import (
    MADDPG,
    MADDPGConfig,
    MADDPGState,
    make_maddpg,
)

# Agent exports
from .agents import (
    DDPGAgentState,
    AgentConfig,
    create_agent,
    select_action,
    select_action_with_noise,
    select_target_action,
    update_actor,
    update_critic,
    update_targets,
    create_all_agents,
    select_all_actions,
    select_all_actions_with_noise,
    get_agent_params,
    load_agent_params,
)

# Network exports
from .networks import (
    Actor,
    ActorDiscrete,
    Critic,
    CriticTwin,
    ActorCritic,
    MADDPGNetworks,
    create_actor,
    create_actor_discrete,
    create_critic,
    create_critic_twin,
    create_maddpg_networks,
    create_maddpg_networks_shared_critic,
    create_all_agents_networks,
    create_shared_critic_networks,
    count_parameters,
    print_network_summary,
)

# Buffer exports
from .buffers import (
    ReplayBuffer,
    ReplayBufferState,
    Transition,
    BatchTransition,
    PerAgentReplayBuffer,
    PerAgentBufferState,
)

# Noise exports
from .noise import (
    GaussianNoiseParams,
    gaussian_noise,
    gaussian_noise_log_prob,
    add_gaussian_noise,
    OUNoiseState,
    OUNoiseParams,
    ou_noise_init,
    ou_noise_reset,
    ou_noise_step,
    add_ou_noise,
    NoiseScheduler,
    NoiseSchedulerState,
    linear_schedule,
    exponential_schedule,
    cosine_schedule,
    warmup_linear_schedule,
    gaussian_noise_batch,
    add_gaussian_noise_batch,
    ou_noise_init_batch,
    ou_noise_step_batch,
)

# Utility exports
from .utils import (
    soft_update,
    hard_update,
    gumbel_softmax,
    onehot_from_logits,
    onehot_from_logits_epsilon_greedy,
    clip_by_global_norm,
    explained_variance,
    normalize_advantages,
    scale_rewards,
    compute_gae,
    polyak_average,
    huber_loss,
    mse_loss,
    td_target,
    get_gradient_norm,
)

__all__ = [
    # Version
    "__version__",
    # MADDPG
    "MADDPG",
    "MADDPGConfig", 
    "MADDPGState",
    "make_maddpg",
    # Agents
    "DDPGAgentState",
    "AgentConfig",
    "create_agent",
    "select_action",
    "select_action_with_noise",
    "select_target_action",
    "update_actor",
    "update_critic",
    "update_targets",
    "create_all_agents",
    "select_all_actions",
    "select_all_actions_with_noise",
    "get_agent_params",
    "load_agent_params",
    # Networks
    "Actor",
    "ActorDiscrete",
    "Critic",
    "CriticTwin",
    "ActorCritic",
    "MADDPGNetworks",
    "create_actor",
    "create_actor_discrete",
    "create_critic",
    "create_critic_twin",
    "create_maddpg_networks",
    "create_maddpg_networks_shared_critic",
    "create_all_agents_networks",
    "create_shared_critic_networks",
    "count_parameters",
    "print_network_summary",
    # Buffers
    "ReplayBuffer",
    "ReplayBufferState",
    "Transition",
    "BatchTransition",
    "PerAgentReplayBuffer",
    "PerAgentBufferState",
    # Noise
    "GaussianNoiseParams",
    "gaussian_noise",
    "gaussian_noise_log_prob",
    "add_gaussian_noise",
    "OUNoiseState",
    "OUNoiseParams",
    "ou_noise_init",
    "ou_noise_reset",
    "ou_noise_step",
    "add_ou_noise",
    "NoiseScheduler",
    "NoiseSchedulerState",
    "linear_schedule",
    "exponential_schedule",
    "cosine_schedule",
    "warmup_linear_schedule",
    "gaussian_noise_batch",
    "add_gaussian_noise_batch",
    "ou_noise_init_batch",
    "ou_noise_step_batch",
    # Utils
    "soft_update",
    "hard_update",
    "gumbel_softmax",
    "onehot_from_logits",
    "onehot_from_logits_epsilon_greedy",
    "clip_by_global_norm",
    "explained_variance",
    "normalize_advantages",
    "scale_rewards",
    "compute_gae",
    "polyak_average",
    "huber_loss",
    "mse_loss",
    "td_target",
    "get_gradient_norm",
]
