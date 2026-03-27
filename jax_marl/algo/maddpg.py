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
        select_target_action_with_smoothing,
        update_critic,
        update_actor,
        update_critic_td3,
        update_actor_td3,
        update_targets,
        reset_noise,
        get_noise_scale,
        get_agent_params,
        load_agent_params,
        update_critic_ctm,
        update_actor_ctm,
    )
    from .networks import Actor, ActorDiscrete, Critic, CriticTwin, CTMCritic, create_ctm_critic
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
        select_target_action_with_smoothing,
        update_critic,
        update_actor,
        update_critic_td3,
        update_actor_td3,
        update_targets,
        reset_noise,
        get_noise_scale,
        get_agent_params,
        load_agent_params,
        update_critic_ctm,
        update_actor_ctm,
    )
    from networks import Actor, ActorDiscrete, Critic, CriticTwin, CTMCritic, create_ctm_critic
    from buffers import ReplayBuffer, ReplayBufferState, PerAgentReplayBuffer, Transition
    from noise import OUNoiseState, ou_noise_init
    from utils import soft_update


# ============================================================================
# Trajectory Utilities
# ============================================================================

def _reorder_trajectory(raw_traj: jax.Array, traj_idx: jax.Array, traj_len: int) -> jax.Array:
    """Reorder circular trajectory buffer to chronological order (oldest first, newest last).

    The environment stores positions in a circular buffer.  At store time we
    pay the gather cost once; at sample time the CTMCritic receives a clean
    chronological sequence with no circular-buffer awareness needed.

    Args:
        raw_traj: (traj_len, n_agents, 2) circular buffer as stored by the env
        traj_idx: scalar int — the index that was written *last* (newest entry)
        traj_len: static length of the trajectory buffer

    Returns:
        (traj_len, n_agents, 2) chronological order: index 0 is oldest, -1 is newest
    """
    indices = (jnp.arange(traj_len) + traj_idx + 1) % traj_len
    return raw_traj[indices]


# ============================================================================
# MADDPG State
# ============================================================================

@struct.dataclass
class MADDPGState:
    """Complete state for MADDPG training.

    Attributes:
        agent_states: Stacked DDPGAgentState where every param leaf has shape (n_agents, ...)
        buffer_state: Replay buffer state
        step: Global training step counter
        episode: Episode counter
        noise_scale: Current exploration noise scale
    """
    agent_states: DDPGAgentState
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
        updates_per_step: Number of gradient updates per step
        discrete_action: Whether actions are discrete
        use_layer_norm: Whether to use layer normalization
        max_grad_norm: Maximum gradient norm (None = no clipping)
        prior_weight: Weight for action prior regularization
        shared_critic: Whether to share critic across agents
        
        # TD3 enhancements
        use_td3: Whether to use TD3 improvements (twin critics, delayed updates, target smoothing)
        policy_delay: Update actor every N critic updates (TD3)
        target_noise: Stddev of noise added to target actions (TD3 target policy smoothing)
        target_noise_clip: Clip range for target noise (TD3)
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
    updates_per_step: int = 1
    discrete_action: bool = False
    use_layer_norm: bool = True  # Enable layer norm for better training stability
    max_grad_norm: Optional[float] = None
    prior_weight: float = 0.0
    shared_critic: bool = False
    # TD3 enhancements
    use_td3: bool = True  # Enable TD3 by default for better convergence
    policy_delay: int = 2  # Update actor every 2 critic updates
    target_noise: float = 0.2  # Noise added to target actions
    target_noise_clip: float = 0.5  # Clip range for target noise

    # CTM temporal critic (use_ctm_critic=True replaces MLP critic with CTMCritic).
    # Set use_td3=False when enabling CTM — TD3's twin-critic and actor updates are
    # bypassed entirely; target smoothing (the only remaining TD3 effect) conflicts
    # with CTM's certainty-discounted Bellman target (D3). Policy delay still applies.
    use_ctm_critic: bool = False
    ctm_traj_len: int = 15              # trajectory length stored by the environment
    ctm_iterations: int = 8
    ctm_d_model: int = 64
    ctm_d_input: int = 128
    ctm_memory_length: int = 5
    ctm_n_synch_out: int = 8
    ctm_memory_hidden_dims: int = 8


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
        state = maddpg.store_transitions_batched(state, obs, actions, rewards, next_obs, dones)

        # JIT-compiled update
        jit_update = maddpg.create_jit_update()
        state, info = jit_update(state, key)
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
        
        # TD3 configuration
        if config.use_ctm_critic and config.use_td3:
            import warnings
            warnings.warn(
                "use_td3=True has no effect on the CTM critic or actor update paths. "
                "Its only active effect is adding target policy smoothing noise to "
                "target actions before the CTM target network evaluates them. "
                "This conflicts with CTM's certainty-discounted Bellman target (D3), "
                "which already provides conservative Q estimates. "
                "Set use_td3=False when use_ctm_critic=True.",
                UserWarning, stacklevel=2,
            )
        self.use_td3 = config.use_td3
        self.policy_delay = config.policy_delay
        self.target_noise = config.target_noise
        self.target_noise_clip = config.target_noise_clip
        
        # Create a SINGLE actor and critic instance (all agents share the same architecture;
        # only their parameters differ, which are stored in the stacked agent_states).
        if config.discrete_action:
            self.actor = ActorDiscrete(
                n_actions=config.action_dims[0],
                hidden_dims=config.hidden_dims,
                use_layer_norm=config.use_layer_norm,
            )
        else:
            self.actor = Actor(
                action_dim=config.action_dims[0],
                hidden_dims=config.hidden_dims,
                use_layer_norm=config.use_layer_norm,
            )

        # Use CriticTwin for TD3, regular Critic otherwise
        if config.use_td3:
            self.critic = CriticTwin(
                hidden_dims=config.hidden_dims,
                use_layer_norm=config.use_layer_norm,
            )
        else:
            self.critic = Critic(
                hidden_dims=config.hidden_dims,
                use_layer_norm=config.use_layer_norm,
            )

        # Backward-compat aliases so that any code still referencing self.actors[i] or
        # self.critics[i] continues to work (they all point to the same instance).
        self.actors = [self.actor] * self.n_agents
        self.critics = [self.critic] * self.n_agents

        # CTM temporal critic (single shared instance, params in stacked agent_states)
        self.use_ctm_critic = config.use_ctm_critic
        if config.use_ctm_critic:
            self.ctm_critic = CTMCritic(
                iterations=config.ctm_iterations,
                d_model=config.ctm_d_model,
                d_input=config.ctm_d_input,
                memory_length=config.ctm_memory_length,
                n_synch_out=config.ctm_n_synch_out,
                memory_hidden_dims=config.ctm_memory_hidden_dims,
                n_agents=config.n_agents,
                traj_len=config.ctm_traj_len,
            )
        else:
            self.ctm_critic = None

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

        # Create replay buffer (with trajectory fields when CTM critic is enabled)
        self.buffer = ReplayBuffer(
            capacity=config.buffer_size,
            n_agents=config.n_agents,
            obs_dim=max(config.obs_dims),
            action_dim=max(config.action_dims),
            global_state_dim=self.global_state_dim if config.global_state_dim else None,
            use_global_state=config.global_state_dim is not None,
            store_log_probs=True,
            store_action_priors=config.prior_weight > 0,
            traj_len=config.ctm_traj_len if config.use_ctm_critic else None,
            store_trajectories=config.use_ctm_critic,
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
        agent_keys = keys[1:]  # shape (n_agents, 2)

        # Build dummy inputs that match what the networks expect at init time
        dummy_obs = jnp.zeros((1, self.obs_dims[0]))
        dummy_ci  = jnp.zeros((1, self.global_state_dim))
        dummy_ca  = jnp.zeros((1, self.total_action_dim))

        # Capture Python objects for the closure (not traced)
        actor           = self.actor
        critic          = self.critic
        ctm_critic      = self.ctm_critic
        use_ctm_critic  = self.use_ctm_critic
        actor_optimizer  = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        config           = self.config

        if use_ctm_critic:
            # CTM critic dummy inputs: (1, traj_len, n_agents, 2) etc.
            dummy_ctm_traj    = jnp.zeros((1, config.ctm_traj_len, config.n_agents, 2))
            dummy_ctm_obs     = jnp.zeros((1, config.n_agents, self.obs_dims[0]))
            dummy_ctm_actions = jnp.zeros((1, config.n_agents, self.action_dims[0]))

        def init_one_agent(k):
            key_a, key_c = random.split(k)
            ap = actor.init(key_a, dummy_obs)
            if use_ctm_critic:
                cp = ctm_critic.init(key_c, dummy_ctm_traj, dummy_ctm_obs, dummy_ctm_actions)
            else:
                cp = critic.init(key_c, dummy_ci, dummy_ca)
            aopt = actor_optimizer.init(ap)
            copt = critic_optimizer.init(cp)
            return DDPGAgentState(
                actor_params=ap,
                critic_params=cp,
                target_actor_params=ap,
                target_critic_params=cp,
                actor_opt_state=aopt,
                critic_opt_state=copt,
                noise_state=None,
                step=jnp.array(0, dtype=jnp.int32),
            )

        # vmap over agent keys -> stacked state with (n_agents, ...) leaves
        stacked_agent_states = jax.vmap(init_one_agent)(agent_keys)

        # Initialize buffer
        buffer_state = self.buffer.init()

        return MADDPGState(
            agent_states=stacked_agent_states,
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
        # observations: List of (obs_dim,) arrays -> stack to (n_agents, obs_dim)
        obs_stacked = jnp.stack(observations)  # (n_agents, obs_dim)

        actor = self.actor

        if explore:
            def select_one(k, actor_params, obs):
                noise = state.noise_scale * random.normal(k, (self.action_dims[0],))
                action = actor.apply(actor_params, obs[None], training=False)[0]
                return jnp.clip(action + noise, -1.0, 1.0)
            actions_stacked = jax.vmap(select_one)(
                keys, state.agent_states.actor_params, obs_stacked
            )
        else:
            def select_one(actor_params, obs):
                return actor.apply(actor_params, obs[None], training=False)[0]
            actions_stacked = jax.vmap(select_one)(
                state.agent_states.actor_params, obs_stacked
            )

        actions = [actions_stacked[i] for i in range(self.n_agents)]
        log_probs = [jnp.zeros((1,))] * self.n_agents
        # Gaussian noise is stateless — state is unchanged
        return actions, log_probs, state
    
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
        actor = self.actor
        obs_dim_0 = self.obs_dims[0]
        action_dim_0 = self.action_dims[0]

        if explore:
            # keys_flat: (n_envs * n_agents, 2) -> (n_envs, n_agents, 2)
            keys_flat = random.split(key, n_envs * self.n_agents).reshape(
                n_envs, self.n_agents, 2
            )

            def select_one_agent(actor_params, obs_all, keys_all):
                # obs_all: (n_envs, obs_dim), keys_all: (n_envs, 2)
                def select_single_env(k, obs):
                    noise = state.noise_scale * random.normal(k, (action_dim_0,))
                    action = actor.apply(actor_params, obs[None], training=False)[0]
                    return jnp.clip(action + noise, -1.0, 1.0)
                return jax.vmap(select_single_env)(keys_all, obs_all)

            # vmap over agents: actor_params axis 0, obs axis 1 (n_agents), keys axis 1
            actions_batch = jax.vmap(select_one_agent, in_axes=(0, 1, 1))(
                state.agent_states.actor_params,
                obs_batch[:, :, :obs_dim_0],
                keys_flat,
            )
            # actions_batch: (n_agents, n_envs, action_dim) -> (n_envs, n_agents, action_dim)
            actions_batch = actions_batch.transpose(1, 0, 2)
        else:
            def select_one_agent(actor_params, obs_all):
                # obs_all: (n_envs, obs_dim)
                return actor.apply(actor_params, obs_all, training=False)

            actions_batch = jax.vmap(select_one_agent, in_axes=(0, 1))(
                state.agent_states.actor_params,
                obs_batch[:, :, :obs_dim_0],
            )
            # actions_batch: (n_agents, n_envs, action_dim) -> (n_envs, n_agents, action_dim)
            actions_batch = actions_batch.transpose(1, 0, 2)

        # Gaussian noise is stateless — state is unchanged
        return actions_batch, state
    
    def store_transitions_batched(
        self,
        state: MADDPGState,
        obs_batch: jax.Array,
        actions_batch: jax.Array,
        rewards_batch: jax.Array,
        next_obs_batch: jax.Array,
        dones_batch: jax.Array,
        action_priors_batch: Optional[jax.Array] = None,
        trajectory_batch: Optional[jax.Array] = None,
        trajectory_idx_batch: Optional[jax.Array] = None,
        next_trajectory_batch: Optional[jax.Array] = None,
        next_trajectory_idx_batch: Optional[jax.Array] = None,
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
            trajectory_batch: Optional raw circular trajectory (n_envs, traj_len, n_agents, 2)
            trajectory_idx_batch: Optional circular buffer write indices (n_envs,)
            next_trajectory_batch: Optional raw next-step trajectory (n_envs, traj_len, n_agents, 2)
            next_trajectory_idx_batch: Optional next-step write indices (n_envs,)

        Returns:
            Updated MADDPGState
        """
        # Reorder trajectories from circular to chronological if provided (D7)
        if self.buffer.store_trajectories and trajectory_batch is not None:
            traj_len = trajectory_batch.shape[1]
            trajectory_ordered = jax.vmap(
                lambda traj, idx: _reorder_trajectory(traj, idx, traj_len)
            )(trajectory_batch, trajectory_idx_batch)
            next_trajectory_ordered = jax.vmap(
                lambda traj, idx: _reorder_trajectory(traj, idx, traj_len)
            )(next_trajectory_batch, next_trajectory_idx_batch)
        else:
            trajectory_ordered = None
            next_trajectory_ordered = None

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
            trajectory=trajectory_ordered,
            next_trajectory=next_trajectory_ordered,
        )

        # Add batch to buffer
        new_buffer_state = self.buffer.add_batch(state.buffer_state, transitions)

        return state.replace(buffer_state=new_buffer_state)

    def create_jit_update(self):
        """Create a JIT-compiled update function.
        
        This method creates a closure over self that can be JIT-compiled.
        Call this once after initialization and use the returned function
        for training to get maximum performance.
        
        Supports both standard DDPG and TD3 modes based on config.use_td3.
        
        Returns:
            jit_update: JIT-compiled update function with signature:
                (key, state) -> (new_state, info)
        """
        # Capture all necessary attributes in closure
        n_agents        = self.n_agents
        actor           = self.actor
        critic          = self.critic
        ctm_critic      = self.ctm_critic
        use_ctm_critic  = self.use_ctm_critic
        buffer          = self.buffer
        actor_optimizer  = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        config           = self.config
        obs_dims         = self.obs_dims
        action_dims      = self.action_dims
        obs_dim_0        = self.obs_dims[0]
        action_dim_0     = self.action_dims[0]
        use_td3          = self.use_td3
        policy_delay     = self.policy_delay
        target_noise     = self.target_noise
        target_noise_clip = self.target_noise_clip

        @jax.jit
        def jit_update(
            key: jax.Array,
            state: MADDPGState,
        ) -> Tuple[MADDPGState, Dict[str, Any]]:
            """JIT-compiled update for all agents."""

            # Check if we can update
            can_update = buffer.can_sample(state.buffer_state, config.batch_size)

            def do_update(carry):
                key, state = carry

                # Sample from buffer
                key, sample_key = random.split(key)
                batch = buffer.sample(state.buffer_state, sample_key, config.batch_size)

                batch_size = batch.obs.shape[0]
                obs_flat = batch.obs.reshape(batch_size, -1)
                next_obs_flat = batch.next_obs.reshape(batch_size, -1)
                actions_flat = batch.actions.reshape(batch_size, -1)

                global_states = obs_flat
                next_global_states = next_obs_flat

                # --- Target actions (vmapped over agents) ---
                target_keys = random.split(key, n_agents)  # (n_agents, 2)
                # next_obs: (batch, n_agents, obs_dim) -> (n_agents, batch, obs_dim)
                next_obs_per_agent = batch.next_obs[:, :, :obs_dim_0].transpose(1, 0, 2)

                if use_td3:
                    def compute_one_target(ag_state, agent_next_obs, tkey):
                        return select_target_action_with_smoothing(
                            tkey, actor, ag_state.target_actor_params, agent_next_obs,
                            target_noise=target_noise,
                            target_noise_clip=target_noise_clip,
                        )
                else:
                    def compute_one_target(ag_state, agent_next_obs, tkey):
                        return select_target_action(
                            actor, ag_state.target_actor_params, agent_next_obs
                        )

                # target_actions_stacked: (n_agents, batch, action_dim)
                target_actions_stacked = jax.vmap(
                    compute_one_target, in_axes=(0, 0, 0)
                )(state.agent_states, next_obs_per_agent, target_keys)
                # Flatten to (batch, n_agents * action_dim)
                target_actions = target_actions_stacked.transpose(1, 0, 2).reshape(batch_size, -1)

                # --- Critic update (vmapped over agents) ---
                # rewards / dones: (batch, n_agents) -> (n_agents, batch, 1)
                rewards_per_agent = batch.rewards.T[:, :, None]
                dones_per_agent   = batch.dones.T[:, :, None]

                if use_ctm_critic:
                    # --- CTM critic update path ---
                    # Target actions structured as (batch, n_agents, action_dim)
                    target_actions_structured = target_actions_stacked.transpose(1, 0, 2)

                    # Per-agent rewards/dones as (n_agents, batch) for vmap
                    rewards_1d = batch.rewards.T        # (n_agents, batch)
                    dones_1d   = batch.dones.T          # (n_agents, batch)

                    def update_one_critic_ctm(ag_state, rewards_i, dones_i):
                        return update_critic_ctm(
                            agent_state=ag_state,
                            ctm_critic=ctm_critic,
                            optimizer=critic_optimizer,
                            trajectory=batch.trajectory,
                            all_obs=batch.obs[:, :, :obs_dim_0],
                            all_actions=batch.actions[:, :, :action_dim_0],
                            next_trajectory=batch.next_trajectory,
                            next_all_obs=batch.next_obs[:, :, :obs_dim_0],
                            next_all_actions=target_actions_structured,
                            rewards=rewards_i,
                            dones=dones_i,
                            gamma=config.gamma,
                        )

                    new_agent_states, critic_infos = jax.vmap(
                        update_one_critic_ctm, in_axes=(0, 0, 0)
                    )(state.agent_states, rewards_1d, dones_1d)
                    total_critic_loss = jnp.mean(critic_infos['ctm_critic_loss'])

                else:
                    # --- MLP critic update path (TD3 or DDPG) ---
                    if use_td3:
                        def update_one_critic(ag_state, rewards_i, dones_i):
                            return update_critic_td3(
                                agent_state=ag_state,
                                critic=critic,
                                optimizer=critic_optimizer,
                                global_obs=global_states,
                                all_actions=actions_flat,
                                rewards=rewards_i,
                                next_global_obs=next_global_states,
                                next_all_actions=target_actions,
                                dones=dones_i,
                                gamma=config.gamma,
                            )
                    else:
                        def update_one_critic(ag_state, rewards_i, dones_i):
                            return update_critic(
                                agent_state=ag_state,
                                critic=critic,
                                actor=actor,
                                optimizer=critic_optimizer,
                                global_obs=global_states,
                                all_actions=actions_flat,
                                rewards=rewards_i,
                                next_global_obs=next_global_states,
                                next_all_actions=target_actions,
                                dones=dones_i,
                                gamma=config.gamma,
                            )

                    new_agent_states, critic_infos = jax.vmap(
                        update_one_critic, in_axes=(0, 0, 0)
                    )(state.agent_states, rewards_per_agent, dones_per_agent)
                    total_critic_loss = jnp.mean(critic_infos['critic_loss'])

                # --- Actor update (conditional on policy delay) ---
                should_update_actor = jnp.logical_or(
                    jnp.logical_not(use_td3 or use_ctm_critic),
                    (state.step % policy_delay) == 0
                )

                # obs: (batch, n_agents, obs_dim) -> (n_agents, batch, obs_dim)
                obs_per_agent = batch.obs[:, :, :obs_dim_0].transpose(1, 0, 2)
                agent_indices = jnp.arange(n_agents, dtype=jnp.int32)

                if config.prior_weight > 0 and batch.action_priors is not None:
                    priors_per_agent = batch.action_priors.transpose(1, 0, 2)
                else:
                    priors_per_agent = jnp.zeros((n_agents, batch_size, action_dim_0))

                if use_ctm_critic:
                    def update_one_actor(ag_state, obs_i, agent_idx, prior_i):
                        return update_actor_ctm(
                            agent_state=ag_state,
                            actor=actor,
                            ctm_critic=ctm_critic,
                            optimizer=actor_optimizer,
                            trajectory=batch.trajectory,
                            all_obs=batch.obs[:, :, :obs_dim_0],
                            agent_obs=obs_i,
                            all_actions_flat=actions_flat,
                            agent_action_idx=agent_idx,
                            action_dim=action_dim_0,
                            n_agents=n_agents,
                            action_prior=prior_i,
                            prior_weight=config.prior_weight,
                        )
                    dummy_actor_infos = {'ctm_actor_loss': jnp.zeros(n_agents)}
                elif use_td3:
                    def update_one_actor(ag_state, obs_i, agent_idx, prior_i):
                        return update_actor_td3(
                            agent_state=ag_state,
                            actor=actor,
                            critic=critic,
                            optimizer=actor_optimizer,
                            global_obs=global_states,
                            agent_obs=obs_i,
                            all_actions_flat=actions_flat,
                            agent_action_idx=agent_idx,
                            action_dim=action_dim_0,
                            action_prior=prior_i,
                            prior_weight=config.prior_weight,
                        )
                    dummy_actor_infos = {
                        'actor_loss': jnp.zeros(n_agents),
                        'actor_grad_norm': jnp.zeros(n_agents),
                        'policy_loss': jnp.zeros(n_agents),
                        'reg_loss': jnp.zeros(n_agents),
                        'q_value_mean': jnp.zeros(n_agents),
                        'action_mean': jnp.zeros(n_agents),
                        'action_std': jnp.zeros(n_agents),
                    }
                else:
                    def update_one_actor(ag_state, obs_i, agent_idx, prior_i):
                        return update_actor(
                            agent_state=ag_state,
                            actor=actor,
                            critic=critic,
                            optimizer=actor_optimizer,
                            global_obs=global_states,
                            agent_obs=obs_i,
                            all_actions_flat=actions_flat,
                            agent_action_idx=agent_idx,
                            action_dim=action_dim_0,
                            action_prior=prior_i,
                            prior_weight=config.prior_weight,
                        )
                    dummy_actor_infos = {
                        'actor_loss': jnp.zeros(n_agents),
                        'actor_grad_norm': jnp.zeros(n_agents),
                        'policy_loss': jnp.zeros(n_agents),
                        'reg_loss': jnp.zeros(n_agents),
                        'q_value_mean': jnp.zeros(n_agents),
                        'action_mean': jnp.zeros(n_agents),
                        'action_std': jnp.zeros(n_agents),
                    }

                def do_actor_updates(ag_states):
                    updated_states, infos = jax.vmap(
                        update_one_actor, in_axes=(0, 0, 0, 0)
                    )(ag_states, obs_per_agent, agent_indices, priors_per_agent)
                    return updated_states, infos

                def skip_actor_updates(ag_states):
                    return ag_states, dummy_actor_infos

                new_agent_states, actor_infos = jax.lax.cond(
                    should_update_actor,
                    do_actor_updates,
                    skip_actor_updates,
                    new_agent_states,
                )

                if use_ctm_critic:
                    total_actor_loss = jnp.mean(actor_infos['ctm_actor_loss'])
                else:
                    total_actor_loss = jnp.mean(actor_infos['actor_loss'])

                # --- Target network update (vmapped) ---
                new_agent_states = jax.vmap(update_targets, in_axes=(0, None))(
                    new_agent_states, config.tau
                )

                new_step = state.step + 1
                new_state = state.replace(
                    agent_states=new_agent_states,
                    step=new_step,
                )

                # Build extra_info carrying per-update metrics
                if use_ctm_critic:
                    extra_info = {
                        'ctm_q_mean':          jnp.mean(critic_infos['q_mean']),
                        'ctm_bellman_target':  jnp.mean(critic_infos['bellman_target_mean']),
                        'ctm_cert_score':      jnp.mean(critic_infos['cert_score_mean']),
                        'ctm_td_error':        jnp.mean(critic_infos['td_error_mean']),
                        'ctm_cert_aux_loss':   jnp.mean(critic_infos['cert_aux_loss']),
                        'ctm_tick_diversity':  jnp.mean(critic_infos['tick_diversity']),
                    }
                else:
                    extra_info = {
                        'ctm_q_mean':          jnp.array(0.0),
                        'ctm_bellman_target':  jnp.array(0.0),
                        'ctm_cert_score':      jnp.array(0.0),
                        'ctm_td_error':        jnp.array(0.0),
                        'ctm_cert_aux_loss':   jnp.array(0.0),
                        'ctm_tick_diversity':  jnp.array(0.0),
                    }

                return new_state, total_actor_loss, total_critic_loss, extra_info

            def skip_update(carry):
                _, state = carry
                empty_extra = {
                    'ctm_q_mean':          jnp.array(0.0),
                    'ctm_bellman_target':  jnp.array(0.0),
                    'ctm_cert_score':      jnp.array(0.0),
                    'ctm_td_error':        jnp.array(0.0),
                    'ctm_cert_aux_loss':   jnp.array(0.0),
                    'ctm_tick_diversity':  jnp.array(0.0),
                }
                return state, jnp.array(0.0), jnp.array(0.0), empty_extra

            new_state, actor_loss, critic_loss, extra_info = jax.lax.cond(
                can_update,
                do_update,
                skip_update,
                (key, state),
            )

            info = {
                'can_update':    can_update,
                'actor_loss':    actor_loss,
                'critic_loss':   critic_loss,
                'buffer_size':   state.buffer_state.size,
                **extra_info,
            }

            return new_state, info

        return jit_update
    
    def reset_noise(self, state: MADDPGState) -> MADDPGState:
        """Reset exploration noise for all agents.

        Gaussian noise is stateless, so there is nothing to reset.

        Args:
            state: Current state

        Returns:
            State unchanged
        """
        return state  # Gaussian noise has no persistent state
    
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
        # Index into the stacked state to get per-agent params
        agent_params = [
            get_agent_params(jax.tree_util.tree_map(lambda x: x[i], state.agent_states))
            for i in range(self.n_agents)
        ]

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
        # Load each agent's params into the stacked state by updating leaf-by-leaf
        new_stacked = state.agent_states
        for i, agent_params in enumerate(params['agent_params']):
            ag_state_i = jax.tree_util.tree_map(lambda x: x[i], new_stacked)
            ag_state_i = load_agent_params(ag_state_i, agent_params)
            new_stacked = jax.tree_util.tree_map(
                lambda stacked, updated: stacked.at[i].set(updated),
                new_stacked,
                ag_state_i,
            )

        return state.replace(
            agent_states=new_stacked,
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


