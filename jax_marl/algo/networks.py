"""Neural network architectures for MADDPG algorithm.

This module provides Flax-based neural networks for:
1. Actor (Policy) network - maps observations to actions
2. Critic (Q-function) network - maps (observation, action) to Q-value
3. ActorCritic - Combined actor-critic for shared feature extraction

Networks are designed to be:
- Fully compatible with JAX transformations (jit, vmap, grad)
- Stateless (parameters passed explicitly)
- Configurable (hidden dims, activation functions, normalization)

Architecture follows the original MADDPG paper with MLP networks,
with optional modern improvements (layer norm, dropout).
"""

from typing import Sequence, Callable, Any, Tuple, Optional, List, Dict
import math
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import random
import flax.linen as nn
from flax import struct
from functools import partial

# Add jaxCTM to path for SuperLinear import
_jaxctm_path = str(Path(__file__).parent.parent.parent / "jaxCTM")
if _jaxctm_path not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Default activation function
DEFAULT_ACTIVATION = nn.leaky_relu
DEFAULT_HIDDEN_DIMS = (64, 64, 64)


# ============================================================================
# Actor Network (Policy)
# ============================================================================

class Actor(nn.Module):
    """Actor network for continuous action spaces.
    
    Maps observations to continuous actions in [-1, 1] range.
    Uses tanh activation on output for bounded actions.
    
    Architecture:
        obs -> FC -> activation -> FC -> activation -> FC -> activation -> FC -> tanh
    
    Attributes:
        action_dim: Dimension of action space
        hidden_dims: Sequence of hidden layer dimensions
        activation: Activation function for hidden layers
        use_layer_norm: Whether to use layer normalization
        dropout_rate: Dropout rate (0 = no dropout)
    """
    action_dim: int
    hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS
    activation: Callable = DEFAULT_ACTIVATION
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, obs: jax.Array, training: bool = False) -> jax.Array:
        """Forward pass.
        
        Args:
            obs: Observation tensor, shape (..., obs_dim)
            training: Whether in training mode (for dropout)
            
        Returns:
            actions: Action tensor, shape (..., action_dim), in [-1, 1]
        """
        x = obs
        
        # Hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                hidden_dim,
                name=f"fc{i+1}",
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                bias_init=nn.initializers.constant(0.0),
            )(x)
            
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"ln{i+1}")(x)
            
            x = self.activation(x)
            
            if self.dropout_rate > 0 and training:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output layer with tanh for bounded actions
        x = nn.Dense(
            self.action_dim,
            name="fc_out",
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        
        return nn.tanh(x)


class ActorDiscrete(nn.Module):
    """Actor network for discrete action spaces.
    
    Maps observations to action logits (unnormalized probabilities).
    
    Attributes:
        n_actions: Number of discrete actions
        hidden_dims: Sequence of hidden layer dimensions
        activation: Activation function for hidden layers
        use_layer_norm: Whether to use layer normalization
        dropout_rate: Dropout rate (0 = no dropout)
    """
    n_actions: int
    hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS
    activation: Callable = DEFAULT_ACTIVATION
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, obs: jax.Array, training: bool = False) -> jax.Array:
        """Forward pass.
        
        Args:
            obs: Observation tensor, shape (..., obs_dim)
            training: Whether in training mode (for dropout)
            
        Returns:
            logits: Action logits, shape (..., n_actions)
        """
        x = obs
        
        # Hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                hidden_dim,
                name=f"fc{i+1}",
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                bias_init=nn.initializers.constant(0.0),
            )(x)
            
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"ln{i+1}")(x)
            
            x = self.activation(x)
            
            if self.dropout_rate > 0 and training:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output layer (no activation - raw logits)
        x = nn.Dense(
            self.n_actions,
            name="fc_out",
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        
        return x


# ============================================================================
# Critic Network (Q-function)
# ============================================================================

class Critic(nn.Module):
    """Critic network for Q-value estimation.
    
    Maps (observation, action) pairs to Q-values.
    In MADDPG, the critic takes observations and actions from ALL agents
    (centralized critic with decentralized actors).
    
    Architecture:
        [obs, action] -> FC -> activation -> FC -> activation -> FC -> activation -> FC -> Q
    
    Attributes:
        hidden_dims: Sequence of hidden layer dimensions
        activation: Activation function for hidden layers
        use_layer_norm: Whether to use layer normalization
        dropout_rate: Dropout rate (0 = no dropout)
    """
    hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS
    activation: Callable = DEFAULT_ACTIVATION
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(
        self,
        obs: jax.Array,
        action: jax.Array,
        training: bool = False,
    ) -> jax.Array:
        """Forward pass.
        
        Args:
            obs: Observation tensor, shape (..., obs_dim)
                 For MADDPG: can be global state or concatenated observations
            action: Action tensor, shape (..., action_dim)
                    For MADDPG: concatenated actions from all agents
            training: Whether in training mode (for dropout)
            
        Returns:
            q_value: Q-value estimate, shape (..., 1)
        """
        # Concatenate observation and action
        x = jnp.concatenate([obs, action], axis=-1)
        
        # Hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                hidden_dim,
                name=f"fc{i+1}",
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                bias_init=nn.initializers.constant(0.0),
            )(x)
            
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"ln{i+1}")(x)
            
            x = self.activation(x)
            
            if self.dropout_rate > 0 and training:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output layer (single Q-value)
        x = nn.Dense(
            1,
            name="fc_out",
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        
        return x


class CriticTwin(nn.Module):
    """Twin Critic network for TD3-style updates.
    
    Implements two independent critic networks to reduce overestimation bias.
    Returns both Q-values for min operation during training.
    
    Attributes:
        hidden_dims: Sequence of hidden layer dimensions
        activation: Activation function for hidden layers
        use_layer_norm: Whether to use layer normalization
        dropout_rate: Dropout rate (0 = no dropout)
    """
    hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS
    activation: Callable = DEFAULT_ACTIVATION
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    
    def setup(self):
        """Initialize twin critics."""
        self.critic1 = Critic(
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            name="critic1",
        )
        self.critic2 = Critic(
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            name="critic2",
        )
    
    def __call__(
        self,
        obs: jax.Array,
        action: jax.Array,
        training: bool = False,
    ) -> Tuple[jax.Array, jax.Array]:
        """Forward pass through both critics.
        
        Args:
            obs: Observation tensor
            action: Action tensor
            training: Whether in training mode
            
        Returns:
            q1, q2: Q-values from both critics
        """
        q1 = self.critic1(obs, action, training)
        q2 = self.critic2(obs, action, training)
        return q1, q2
    
    def q1(self, obs: jax.Array, action: jax.Array, training: bool = False) -> jax.Array:
        """Get Q-value from first critic only (for actor update)."""
        return self.critic1(obs, action, training)
    
    def q_min(self, obs: jax.Array, action: jax.Array, training: bool = False) -> jax.Array:
        """Get minimum Q-value from both critics (for conservative updates)."""
        q1, q2 = self(obs, action, training)
        return jnp.minimum(q1, q2)


# ============================================================================
# Network Initialization Utilities
# ============================================================================

def create_actor(
    key: jax.Array,
    obs_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int] = (64, 64, 64),
) -> Tuple[Actor, Any]:
    """Create and initialize an Actor network.
    
    Args:
        key: JAX random key
        obs_dim: Dimension of observation space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        
    Returns:
        actor: Actor module
        params: Initialized parameters
    """
    actor = Actor(action_dim=action_dim, hidden_dims=hidden_dims)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = actor.init(key, dummy_obs)
    return actor, params


def create_actor_discrete(
    key: jax.Array,
    obs_dim: int,
    n_actions: int,
    hidden_dims: Sequence[int] = (64, 64, 64),
) -> Tuple[ActorDiscrete, Any]:
    """Create and initialize a discrete Actor network.
    
    Args:
        key: JAX random key
        obs_dim: Dimension of observation space
        n_actions: Number of discrete actions
        hidden_dims: Hidden layer dimensions
        
    Returns:
        actor: ActorDiscrete module
        params: Initialized parameters
    """
    actor = ActorDiscrete(n_actions=n_actions, hidden_dims=hidden_dims)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = actor.init(key, dummy_obs)
    return actor, params


def create_critic(
    key: jax.Array,
    obs_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int] = (64, 64, 64),
) -> Tuple[Critic, Any]:
    """Create and initialize a Critic network.
    
    Args:
        key: JAX random key
        obs_dim: Dimension of observation/state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        
    Returns:
        critic: Critic module
        params: Initialized parameters
    """
    critic = Critic(hidden_dims=hidden_dims)
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_action = jnp.zeros((1, action_dim))
    params = critic.init(key, dummy_obs, dummy_action)
    return critic, params


def create_critic_twin(
    key: jax.Array,
    obs_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int] = (64, 64, 64),
) -> Tuple[CriticTwin, Any]:
    """Create and initialize a Twin Critic network.
    
    Args:
        key: JAX random key
        obs_dim: Dimension of observation/state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        
    Returns:
        critic: CriticTwin module
        params: Initialized parameters
    """
    critic = CriticTwin(hidden_dims=hidden_dims)
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_action = jnp.zeros((1, action_dim))
    params = critic.init(key, dummy_obs, dummy_action)
    return critic, params


# ============================================================================
# MADDPG-specific Network Utilities
# ============================================================================

def create_maddpg_networks(
    key: jax.Array,
    n_agents: int,
    obs_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int] = (64, 64, 64),
) -> Tuple[Actor, Critic, Any, Any]:
    """Create networks for a single MADDPG agent.
    
    In MADDPG:
    - Actor takes local observation -> local action
    - Critic takes global state (all obs) + all actions -> Q-value
    
    Args:
        key: JAX random key
        n_agents: Number of agents in environment
        obs_dim: Observation dimension per agent
        action_dim: Action dimension per agent
        hidden_dims: Hidden layer dimensions
        
    Returns:
        actor: Actor module
        critic: Critic module
        actor_params: Actor parameters
        critic_params: Critic parameters
    """
    key_actor, key_critic = random.split(key)
    
    # Actor: local obs -> local action
    actor, actor_params = create_actor(
        key_actor, obs_dim, action_dim, hidden_dims
    )
    
    # Critic: global state + all actions -> Q-value
    # Input dim = all observations + all actions
    critic_obs_dim = obs_dim * n_agents  # Or use global_state_dim
    critic_action_dim = action_dim * n_agents
    
    critic, critic_params = create_critic(
        key_critic, critic_obs_dim, critic_action_dim, hidden_dims
    )
    
    return actor, critic, actor_params, critic_params


def create_maddpg_networks_shared_critic(
    key: jax.Array,
    n_agents: int,
    obs_dim: int,
    action_dim: int,
    global_state_dim: int,
    hidden_dims: Sequence[int] = (64, 64, 64),
) -> Tuple[Actor, Critic, Any, Any]:
    """Create networks for MADDPG with explicit global state for critic.
    
    This version uses a separate global state dimension for the critic,
    which is common when the environment provides a global observation.
    
    Args:
        key: JAX random key
        n_agents: Number of agents
        obs_dim: Local observation dimension per agent
        action_dim: Action dimension per agent
        global_state_dim: Dimension of global state for critic
        hidden_dims: Hidden layer dimensions
        
    Returns:
        actor: Actor module
        critic: Critic module  
        actor_params: Actor parameters
        critic_params: Critic parameters
    """
    key_actor, key_critic = random.split(key)
    
    # Actor: local obs -> local action
    actor, actor_params = create_actor(
        key_actor, obs_dim, action_dim, hidden_dims
    )
    
    # Critic: global state + all actions -> Q-value
    critic_action_dim = action_dim * n_agents
    
    critic, critic_params = create_critic(
        key_critic, global_state_dim, critic_action_dim, hidden_dims
    )
    
    return actor, critic, actor_params, critic_params


# ============================================================================
# Multi-Agent Network Creation
# ============================================================================

@struct.dataclass
class MADDPGNetworks:
    """Container for all MADDPG agent networks.
    
    Stores actor and critic networks and parameters for all agents.
    
    Attributes:
        actors: List of Actor modules
        critics: List of Critic modules
        actor_params: List of actor parameter dicts
        critic_params: List of critic parameter dicts
        target_actor_params: List of target actor parameter dicts
        target_critic_params: List of target critic parameter dicts
    """
    actors: List[Actor]
    critics: List[Critic]
    actor_params: List[Any]
    critic_params: List[Any]
    target_actor_params: List[Any]
    target_critic_params: List[Any]


def create_all_agents_networks(
    key: jax.Array,
    n_agents: int,
    obs_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
    use_global_state: bool = False,
    global_state_dim: Optional[int] = None,
    use_layer_norm: bool = False,
) -> Tuple[List[Actor], List[Critic], List[Any], List[Any]]:
    """Create networks for all agents in MADDPG.
    
    Each agent gets its own actor network (decentralized execution).
    Each agent gets its own critic network (centralized training).
    
    Args:
        key: JAX random key
        n_agents: Number of agents
        obs_dim: Observation dimension per agent
        action_dim: Action dimension per agent
        hidden_dims: Hidden layer dimensions
        use_global_state: If True, critic uses global_state_dim
        global_state_dim: Global state dimension (if use_global_state)
        use_layer_norm: Whether to use layer normalization
        
    Returns:
        actors: List of Actor modules (one per agent)
        critics: List of Critic modules (one per agent)
        actor_params_list: List of actor parameters
        critic_params_list: List of critic parameters
    """
    # Calculate critic input dimensions
    if use_global_state and global_state_dim is not None:
        critic_obs_dim = global_state_dim
    else:
        critic_obs_dim = obs_dim * n_agents
    critic_action_dim = action_dim * n_agents
    
    actors = []
    critics = []
    actor_params_list = []
    critic_params_list = []
    
    for i in range(n_agents):
        key, key_actor, key_critic = random.split(key, 3)
        
        # Create actor for this agent
        actor = Actor(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
        )
        dummy_obs = jnp.zeros((1, obs_dim))
        actor_params = actor.init(key_actor, dummy_obs)
        
        # Create critic for this agent
        critic = Critic(
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
        )
        dummy_critic_obs = jnp.zeros((1, critic_obs_dim))
        dummy_critic_action = jnp.zeros((1, critic_action_dim))
        critic_params = critic.init(key_critic, dummy_critic_obs, dummy_critic_action)
        
        actors.append(actor)
        critics.append(critic)
        actor_params_list.append(actor_params)
        critic_params_list.append(critic_params)
    
    return actors, critics, actor_params_list, critic_params_list


def create_shared_critic_networks(
    key: jax.Array,
    n_agents: int,
    obs_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
    use_global_state: bool = False,
    global_state_dim: Optional[int] = None,
    use_layer_norm: bool = False,
) -> Tuple[List[Actor], Critic, List[Any], Any]:
    """Create networks with a single shared critic.
    
    Each agent gets its own actor, but they share one critic.
    More parameter efficient but may be less expressive.
    
    Args:
        key: JAX random key
        n_agents: Number of agents
        obs_dim: Observation dimension per agent
        action_dim: Action dimension per agent
        hidden_dims: Hidden layer dimensions
        use_global_state: If True, critic uses global_state_dim
        global_state_dim: Global state dimension (if use_global_state)
        use_layer_norm: Whether to use layer normalization
        
    Returns:
        actors: List of Actor modules (one per agent)
        critic: Shared Critic module
        actor_params_list: List of actor parameters
        critic_params: Shared critic parameters
    """
    # Calculate critic input dimensions
    if use_global_state and global_state_dim is not None:
        critic_obs_dim = global_state_dim
    else:
        critic_obs_dim = obs_dim * n_agents
    critic_action_dim = action_dim * n_agents
    
    actors = []
    actor_params_list = []
    
    # Create actor for each agent
    for i in range(n_agents):
        key, key_actor = random.split(key)
        
        actor = Actor(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
        )
        dummy_obs = jnp.zeros((1, obs_dim))
        actor_params = actor.init(key_actor, dummy_obs)
        
        actors.append(actor)
        actor_params_list.append(actor_params)
    
    # Create single shared critic
    key, key_critic = random.split(key)
    critic = Critic(
        hidden_dims=hidden_dims,
        use_layer_norm=use_layer_norm,
    )
    dummy_critic_obs = jnp.zeros((1, critic_obs_dim))
    dummy_critic_action = jnp.zeros((1, critic_action_dim))
    critic_params = critic.init(key_critic, dummy_critic_obs, dummy_critic_action)
    
    return actors, critic, actor_params_list, critic_params


# ============================================================================
# Combined Actor-Critic Network
# ============================================================================

class ActorCritic(nn.Module):
    """Combined Actor-Critic network with shared feature extraction.
    
    Useful for PPO-style algorithms where actor and critic share
    early layers for more efficient learning.
    
    Attributes:
        action_dim: Dimension of action space
        hidden_dims: Shared hidden layer dimensions
        activation: Activation function
        use_layer_norm: Whether to use layer normalization
    """
    action_dim: int
    hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS
    activation: Callable = DEFAULT_ACTIVATION
    use_layer_norm: bool = False
    
    @nn.compact
    def __call__(
        self, 
        obs: jax.Array,
        training: bool = False,
    ) -> Tuple[jax.Array, jax.Array]:
        """Forward pass.
        
        Args:
            obs: Observation tensor
            training: Whether in training mode
            
        Returns:
            actions: Action tensor (tanh bounded to [-1, 1])
            value: State value estimate
        """
        x = obs
        
        # Shared layers
        for i, hidden_dim in enumerate(self.hidden_dims[:-1]):
            x = nn.Dense(
                hidden_dim,
                name=f"shared_fc{i+1}",
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"shared_ln{i+1}")(x)
            x = self.activation(x)
        
        # Actor head
        actor_x = nn.Dense(
            self.hidden_dims[-1],
            name="actor_fc",
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
        )(x)
        actor_x = self.activation(actor_x)
        actions = nn.Dense(
            self.action_dim,
            name="actor_out",
            kernel_init=nn.initializers.orthogonal(0.01),
        )(actor_x)
        actions = nn.tanh(actions)
        
        # Critic head
        critic_x = nn.Dense(
            self.hidden_dims[-1],
            name="critic_fc",
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
        )(x)
        critic_x = self.activation(critic_x)
        value = nn.Dense(
            1,
            name="critic_out",
            kernel_init=nn.initializers.orthogonal(1.0),
        )(critic_x)
        
        return actions, value


# ============================================================================
# Parameter Counting Utility
# ============================================================================

def count_parameters(params: Any) -> int:
    """Count total number of parameters in a parameter tree.
    
    Args:
        params: Flax parameter tree
        
    Returns:
        Total number of parameters
    """
    return sum(p.size for p in jax.tree.leaves(params))


def print_network_summary(
    name: str,
    params: Any,
) -> None:
    """Print a summary of network parameters.
    
    Args:
        name: Name of the network
        params: Flax parameter tree
    """
    total_params = count_parameters(params)
    print(f"\n{name} Network Summary:")
    print(f"  Total parameters: {total_params:,}")
    
    # Print layer-wise breakdown
    def print_layer(prefix: str, p: Any, depth: int = 0):
        indent = "  " * (depth + 1)
        if isinstance(p, dict):
            for k, v in p.items():
                print_layer(f"{prefix}/{k}" if prefix else k, v, depth)
        else:
            print(f"{indent}{prefix}: {p.shape} ({p.size:,} params)")
    
    print_layer("", params['params'])


def get_activation_fn(name: str) -> Callable:
    """Get activation function by name.

    Args:
        name: Activation name ('relu', 'leaky_relu', 'tanh', 'elu', 'gelu')

    Returns:
        Activation function
    """
    activations = {
        'relu': nn.relu,
        'leaky_relu': nn.leaky_relu,
        'tanh': nn.tanh,
        'elu': nn.elu,
        'gelu': nn.gelu,
        'silu': nn.silu,
        'swish': nn.swish,
    }
    return activations.get(name, nn.leaky_relu)


# ============================================================================
# CTM Temporal Critic
# ============================================================================

class TrajectoryEncoder(nn.Module):
    """Encode trajectory + current obs/actions into kv tokens for CTMCritic.

    Produces hybrid tokens (D4):
      - traj_len temporal tokens  — one per step, joint over all agents (pos + approx-vel)
      - n_agents per-agent tokens — one per agent, current (obs, action) pair

    Total kv sequence length: traj_len + n_agents

    Inputs:
        trajectory:  (B, traj_len, n_agents, 2)  float16 or float32 position history
        all_obs:     (B, n_agents, obs_dim)       current observations
        all_actions: (B, n_agents, action_dim)    actions being evaluated

    Outputs:
        kv:              (B, traj_len + n_agents, d_input)  LayerNorm'd kv tokens
        per_agent_feat:  (B, n_agents, d_model)             per-agent features for Option-3 upgrade
    """
    d_input: int    # attention embedding dimension (128)
    d_model: int    # CTM neuron count; used for per_agent_feat (Option 3 ready)
    n_agents: int
    traj_len: int
    vel_max: float = 0.8

    @nn.compact
    def __call__(
        self,
        trajectory: jax.Array,
        all_obs: jax.Array,
        all_actions: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        B = trajectory.shape[0]

        # Cast to float32 — trajectory stored as float16 in buffer (D6)
        pos = trajectory.astype(jnp.float32)  # (B, traj_len, n_agents, 2)

        # --- Temporal branch ---
        # Finite-difference velocity (D5): vel[0]=0, vel[t]=(p[t]-p[t-1])/vel_max
        vel = jnp.concatenate([
            jnp.zeros_like(pos[:, :1]),                               # (B, 1, n_agents, 2)
            (pos[:, 1:] - pos[:, :-1]) / self.vel_max,               # (B, traj_len-1, n_agents, 2)
        ], axis=1)  # (B, traj_len, n_agents, 2)

        # Flatten agents into feature dim: (B, traj_len, n_agents*4)
        joint = jnp.concatenate([pos, vel], axis=-1)                 # (B, traj_len, n_agents, 4)
        joint = joint.reshape(B, self.traj_len, self.n_agents * 4)

        traj_kv = nn.Dense(self.d_input, name='traj_proj')(joint)    # (B, traj_len, d_input)

        # --- Per-agent branch ---
        per_agent = jnp.concatenate(
            [all_obs, all_actions.astype(jnp.float32)], axis=-1
        )  # (B, n_agents, obs_dim + action_dim)

        agent_kv = nn.Dense(self.d_input, name='agent_proj')(per_agent)   # (B, n_agents, d_input)

        # Per-agent features for Option-3 credit-assignment upgrade (not wired here)
        per_agent_feat = nn.Dense(self.d_model, name='agent_feat')(per_agent)  # (B, n_agents, d_model)

        # Combine and normalise (D4)
        kv = jnp.concatenate([traj_kv, agent_kv], axis=1)            # (B, traj_len+n_agents, d_input)
        kv = nn.LayerNorm(name='kv_ln')(kv)

        return kv, per_agent_feat


class CTMCritic(nn.Module):
    """CTM-based temporal critic for MADDPG (Option 2, D1–D10).

    Replaces the MLP critic with a Continuous Thought Machine that
    cross-attends over a hybrid trajectory-token sequence at each internal tick.

    Architecture mirrors ContinuousThoughtMachine (jaxCTM/model.py) but:
      - TrajectoryEncoder replaces the CNN backbone (no BatchNorm, no CNN weights)
      - Scalar Q head  +  hybrid certainty head  replace the classification projector
      - out_dims is always 1 (scalar Q)

    Inputs:
        trajectory:  (B, traj_len, n_agents, 2)  position history
        all_obs:     (B, n_agents, obs_dim)       current observations
        all_actions: (B, n_agents, action_dim)    actions being evaluated
        alpha:       float in [0, 1]              certainty blend (1=tick-var, 0=learned)
        deterministic: bool                       True disables dropout

    Outputs:
        q_values:    (B, 1, iterations)  Q-value at each internal tick
        certainties: (B, 2, iterations)  [uncertainty, certainty] at each tick
            certainties[:, 1, :] is the certainty score used for tick selection
    """
    # Architecture (D10)
    iterations: int        # 8
    d_model: int           # 64
    d_input: int           # 128
    memory_length: int     # 5
    heads: int             # 2
    n_synch_out: int       # 8
    n_synch_action: int    # 8
    memory_hidden_dims: int  # 8

    # Trajectory config
    n_agents: int
    traj_len: int

    vel_max: float = 0.8
    dropout_rate: float = 0.0

    def setup(self):
        from jaxCTM.layers import SuperLinear

        d = self.d_input

        # --- Trajectory encoder (replaces CNN backbone, D1) ---
        self.traj_encoder = TrajectoryEncoder(
            d_input=d,
            d_model=self.d_model,
            n_agents=self.n_agents,
            traj_len=self.traj_len,
            vel_max=self.vel_max,
        )

        # --- Attention query projection (from synch_action) ---
        self.q_proj_linear = nn.Dense(features=d)

        # --- Multi-head attention internal projections ---
        assert d % self.heads == 0, "d_input must be divisible by heads"
        self.head_dim = d // self.heads
        self.attn_q_proj   = nn.Dense(features=d)
        self.attn_k_proj   = nn.Dense(features=d)
        self.attn_v_proj   = nn.Dense(features=d)
        self.attn_out_proj = nn.Dense(features=d)

        # --- Synapse layer (state update at each tick) ---
        self.synapse_drop   = nn.Dropout(rate=self.dropout_rate)
        self.synapse_linear = nn.Dense(features=self.d_model * 2)
        self.synapse_ln     = nn.LayerNorm()

        # --- Trace processor (neuron-level models over activation history) ---
        self.trace_sl1 = SuperLinear(
            out_dims=2 * self.memory_hidden_dims,
            N=self.d_model,
            dropout_rate=self.dropout_rate,
        )
        self.trace_sl2 = SuperLinear(
            out_dims=2,
            N=self.d_model,
            dropout_rate=self.dropout_rate,
        )

        # --- Q output head ---
        self.q_projector = nn.Dense(features=1)

        # --- Learned certainty head (D2) ---
        # Bias initialised to +2.0 → sigmoid(2.0) ≈ 0.88 (starts fairly confident)
        self.cert_projector = nn.Dense(
            features=1,
            bias_init=nn.initializers.constant(2.0),
        )

        # --- Synchronization sizes ---
        self.synch_repr_size_action = (self.n_synch_action * (self.n_synch_action + 1)) // 2
        self.synch_repr_size_out    = (self.n_synch_out    * (self.n_synch_out    + 1)) // 2

        # Upper-triangular indices (static, computed once at setup)
        self.triu_idx_action = jnp.triu_indices(self.n_synch_action)
        self.triu_idx_out    = jnp.triu_indices(self.n_synch_out)

        # --- Learned initial recurrent state ---
        scale_state = 1.0 / math.sqrt(self.d_model)
        scale_trace = 1.0 / math.sqrt(self.d_model + self.memory_length)

        self.start_activated_state = self.param(
            'start_activated_state',
            lambda rng, shape: jrandom.uniform(rng, shape, minval=-scale_state, maxval=scale_state),
            (self.d_model,),
        )
        self.start_trace = self.param(
            'start_trace',
            lambda rng, shape: jrandom.uniform(rng, shape, minval=-scale_trace, maxval=scale_trace),
            (self.d_model, self.memory_length),
        )

        # --- Learned decay parameters (D8) ---
        # Clamped to [0, 15]; r = exp(-decay) ∈ [exp(-15), 1]
        self.decay_params_action = self.param(
            'decay_params_action',
            nn.initializers.zeros_init(),
            (self.synch_repr_size_action,),
        )
        self.decay_params_out = self.param(
            'decay_params_out',
            nn.initializers.zeros_init(),
            (self.synch_repr_size_out,),
        )

    # ------------------------------------------------------------------
    # Internal helpers (mirrors ContinuousThoughtMachine)
    # ------------------------------------------------------------------

    def _compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        """Pairwise outer-product EMA synchronization."""
        if synch_type == 'action':
            n   = self.n_synch_action
            selected = activated_state[:, -n:]
            idx = self.triu_idx_action
        else:
            n   = self.n_synch_out
            selected = activated_state[:, :n]
            idx = self.triu_idx_out

        outer    = selected[:, :, None] * selected[:, None, :]  # (B, n, n)
        pairwise = outer[:, idx[0], idx[1]]                     # (B, synch_size)

        if decay_alpha is None:
            decay_alpha = pairwise
            decay_beta  = jnp.ones_like(pairwise)
        else:
            decay_alpha = r * decay_alpha + pairwise
            decay_beta  = r * decay_beta  + 1.0

        return decay_alpha / jnp.sqrt(decay_beta), decay_alpha, decay_beta

    def _multi_head_attention(self, q, kv):
        """Scaled dot-product multi-head cross-attention."""
        B, S, _ = kv.shape
        H = self.heads
        D = self.head_dim

        q_proj = self.attn_q_proj(q)   # (B, 1, d_input)
        k_proj = self.attn_k_proj(kv)  # (B, S, d_input)
        v_proj = self.attn_v_proj(kv)  # (B, S, d_input)

        q_heads = q_proj.reshape(B, 1, H, D).transpose(0, 2, 1, 3)  # (B, H, 1, D)
        k_heads = k_proj.reshape(B, S, H, D).transpose(0, 2, 1, 3)  # (B, H, S, D)
        v_heads = v_proj.reshape(B, S, H, D).transpose(0, 2, 1, 3)  # (B, H, S, D)

        scale       = jnp.sqrt(jnp.array(D, dtype=q_proj.dtype))
        attn_logits = jnp.einsum('bhqd,bhkd->bhqk', q_heads, k_heads) / scale
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v_heads)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, 1, self.d_input)
        return self.attn_out_proj(attn_out)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        trajectory: jax.Array,
        all_obs: jax.Array,
        all_actions: jax.Array,
        alpha: float = 0.0,
        deterministic: bool = True,
    ) -> Tuple[jax.Array, jax.Array]:
        """Forward pass through the CTM temporal critic.

        Args:
            trajectory:   (B, traj_len, n_agents, 2)  chronological position history
            all_obs:      (B, n_agents, obs_dim)       current observations
            all_actions:  (B, n_agents, action_dim)    actions being evaluated
            alpha:        certainty blend in [0, 1] — 1 = tick-variance only, 0 = learned only
            deterministic: True disables dropout

        Returns:
            q_values:    (B, 1, iterations)  Q-value at each tick
            certainties: (B, 2, iterations)  [:, 0, :] = uncertainty,  [:, 1, :] = certainty
        """
        B = trajectory.shape[0]

        # D8: clamp decay params at forward entry
        decay_params_action = jnp.clip(self.decay_params_action, 0.0, 15.0)
        decay_params_out    = jnp.clip(self.decay_params_out,    0.0, 15.0)

        # --- Encode trajectory into kv tokens (replaces CNN, D1 / D4) ---
        kv, _ = self.traj_encoder(trajectory, all_obs, all_actions)

        # --- Initialise recurrent state ---
        state_trace = jnp.broadcast_to(
            self.start_trace[None, :, :], (B, self.d_model, self.memory_length)
        )
        activated_state = jnp.broadcast_to(
            self.start_activated_state[None, :], (B, self.d_model)
        )

        r_action = jnp.exp(-decay_params_action)[None, :]  # (1, synch_size_action)
        r_out    = jnp.exp(-decay_params_out)[None, :]     # (1, synch_size_out)

        # Seed output EMA with initial state
        _, decay_alpha_out, decay_beta_out = self._compute_synchronisation(
            activated_state, None, None, r_out, 'out'
        )
        decay_alpha_action = None
        decay_beta_action  = None

        all_q_t      = []  # list of (B, 1) Q-values per tick
        all_synch_t  = []  # list of (B, synch_size_out) for certainty projection

        # --- Recurrent loop (unrolled by XLA) ---
        for _ in range(self.iterations):

            # 1. Action synch → attention query
            synch_action, decay_alpha_action, decay_beta_action = self._compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action, r_action, 'action'
            )

            # 2. Cross-attention over trajectory tokens
            q_query  = self.q_proj_linear(synch_action)[:, None, :]  # (B, 1, d_input)
            attn_out = self._multi_head_attention(q_query, kv).squeeze(1)  # (B, d_input)

            # 3. Synapse — GLU state update
            pre = jnp.concatenate([attn_out, activated_state], axis=-1)
            s   = self.synapse_drop(pre, deterministic=deterministic)
            s   = self.synapse_linear(s)
            a, b = jnp.split(s, 2, axis=-1)
            s   = a * jax.nn.sigmoid(b)
            s   = self.synapse_ln(s)

            # 4. Slide trace window (drop oldest, append current pre-activation)
            state_trace = jnp.concatenate(
                [state_trace[:, :, 1:], s[:, :, None]], axis=-1
            )

            # 5. Trace processor — neuron-level models (GLU × 2)
            tp = self.trace_sl1(state_trace, deterministic=deterministic)
            tp_a, tp_b = jnp.split(tp, 2, axis=-1)
            tp = tp_a * jax.nn.sigmoid(tp_b)
            tp = self.trace_sl2(tp, deterministic=deterministic)
            tp_a2, tp_b2 = jnp.split(tp, 2, axis=-1)
            activated_state = (tp_a2 * jax.nn.sigmoid(tp_b2)).squeeze(-1)

            # 6. Output synch
            synch_out, decay_alpha_out, decay_beta_out = self._compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, 'out'
            )

            # 7. Q-value head
            q_t = self.q_projector(synch_out)  # (B, 1)
            all_q_t.append(q_t)
            all_synch_t.append(synch_out)      # (B, synch_size_out)

        # Stack across ticks
        q_values = jnp.stack(all_q_t, axis=-1)    # (B, 1, T)
        q_flat   = q_values[:, 0, :]               # (B, T)

        # --- Hybrid certainty (D2) ---
        cert_list = []
        for t in range(self.iterations):
            # Learned component: sigmoid of projector applied to synch_out
            cert_raw = jax.nn.sigmoid(
                self.cert_projector(all_synch_t[t])
            ).squeeze(-1)  # (B,)

            # Structural component: tick-variance (parameter-free, calibrated from step 0)
            if t == 0:
                tick_var = jnp.zeros_like(cert_raw)
            else:
                q_so_far = q_flat[:, :t + 1]           # (B, t+1)
                var      = jnp.var(q_so_far, axis=-1)  # (B,)
                tick_var = var / (1.0 + var)            # ∈ [0, 1)

            # Blend: alpha=1 → tick-variance; alpha=0 → learned (D2)
            cert_t = cert_raw * (1.0 - alpha) + (1.0 - tick_var) * alpha  # (B,)

            cert_list.append(jnp.stack([1.0 - cert_t, cert_t], axis=-1))  # (B, 2)

        certainties = jnp.stack(cert_list, axis=-1)  # (B, 2, T)

        return q_values, certainties


def create_ctm_critic(
    key: jax.Array,
    n_agents: int,
    obs_dim: int,
    action_dim: int,
    traj_len: int,
    iterations: int = 8,
    d_model: int = 64,
    d_input: int = 128,
    memory_length: int = 5,
    heads: int = 2,
    n_synch_out: int = 8,
    n_synch_action: int = 8,
    memory_hidden_dims: int = 8,
) -> Tuple['CTMCritic', Any]:
    """Create and initialise a CTMCritic.

    Returns:
        critic: CTMCritic module
        params: Initialised parameters
    """
    critic = CTMCritic(
        iterations=iterations,
        d_model=d_model,
        d_input=d_input,
        memory_length=memory_length,
        heads=heads,
        n_synch_out=n_synch_out,
        n_synch_action=n_synch_action,
        memory_hidden_dims=memory_hidden_dims,
        n_agents=n_agents,
        traj_len=traj_len,
    )
    dummy_traj    = jnp.zeros((1, traj_len, n_agents, 2))
    dummy_obs     = jnp.zeros((1, n_agents, obs_dim))
    dummy_actions = jnp.zeros((1, n_agents, action_dim))
    params = critic.init(key, dummy_traj, dummy_obs, dummy_actions)
    return critic, params
