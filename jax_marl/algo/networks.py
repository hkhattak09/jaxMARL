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
    """CTM-based temporal critic for MADDPG, aligned with the paper's RL approach.

    Follows ContinuousThoughtMachineRL (ctm_rl.py) adapted for MADDPG:
      - TrajectoryEncoder replaces the CNN backbone; mean-pooled tokens give a
        single feature vector that is concatenated to the activated state each tick
        (paper: backbone features concat'd to activated_state — heads=0, no attention)
      - Two synapse GLU+LN blocks (paper's RL synapse_depth=1)
      - Sliding-window post-activation synchronisation (paper's RL compute_synchronisation)
      - No action synch (n_synch_action=0), no certainty — per paper's RL findings
      - State re-initialised each forward call; TrajectoryEncoder supplies the
        temporal context that persistent hidden state would provide in PPO

    Inputs:
        trajectory:    (B, traj_len, n_agents, 2)
        all_obs:       (B, n_agents, obs_dim)
        all_actions:   (B, n_agents, action_dim)
        deterministic: True disables dropout

    Outputs:
        q_values: (B, 1, iterations) — use [:, 0, -1] for the final-tick Q estimate
    """
    iterations: int
    d_model: int
    d_input: int
    memory_length: int
    n_synch_out: int
    memory_hidden_dims: int

    n_agents: int
    traj_len: int

    vel_max: float = 0.8
    dropout_rate: float = 0.0

    def setup(self):
        from jaxCTM.layers import SuperLinear

        d = self.d_input

        # --- Trajectory encoder (replaces CNN backbone) ---
        self.traj_encoder = TrajectoryEncoder(
            d_input=d,
            d_model=self.d_model,
            n_agents=self.n_agents,
            traj_len=self.traj_len,
            vel_max=self.vel_max,
        )

        # --- Synapse: two GLU+LN blocks (paper's RL synapse_depth=1) ---
        # Block 1 input: concat(features, activated_state) = d_input + d_model
        self.synapse_drop1  = nn.Dropout(rate=self.dropout_rate)
        self.synapse_proj1  = nn.Dense(features=self.d_model * 2)
        self.synapse_ln1    = nn.LayerNorm()
        # Block 2 input: d_model (output of block 1)
        self.synapse_drop2  = nn.Dropout(rate=self.dropout_rate)
        self.synapse_proj2  = nn.Dense(features=self.d_model * 2)
        self.synapse_ln2    = nn.LayerNorm()

        # --- Trace processor (neuron-level models over pre-activation history) ---
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

        # --- Output synchronisation (no action synch) ---
        self.synch_repr_size_out = (self.n_synch_out * (self.n_synch_out + 1)) // 2
        self.triu_idx_out = jnp.triu_indices(self.n_synch_out)

        # --- Learned initial recurrent state ---
        scale_trace = 1.0 / math.sqrt(self.d_model + self.memory_length)

        self.start_trace = self.param(
            'start_trace',
            lambda rng, shape: jrandom.uniform(rng, shape, minval=-scale_trace, maxval=scale_trace),
            (self.d_model, self.memory_length),
        )
        self.start_activated_trace = self.param(
            'start_activated_trace',
            lambda rng, shape: jrandom.uniform(rng, shape, minval=-scale_trace, maxval=scale_trace),
            (self.d_model, self.memory_length),
        )

        # --- Decay parameters for output synchronisation ---
        # Clamped to [0, 4] per paper's RL clamp_lims
        self.decay_params_out = self.param(
            'decay_params_out',
            nn.initializers.zeros_init(),
            (self.synch_repr_size_out,),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_synchronisation(self, activated_state_trace):
        """Sliding-window pairwise synchronisation over post-activation history.

        Mirrors ContinuousThoughtMachineRL.compute_synchronisation():
            synch = sum_m(decay_m * outer_m) / sqrt(sum_m decay_m)

        where outer_m is the upper-triangular pairwise product of the n_synch_out
        selected neurons at history step m, and decay_m decays exponentially from
        the most recent step backward.

        Args:
            activated_state_trace: (B, d_model, M)
        Returns:
            synch: (B, synch_repr_size_out)
        """
        M = activated_state_trace.shape[-1]
        n = self.n_synch_out
        decay_params = jnp.clip(self.decay_params_out, 0.0, 4.0)  # (synch_size,)

        # Decay weight per position: newest step (M-1) → weight=1, oldest (0) → most decayed
        positions = jnp.arange(M, dtype=jnp.float32)
        reversed_pos = (M - 1 - positions)                                  # [M-1, …, 0]
        decay = jnp.exp(-decay_params[:, None] * reversed_pos[None, :])    # (synch_size, M)

        # Select last n neurons (paper's first-last neuron_select_type → last n)
        selected = activated_state_trace[:, -n:, :]    # (B, n, M)
        S_T = selected.transpose(0, 2, 1)              # (B, M, n)

        outer    = S_T[:, :, :, None] * S_T[:, :, None, :]          # (B, M, n, n)
        triu_vals = outer[:, :, self.triu_idx_out[0], self.triu_idx_out[1]]  # (B, M, synch_size)

        triu_t = triu_vals.transpose(0, 2, 1)                        # (B, synch_size, M)
        synch  = (decay[None, :, :] * triu_t).sum(-1)                # (B, synch_size)
        denom  = jnp.sqrt(decay.sum(-1) + 1e-8)[None, :]             # (1, synch_size)
        return synch / denom

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        trajectory: jax.Array,
        all_obs: jax.Array,
        all_actions: jax.Array,
        deterministic: bool = True,
    ) -> jax.Array:
        """
        Returns:
            q_values: (B, 1, iterations) — use [:, 0, -1] for the final-tick Q estimate
        """
        B = trajectory.shape[0]

        # --- Encode trajectory into a single feature vector (replaces CNN backbone) ---
        # Mean-pool the token sequence: (B, S, d_input) → (B, d_input)
        kv, _ = self.traj_encoder(trajectory, all_obs, all_actions)
        features = kv.mean(axis=1)  # (B, d_input)

        # --- Initialise recurrent state ---
        state_trace = jnp.broadcast_to(
            self.start_trace[None, :, :], (B, self.d_model, self.memory_length)
        )
        activated_state_trace = jnp.broadcast_to(
            self.start_activated_trace[None, :, :], (B, self.d_model, self.memory_length)
        )

        # --- Recurrent loop via lax.scan ---
        def _tick_body(carry, _):
            state_trace, activated_state_trace = carry

            # Current activated state = most recent entry in post-activation history
            activated_state = activated_state_trace[:, :, -1]  # (B, d_model)

            # 1. Synapse input: concat(backbone_features, activated_state)
            #    Paper: pre_synapse_input = cat(features, activated_state_trace[:,:,-1])
            pre = jnp.concatenate([features, activated_state], axis=-1)  # (B, d_input + d_model)

            # 2. Synapse block 1 — GLU + LayerNorm
            s = self.synapse_drop1(pre, deterministic=deterministic)
            s = self.synapse_proj1(s)                          # (B, d_model * 2)
            s_a, s_b = jnp.split(s, 2, axis=-1)
            s = s_a * jax.nn.sigmoid(s_b)                     # GLU → (B, d_model)
            s = self.synapse_ln1(s)

            # 3. Synapse block 2 — GLU + LayerNorm
            s = self.synapse_drop2(s, deterministic=deterministic)
            s = self.synapse_proj2(s)                          # (B, d_model * 2)
            s_a, s_b = jnp.split(s, 2, axis=-1)
            s = s_a * jax.nn.sigmoid(s_b)                     # GLU → (B, d_model)
            s = self.synapse_ln2(s)

            # 4. Slide pre-activation trace (drop oldest, append current)
            state_trace = jnp.concatenate(
                [state_trace[:, :, 1:], s[:, :, None]], axis=-1
            )  # (B, d_model, M)

            # 5. Trace processor — neuron-level models (GLU × 2)
            tp = self.trace_sl1(state_trace, deterministic=deterministic)
            tp_a, tp_b = jnp.split(tp, 2, axis=-1)
            tp = tp_a * jax.nn.sigmoid(tp_b)
            tp = self.trace_sl2(tp, deterministic=deterministic)
            tp_a2, tp_b2 = jnp.split(tp, 2, axis=-1)
            new_activated = (tp_a2 * jax.nn.sigmoid(tp_b2)).squeeze(-1)  # (B, d_model)

            # 6. Slide post-activation trace
            activated_state_trace = jnp.concatenate(
                [activated_state_trace[:, :, 1:], new_activated[:, :, None]], axis=-1
            )  # (B, d_model, M)

            # 7. Output synchronisation from post-activation history (paper's RL synch)
            synch_out = self._compute_synchronisation(activated_state_trace)  # (B, synch_size)

            # 8. Q-value head
            q_t = self.q_projector(synch_out)  # (B, 1)

            return (state_trace, activated_state_trace), q_t

        init_carry = (state_trace, activated_state_trace)

        # Prime sub-modules during init so Flax registers params before lax.scan traces
        # the body. During apply() is_mutable_collection('params') is False — no overhead.
        if self.is_mutable_collection('params'):
            _tick_body(init_carry, None)

        _, q_ticks = jax.lax.scan(
            _tick_body, init_carry, xs=None, length=self.iterations
        )
        # q_ticks: (T, B, 1) → (B, 1, T)
        return jnp.moveaxis(q_ticks, 0, -1)


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
