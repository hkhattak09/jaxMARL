"""Utility functions for MADDPG algorithm.

This module provides common utility functions for:
1. Target network updates (soft and hard)
2. Gradient operations
3. Action processing (gumbel softmax, one-hot)
4. RL-specific utilities (advantage normalization, reward scaling)

All functions are JAX-compatible and JIT-compilable.
"""

from typing import Any, Dict, Tuple, Optional, Union
import jax
import jax.numpy as jnp
from jax import random
from functools import partial


# Type alias for PyTree of parameters
Params = Dict[str, Any]


def soft_update(
    target_params: Params,
    online_params: Params,
    tau: float,
) -> Params:
    """Soft update of target network parameters.
    
    Performs exponential moving average update:
        target = target * (1 - tau) + online * tau
    
    This is the standard DDPG/MADDPG target network update.
    
    Args:
        target_params: Current target network parameters
        online_params: Current online (trained) network parameters
        tau: Interpolation factor (0 < tau < 1), typically 0.01 or 0.005
        
    Returns:
        Updated target parameters
        
    Example:
        >>> target_params = soft_update(target_params, online_params, tau=0.01)
    """
    return jax.tree.map(
        lambda t, o: t * (1.0 - tau) + o * tau,
        target_params,
        online_params,
    )


def hard_update(
    target_params: Params,
    online_params: Params,
) -> Params:
    """Hard update (copy) of target network parameters.
    
    Directly copies online parameters to target:
        target = online
    
    Used for initialization of target networks.
    
    Args:
        target_params: Target network parameters (unused, but kept for API consistency)
        online_params: Online network parameters to copy
        
    Returns:
        Copy of online parameters
        
    Example:
        >>> target_params = hard_update(target_params, online_params)
    """
    return jax.tree.map(lambda o: o, online_params)


def sample_gumbel(
    key: jax.Array,
    shape: tuple,
    eps: float = 1e-20,
) -> jax.Array:
    """Sample from Gumbel(0, 1) distribution.
    
    Uses the inverse CDF method:
        g = -log(-log(U)) where U ~ Uniform(0, 1)
    
    Args:
        key: JAX random key
        shape: Shape of the output array
        eps: Small constant for numerical stability
        
    Returns:
        Samples from Gumbel(0, 1)
    """
    u = random.uniform(key, shape, minval=eps, maxval=1.0 - eps)
    return -jnp.log(-jnp.log(u))


def gumbel_softmax_sample(
    key: jax.Array,
    logits: jax.Array,
    temperature: float = 1.0,
) -> jax.Array:
    """Draw a sample from the Gumbel-Softmax distribution.
    
    Args:
        key: JAX random key
        logits: Unnormalized log probabilities, shape (..., n_classes)
        temperature: Temperature parameter (lower = more discrete)
        
    Returns:
        Soft samples from the categorical distribution
    """
    gumbel_noise = sample_gumbel(key, logits.shape)
    y = logits + gumbel_noise
    return jax.nn.softmax(y / temperature, axis=-1)


def gumbel_softmax(
    key: jax.Array,
    logits: jax.Array,
    temperature: float = 1.0,
    hard: bool = False,
) -> jax.Array:
    """Sample from the Gumbel-Softmax distribution with optional discretization.
    
    The Gumbel-Softmax is a continuous relaxation of categorical distributions
    that allows for gradient-based optimization.
    
    Args:
        key: JAX random key
        logits: Unnormalized log probabilities, shape (..., n_classes)
        temperature: Temperature parameter (lower = more discrete)
        hard: If True, return one-hot but with gradients through soft sample
        
    Returns:
        If hard=False: Soft samples (probability distribution)
        If hard=True: One-hot samples with straight-through gradients
        
    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> logits = jnp.array([1.0, 2.0, 0.5])
        >>> sample = gumbel_softmax(key, logits, temperature=0.5, hard=True)
    """
    y_soft = gumbel_softmax_sample(key, logits, temperature)
    
    if hard:
        # Straight-through estimator: use argmax in forward, soft in backward
        y_hard = onehot_from_logits(y_soft)
        # Stop gradient on (y_hard - y_soft) so gradients flow through y_soft
        y = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
        return y
    else:
        return y_soft


def onehot_from_logits(
    logits: jax.Array,
    eps: float = 0.0,
) -> jax.Array:
    """Convert logits to one-hot encoding using argmax.
    
    Returns a one-hot vector with 1 at the position of the maximum logit.
    
    Args:
        logits: Input logits, shape (..., n_classes)
        eps: Epsilon for epsilon-greedy (0 = pure argmax)
        
    Returns:
        One-hot encoded array, shape (..., n_classes)
        
    Example:
        >>> logits = jnp.array([1.0, 3.0, 2.0])
        >>> onehot_from_logits(logits)
        Array([0., 1., 0.], dtype=float32)
    """
    n_classes = logits.shape[-1]
    argmax_idx = jnp.argmax(logits, axis=-1)
    return jax.nn.one_hot(argmax_idx, n_classes)


def onehot_from_logits_epsilon_greedy(
    key: jax.Array,
    logits: jax.Array,
    eps: float,
) -> jax.Array:
    """Convert logits to one-hot with epsilon-greedy exploration.
    
    With probability (1-eps), select argmax action.
    With probability eps, select random action.
    
    Args:
        key: JAX random key
        logits: Input logits, shape (..., n_classes)
        eps: Exploration probability
        
    Returns:
        One-hot encoded array with exploration
    """
    n_classes = logits.shape[-1]
    batch_shape = logits.shape[:-1]
    
    key_choice, key_random = random.split(key)
    
    # Argmax actions
    argmax_actions = onehot_from_logits(logits)
    
    # Random actions
    random_indices = random.randint(key_random, batch_shape, 0, n_classes)
    random_actions = jax.nn.one_hot(random_indices, n_classes)
    
    # Choose between argmax and random
    use_random = random.uniform(key_choice, batch_shape) < eps
    
    # Expand use_random for broadcasting
    use_random_expanded = jnp.expand_dims(use_random, axis=-1)
    
    return jnp.where(use_random_expanded, random_actions, argmax_actions)
    
    key_choice, key_random = random.split(key)
    
    # Argmax actions
    argmax_actions = onehot_from_logits(logits)
    
    # Random actions
    random_indices = random.randint(key_random, batch_shape, 0, n_classes)
    random_actions = jax.nn.one_hot(random_indices, n_classes)
    
    # Choose between argmax and random
    use_random = random.uniform(key_choice, batch_shape) < eps
    
    # Expand use_random for broadcasting
    use_random_expanded = jnp.expand_dims(use_random, axis=-1)
    
    return jnp.where(use_random_expanded, random_actions, argmax_actions)


def clip_by_global_norm(
    grads: Params,
    max_norm: float,
) -> Params:
    """Clip gradients by global norm.
    
    Scales all gradients uniformly so that the global norm is at most max_norm.
    
    Args:
        grads: PyTree of gradients
        max_norm: Maximum allowed global norm
        
    Returns:
        Clipped gradients
    """
    # Compute global norm
    leaves = jax.tree.leaves(grads)
    global_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    
    # Compute scaling factor
    scale = jnp.minimum(1.0, max_norm / (global_norm + 1e-8))
    
    # Scale all gradients
    return jax.tree.map(lambda g: g * scale, grads)


def explained_variance(
    y_pred: jax.Array,
    y_true: jax.Array,
) -> jax.Array:
    """Compute explained variance between predictions and targets.
    
    Explained variance = 1 - Var(y_true - y_pred) / Var(y_true)
    
    Returns:
        - 1.0: Perfect predictions
        - 0.0: Predicting the mean would be equally good
        - < 0: Predictions are worse than predicting the mean
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        Explained variance score
    """
    var_true = jnp.var(y_true)
    var_residual = jnp.var(y_true - y_pred)
    return jnp.where(var_true == 0, 0.0, 1.0 - var_residual / var_true)


# ============================================================================
# RL-Specific Utilities
# ============================================================================

def normalize_advantages(
    advantages: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Normalize advantages to have zero mean and unit variance.
    
    Common preprocessing step that improves training stability.
    
    Args:
        advantages: Advantage values
        eps: Small constant for numerical stability
        
    Returns:
        Normalized advantages
    """
    return (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + eps)


def scale_rewards(
    rewards: jax.Array,
    scale: float = 1.0,
    shift: float = 0.0,
) -> jax.Array:
    """Scale and shift rewards.
    
    Useful for reward normalization or shaping.
    
    Args:
        rewards: Raw rewards
        scale: Multiplicative scaling factor
        shift: Additive shift
        
    Returns:
        Scaled rewards = rewards * scale + shift
    """
    return rewards * scale + shift


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jax.Array, jax.Array]:
    """Compute Generalized Advantage Estimation (GAE).
    
    GAE provides a balance between bias and variance in advantage estimation.
    
    Args:
        rewards: Rewards at each timestep (T,)
        values: Value estimates at each timestep (T,)
        dones: Done flags at each timestep (T,)
        next_value: Value estimate for the state after the last timestep
        gamma: Discount factor
        gae_lambda: GAE lambda parameter (0 = TD(0), 1 = MC)
        
    Returns:
        advantages: GAE advantages (T,)
        returns: Discounted returns (T,)
    """
    T = rewards.shape[0]
    
    # Append next_value for bootstrapping
    values_extended = jnp.concatenate([values, jnp.array([next_value])])
    
    def gae_step(carry, t):
        gae, next_gae = carry
        
        # Reverse index
        idx = T - 1 - t
        
        delta = rewards[idx] + gamma * values_extended[idx + 1] * (1 - dones[idx]) - values_extended[idx]
        gae = delta + gamma * gae_lambda * (1 - dones[idx]) * next_gae
        
        return (gae, gae), gae
    
    # Scan backwards through time
    _, advantages_reversed = jax.lax.scan(
        gae_step,
        (jnp.array(0.0), jnp.array(0.0)),
        jnp.arange(T),
    )
    
    # Reverse to get correct order
    advantages = advantages_reversed[::-1]
    returns = advantages + values
    
    return advantages, returns


def polyak_average(
    target_params: Params,
    online_params: Params,
    tau: float,
) -> Params:
    """Alias for soft_update (Polyak averaging).
    
    Same as soft_update but with more descriptive name.
    """
    return soft_update(target_params, online_params, tau)


def huber_loss(
    predictions: jax.Array,
    targets: jax.Array,
    delta: float = 1.0,
) -> jax.Array:
    """Huber loss (smooth L1 loss).
    
    Less sensitive to outliers than MSE.
    
    Args:
        predictions: Predicted values
        targets: Target values
        delta: Threshold for switching between L1 and L2
        
    Returns:
        Huber loss value
    """
    abs_diff = jnp.abs(predictions - targets)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def mse_loss(
    predictions: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """Mean squared error loss.
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        MSE loss value
    """
    return jnp.mean((predictions - targets) ** 2)


def td_target(
    rewards: jax.Array,
    next_q_values: jax.Array,
    dones: jax.Array,
    gamma: float = 0.99,
) -> jax.Array:
    """Compute TD target for Q-learning.
    
    target = reward + gamma * (1 - done) * next_q
    
    Args:
        rewards: Rewards received
        next_q_values: Q-values of next states
        dones: Done flags
        gamma: Discount factor
        
    Returns:
        TD targets
    """
    return rewards + gamma * (1.0 - dones) * next_q_values


# ============================================================================
# Gradient Utilities
# ============================================================================

def get_gradient_norm(grads: Params) -> jax.Array:
    """Compute the global norm of gradients.
    
    Args:
        grads: PyTree of gradients
        
    Returns:
        Global L2 norm of all gradients
    """
    leaves = jax.tree.leaves(grads)
    return jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))


def tree_zeros_like(tree: Params) -> Params:
    """Create a tree of zeros with same structure.
    
    Args:
        tree: Input parameter tree
        
    Returns:
        Tree of zeros with same structure
    """
    return jax.tree.map(jnp.zeros_like, tree)


def tree_add(tree1: Params, tree2: Params) -> Params:
    """Element-wise addition of two parameter trees.
    
    Args:
        tree1: First tree
        tree2: Second tree
        
    Returns:
        Sum of trees
    """
    return jax.tree.map(lambda a, b: a + b, tree1, tree2)


def tree_scalar_multiply(tree: Params, scalar: float) -> Params:
    """Multiply all leaves of a tree by a scalar.
    
    Args:
        tree: Parameter tree
        scalar: Scalar multiplier
        
    Returns:
        Scaled tree
    """
    return jax.tree.map(lambda x: x * scalar, tree)
