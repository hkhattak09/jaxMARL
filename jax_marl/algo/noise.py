"""Exploration noise for MADDPG algorithm.

This module provides JAX-compatible noise generators for exploration:
1. GaussianNoise - Simple Gaussian noise, most commonly used
2. OUNoise - Ornstein-Uhlenbeck process for temporally correlated noise
3. NoiseScheduler - Unified interface for noise with scheduling

All noise generators are designed to be:
- Stateless (state passed explicitly for OU noise)
- JIT-compilable
- vmap-compatible for multi-agent scenarios
"""

from typing import Tuple, NamedTuple, Optional, Callable, Union
import jax
import jax.numpy as jnp
from jax import random
from flax import struct
from functools import partial


# ============================================================================
# Gaussian Noise
# ============================================================================

@struct.dataclass
class GaussianNoiseParams:
    """Parameters for Gaussian noise.
    
    Attributes:
        scale: Standard deviation of the noise
        action_dim: Dimension of action space
    """
    scale: float = 0.1
    action_dim: int = 2


def gaussian_noise(
    key: jax.Array,
    shape: Tuple[int, ...],
    scale: float,
) -> jax.Array:
    """Generate Gaussian noise.
    
    Args:
        key: JAX random key
        shape: Shape of noise array (e.g., (n_agents, action_dim) or (action_dim,))
        scale: Standard deviation of the noise
        
    Returns:
        Noise array with given shape
    """
    return random.normal(key, shape) * scale


def gaussian_noise_log_prob(
    noise: jax.Array,
    scale: float,
) -> jax.Array:
    """Compute log probability of Gaussian noise.
    
    Args:
        noise: Noise values
        scale: Standard deviation
        
    Returns:
        Log probability (sum over dimensions)
    """
    # Log prob of N(0, scale^2)
    # log p(x) = -0.5 * (x/scale)^2 - log(scale) - 0.5*log(2*pi)
    log_prob = -0.5 * (noise / scale) ** 2 - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi)
    return jnp.sum(log_prob, axis=-1)


def add_gaussian_noise(
    key: jax.Array,
    action: jax.Array,
    scale: float,
    clip_low: float = -1.0,
    clip_high: float = 1.0,
) -> Tuple[jax.Array, jax.Array]:
    """Add Gaussian noise to action and clip.
    
    Args:
        key: JAX random key
        action: Action to add noise to
        scale: Noise standard deviation
        clip_low: Lower bound for clipping
        clip_high: Upper bound for clipping
        
    Returns:
        noisy_action: Action with noise, clipped to bounds
        noise: The noise that was added (before clipping)
    """
    noise = gaussian_noise(key, action.shape, scale)
    noisy_action = action + noise
    noisy_action = jnp.clip(noisy_action, clip_low, clip_high)
    return noisy_action, noise


# ============================================================================
# Ornstein-Uhlenbeck Noise
# ============================================================================

@struct.dataclass
class OUNoiseState:
    """State for Ornstein-Uhlenbeck noise process.
    
    The OU process is defined by:
        dx = theta * (mu - x) * dt + sigma * dW
    
    Attributes:
        state: Current state of the OU process
    """
    state: jax.Array


@struct.dataclass  
class OUNoiseParams:
    """Parameters for Ornstein-Uhlenbeck noise.
    
    Attributes:
        mu: Mean to revert to (usually 0)
        theta: Rate of mean reversion (higher = faster reversion)
        sigma: Volatility (noise magnitude)
        scale: Output scaling factor (multiplied to final noise)
        dt: Time step
        action_dim: Dimension of action space
    """
    mu: float = 0.0
    theta: float = 0.15
    sigma: float = 0.2
    scale: float = 1.0  # Output scaling factor
    dt: float = 1.0
    action_dim: int = 2


def ou_noise_init(
    action_dim: int,
    mu: float = 0.0,
) -> OUNoiseState:
    """Initialize OU noise state.
    
    Args:
        action_dim: Dimension of action space
        mu: Initial state value (usually same as mean)
        
    Returns:
        Initial OUNoiseState
    """
    return OUNoiseState(state=jnp.full(action_dim, mu))


def ou_noise_reset(
    params: OUNoiseParams,
) -> OUNoiseState:
    """Reset OU noise state to mean.
    
    Args:
        params: OU noise parameters
        
    Returns:
        Reset OUNoiseState
    """
    return OUNoiseState(state=jnp.full(params.action_dim, params.mu))


def ou_noise_step(
    key: jax.Array,
    state: OUNoiseState,
    params: OUNoiseParams,
) -> Tuple[OUNoiseState, jax.Array]:
    """Generate one step of OU noise and update state.
    
    OU process update:
        x_{t+1} = x_t + theta * (mu - x_t) * dt + sigma * sqrt(dt) * N(0, 1)
    
    Args:
        key: JAX random key
        state: Current OU state
        params: OU parameters
        
    Returns:
        new_state: Updated OU state
        noise: Generated noise (scaled by params.scale)
    """
    # Mean reversion term
    dx_mean = params.theta * (params.mu - state.state) * params.dt
    
    # Stochastic term
    dx_noise = params.sigma * jnp.sqrt(params.dt) * random.normal(key, state.state.shape)
    
    # Update state
    new_state_value = state.state + dx_mean + dx_noise
    new_state = OUNoiseState(state=new_state_value)
    
    # Return scaled noise
    return new_state, new_state_value * params.scale


def add_ou_noise(
    key: jax.Array,
    action: jax.Array,
    ou_state: OUNoiseState,
    params: OUNoiseParams,
    clip_low: float = -1.0,
    clip_high: float = 1.0,
) -> Tuple[jax.Array, OUNoiseState, jax.Array]:
    """Add OU noise to action and update state.
    
    Args:
        key: JAX random key
        action: Action to add noise to
        ou_state: Current OU state
        params: OU parameters
        clip_low: Lower bound for clipping
        clip_high: Upper bound for clipping
        
    Returns:
        noisy_action: Action with noise, clipped
        new_ou_state: Updated OU state
        noise: The noise that was added
    """
    new_state, noise = ou_noise_step(key, ou_state, params)
    noisy_action = action + noise
    noisy_action = jnp.clip(noisy_action, clip_low, clip_high)
    return noisy_action, new_state, noise


# ============================================================================
# Epsilon-Greedy Exploration (for continuous actions)
# ============================================================================

def epsilon_greedy_continuous(
    key: jax.Array,
    action: jax.Array,
    epsilon: float,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> Tuple[jax.Array, jax.Array]:
    """Apply epsilon-greedy exploration for continuous actions.
    
    With probability epsilon, return random uniform action.
    Otherwise, return the original action.
    
    Args:
        key: JAX random key
        action: Policy action
        epsilon: Exploration probability
        action_low: Lower bound for random actions
        action_high: Upper bound for random actions
        
    Returns:
        final_action: Either original or random action
        log_prob: Log probability of action (approximate)
    """
    key_choice, key_random = random.split(key)
    
    # Generate random action
    random_action = random.uniform(
        key_random, 
        action.shape, 
        minval=action_low, 
        maxval=action_high
    )
    
    # Choose between policy action and random action
    use_random = random.uniform(key_choice, ()) < epsilon
    
    final_action = jnp.where(use_random, random_action, action)
    
    # Approximate log probability
    # If random: uniform distribution log prob
    # If policy: assume uniform (this is approximate)
    action_range = action_high - action_low
    log_prob_uniform = -action.shape[-1] * jnp.log(action_range)
    
    return final_action, log_prob_uniform


# ============================================================================
# Batched versions for multi-agent scenarios
# ============================================================================

def gaussian_noise_batch(
    key: jax.Array,
    n_agents: int,
    action_dim: int,
    scale: float,
) -> jax.Array:
    """Generate Gaussian noise for multiple agents.
    
    Args:
        key: JAX random key
        n_agents: Number of agents
        action_dim: Action dimension per agent
        scale: Noise standard deviation
        
    Returns:
        Noise array of shape (n_agents, action_dim)
    """
    return gaussian_noise(key, (n_agents, action_dim), scale)


def add_gaussian_noise_batch(
    key: jax.Array,
    actions: jax.Array,
    scale: float,
    clip_low: float = -1.0,
    clip_high: float = 1.0,
) -> Tuple[jax.Array, jax.Array]:
    """Add Gaussian noise to batched actions.
    
    Args:
        key: JAX random key
        actions: Actions of shape (n_agents, action_dim) or (batch, n_agents, action_dim)
        scale: Noise standard deviation
        clip_low: Lower bound
        clip_high: Upper bound
        
    Returns:
        noisy_actions: Actions with noise
        noise: The noise that was added
    """
    return add_gaussian_noise(key, actions, scale, clip_low, clip_high)


def ou_noise_init_batch(
    n_agents: int,
    action_dim: int,
    mu: float = 0.0,
) -> OUNoiseState:
    """Initialize OU noise state for multiple agents.
    
    Args:
        n_agents: Number of agents
        action_dim: Action dimension per agent
        mu: Initial value
        
    Returns:
        OUNoiseState with shape (n_agents, action_dim)
    """
    return OUNoiseState(state=jnp.full((n_agents, action_dim), mu))


def ou_noise_step_batch(
    key: jax.Array,
    state: OUNoiseState,
    params: OUNoiseParams,
) -> Tuple[OUNoiseState, jax.Array]:
    """Generate OU noise for multiple agents.
    
    Args:
        key: JAX random key
        state: OU state of shape (n_agents, action_dim)
        params: OU parameters
        
    Returns:
        new_state: Updated state
        noise: Generated noise of shape (n_agents, action_dim)
    """
    # Same as single-agent but works with batched state
    dx_mean = params.theta * (params.mu - state.state) * params.dt
    dx_noise = params.sigma * jnp.sqrt(params.dt) * random.normal(key, state.state.shape)
    new_state_value = state.state + dx_mean + dx_noise
    new_state = OUNoiseState(state=new_state_value)
    return new_state, new_state_value


# ============================================================================
# Noise schedule helpers
# ============================================================================

def linear_schedule(
    initial_value: float,
    final_value: float,
    current_step: int,
    total_steps: int,
) -> float:
    """Linear interpolation between initial and final values.
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        current_step: Current timestep
        total_steps: Total timesteps for schedule
        
    Returns:
        Interpolated value
    """
    fraction = jnp.minimum(current_step / total_steps, 1.0)
    return initial_value + fraction * (final_value - initial_value)


def exponential_schedule(
    initial_value: float,
    final_value: float,
    current_step: int,
    decay_rate: float,
) -> float:
    """Exponential decay from initial to final value.
    
    value = final + (initial - final) * exp(-decay_rate * step)
    
    Args:
        initial_value: Starting value
        final_value: Ending value (asymptotic)
        current_step: Current timestep
        decay_rate: Rate of decay
        
    Returns:
        Decayed value
    """
    return final_value + (initial_value - final_value) * jnp.exp(-decay_rate * current_step)


def cosine_schedule(
    initial_value: float,
    final_value: float,
    current_step: int,
    total_steps: int,
) -> float:
    """Cosine annealing from initial to final value.
    
    Smoother transition than linear schedule, commonly used in learning rates.
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        current_step: Current timestep
        total_steps: Total timesteps for schedule
        
    Returns:
        Annealed value following cosine curve
    """
    fraction = jnp.minimum(current_step / total_steps, 1.0)
    cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * fraction))
    return final_value + (initial_value - final_value) * cosine_decay


def warmup_linear_schedule(
    initial_value: float,
    peak_value: float,
    final_value: float,
    current_step: int,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Linear warmup followed by linear decay.
    
    Args:
        initial_value: Starting value
        peak_value: Peak value after warmup
        final_value: Final value
        current_step: Current timestep
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        
    Returns:
        Scheduled value
    """
    # Warmup phase
    warmup_value = initial_value + (peak_value - initial_value) * (current_step / warmup_steps)
    
    # Decay phase
    decay_fraction = (current_step - warmup_steps) / (total_steps - warmup_steps)
    decay_value = peak_value + (final_value - peak_value) * decay_fraction
    
    return jnp.where(
        current_step < warmup_steps,
        warmup_value,
        jnp.where(current_step < total_steps, decay_value, final_value)
    )


# ============================================================================
# Unified Noise Scheduler
# ============================================================================

@struct.dataclass
class NoiseSchedulerState:
    """State for noise scheduler.
    
    Attributes:
        step: Current timestep
        ou_state: Optional OU noise state
    """
    step: jnp.ndarray
    ou_state: Optional[OUNoiseState] = None


class NoiseScheduler:
    """Unified noise scheduler with automatic scaling.
    
    Combines noise generation with schedule-based scaling.
    Supports both Gaussian and OU noise.
    
    Example:
        ```python
        scheduler = NoiseScheduler(
            noise_type='gaussian',
            initial_scale=0.3,
            final_scale=0.05,
            schedule='linear',
            total_steps=100000,
            action_dim=2,
        )
        
        state = scheduler.init()
        
        # During training loop:
        action, new_state, noise = scheduler.add_noise(key, action, state)
        ```
    """
    
    def __init__(
        self,
        noise_type: str = 'gaussian',
        initial_scale: float = 0.3,
        final_scale: float = 0.05,
        schedule: str = 'linear',
        total_steps: int = 100000,
        action_dim: int = 2,
        # OU-specific params
        ou_theta: float = 0.15,
        ou_sigma: float = 0.2,
        ou_mu: float = 0.0,
        # Clipping
        clip_low: float = -1.0,
        clip_high: float = 1.0,
    ):
        """Initialize noise scheduler.
        
        Args:
            noise_type: 'gaussian' or 'ou'
            initial_scale: Starting noise scale
            final_scale: Final noise scale
            schedule: 'linear', 'exponential', or 'cosine'
            total_steps: Total steps for schedule
            action_dim: Action dimension
            ou_theta: OU mean reversion rate
            ou_sigma: OU volatility
            ou_mu: OU mean
            clip_low: Lower bound for actions
            clip_high: Upper bound for actions
        """
        self.noise_type = noise_type
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.schedule = schedule
        self.total_steps = total_steps
        self.action_dim = action_dim
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.ou_mu = ou_mu
        self.clip_low = clip_low
        self.clip_high = clip_high
    
    def init(self) -> NoiseSchedulerState:
        """Initialize scheduler state."""
        ou_state = None
        if self.noise_type == 'ou':
            ou_state = ou_noise_init(self.action_dim, self.ou_mu)
        
        return NoiseSchedulerState(
            step=jnp.array(0, dtype=jnp.int32),
            ou_state=ou_state,
        )
    
    def get_scale(self, step: jnp.ndarray) -> float:
        """Get current noise scale based on schedule."""
        if self.schedule == 'linear':
            return linear_schedule(
                self.initial_scale, self.final_scale, step, self.total_steps
            )
        elif self.schedule == 'exponential':
            decay_rate = -jnp.log(self.final_scale / self.initial_scale) / self.total_steps
            return exponential_schedule(
                self.initial_scale, self.final_scale, step, decay_rate
            )
        elif self.schedule == 'cosine':
            return cosine_schedule(
                self.initial_scale, self.final_scale, step, self.total_steps
            )
        else:
            return self.initial_scale  # constant
    
    def add_noise(
        self,
        key: jax.Array,
        action: jax.Array,
        state: NoiseSchedulerState,
    ) -> Tuple[jax.Array, NoiseSchedulerState, jax.Array]:
        """Add noise to action and update state.
        
        Args:
            key: JAX random key
            action: Action to add noise to
            state: Current scheduler state
            
        Returns:
            noisy_action: Action with noise (clipped)
            new_state: Updated scheduler state
            noise: The noise that was added
        """
        scale = self.get_scale(state.step)
        
        if self.noise_type == 'gaussian':
            noise = gaussian_noise(key, action.shape, scale)
            noisy_action = jnp.clip(action + noise, self.clip_low, self.clip_high)
            new_state = NoiseSchedulerState(
                step=state.step + 1,
                ou_state=None,
            )
        else:  # ou
            params = OUNoiseParams(
                mu=self.ou_mu,
                theta=self.ou_theta,
                sigma=self.ou_sigma,
                scale=scale,
                action_dim=self.action_dim,
            )
            new_ou_state, noise = ou_noise_step(key, state.ou_state, params)
            noisy_action = jnp.clip(action + noise, self.clip_low, self.clip_high)
            new_state = NoiseSchedulerState(
                step=state.step + 1,
                ou_state=new_ou_state,
            )
        
        return noisy_action, new_state, noise
    
    def reset(self, state: NoiseSchedulerState) -> NoiseSchedulerState:
        """Reset noise state (keeps step counter)."""
        ou_state = None
        if self.noise_type == 'ou':
            ou_state = ou_noise_init(self.action_dim, self.ou_mu)
        
        return NoiseSchedulerState(
            step=state.step,
            ou_state=ou_state,
        )
