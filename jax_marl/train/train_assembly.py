#!/usr/bin/env python3
"""JAX-based MADDPG training for Assembly Swarm Environment.

This module provides the main training loop for MADDPG on the Assembly
Swarm environment, using JAX for accelerated computation.

Key features:
- JIT-compiled training step for GPU acceleration
- Vectorized parallel environments (n_parallel_envs)
- Functional state updates (no in-place mutations)
- Uses JAX's random key management

Usage:
    # Configure training in cfg/assembly_cfg.py, then:
    python training.py
    
    # Or from Python:
    from train import train
    from cfg import get_config
    
    config = get_config()  # Uses values from assembly_cfg.py
    train(config)

The training script supports:
- Parallel environment execution (configure n_parallel_envs)
- Checkpointing and resuming
- JSON file logging (no TensorBoard required)
- Prior policy regularization (LLM-guided training)
- Flexible noise scheduling
"""

import os
import sys
import time
import pickle
from typing import Tuple, Dict, Any, Optional, NamedTuple, List
from pathlib import Path
from datetime import datetime
from functools import partial
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
import jax
import jax.numpy as jnp
from jax import random
from flax import struct
import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "jax_cus_gym"))

# Configuration imports
from cfg import (
    AssemblyTrainConfig,
    get_config,
    get_shape_file_path,
    get_checkpoint_dir,
    get_log_dir,
    get_eval_dir,
    config_to_maddpg_config,
    config_to_assembly_params,
)

# Algorithm imports
from algo import MADDPG, MADDPGConfig, MADDPGState

# Environment imports
from assembly_env import (
    AssemblySwarmEnv,
    AssemblyParams,
    AssemblyState,
    make_assembly_env,
    make_vec_env,
    compute_prior_policy,
)
from shape_loader import load_shapes_from_pickle
from observations import compute_observation_dim


# ============================================================================
# Training State
# ============================================================================

class TrainingMetrics(NamedTuple):
    """Metrics collected during training."""
    episode_reward_mean: float
    episode_reward_std: float
    coverage_rate: float
    collision_rate: float
    avg_dist_to_target: float
    step_time: float
    train_time: float
    noise_scale: float
    buffer_size: int


@struct.dataclass
class TrainingState:
    """Complete training state.
    
    Attributes:
        maddpg_state: MADDPG algorithm state
        env_states: Batched environment states (n_parallel_envs,)
        key: JAX random key
        episode: Current episode number
        total_steps: Total environment steps
        best_reward: Best episode reward seen
    """
    maddpg_state: MADDPGState
    env_states: AssemblyState  # Batched: (n_parallel_envs, ...)
    key: jax.Array
    episode: int
    total_steps: int
    best_reward: float


# ============================================================================
# Training Functions
# ============================================================================

def create_training_state(
    config: AssemblyTrainConfig,
    key: jax.Array,
) -> Tuple[TrainingState, AssemblySwarmEnv, MADDPG, AssemblyParams, callable, callable]:
    """Create initial training state with parallel environments.
    
    Args:
        config: Training configuration
        key: JAX random key
        
    Returns:
        training_state: Initial training state
        env: Environment instance
        maddpg: MADDPG instance
        params: Environment parameters
        vec_reset: Vectorized reset function
        vec_step: Vectorized step function
    """
    key, env_key, algo_key = random.split(key, 3)
    
    # Load shapes
    shape_file = get_shape_file_path(config)
    print(f"Loading shapes from: {shape_file}")
    
    try:
        shape_library = load_shapes_from_pickle(shape_file)
        print(f"Loaded {shape_library.n_shapes} shapes")
    except FileNotFoundError:
        print(f"Shape file not found, using procedural shapes")
        shape_library = None
    
    # Create vectorized environment
    n_envs = config.n_parallel_envs
    env, _, _, _ = make_vec_env(
        n_envs=n_envs,
        n_agents=config.n_agents,
        shape_library=shape_library,
    )
    
    # Override params with config values
    params = config_to_assembly_params(config)
    
    # Create vectorized functions with updated params
    @jax.jit
    def vec_reset(keys: jnp.ndarray):
        return jax.vmap(lambda k: env.reset(k, params))(keys)
    
    @jax.jit
    def vec_step(keys: jnp.ndarray, states, actions: jnp.ndarray):
        return jax.vmap(lambda k, s, a: env.step(k, s, a, params))(keys, states, actions)
    
    # Get actual observation dimension from environment reset
    # (computed obs_dim may differ from actual environment output)
    test_keys = random.split(env_key, n_envs)
    test_obs, _ = vec_reset(test_keys)
    obs_dim = test_obs.shape[-1]  # Actual obs dim from environment
    action_dim = 2  # 2D acceleration
    
    print(f"Environment: {config.n_agents} agents, obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"Parallel environments: {n_envs}")
    
    # Create MADDPG
    maddpg_config = config_to_maddpg_config(config, obs_dim, action_dim)
    maddpg = MADDPG(maddpg_config)
    maddpg_state = maddpg.init(algo_key)
    
    # Reset all parallel environments
    env_keys = random.split(env_key, n_envs)
    obs_batch, env_states = vec_reset(env_keys)
    
    training_state = TrainingState(
        maddpg_state=maddpg_state,
        env_states=env_states,
        key=key,
        episode=0,
        total_steps=0,
        best_reward=float('-inf'),
    )
    
    return training_state, env, maddpg, params, vec_reset, vec_step


# ============================================================================
# JIT-Compiled Rollout State
# ============================================================================

@struct.dataclass
class RolloutCarry:
    """State carried through the JIT-compiled rollout loop.
    
    All fields must be JAX arrays or pytrees of JAX arrays for JIT compatibility.
    """
    key: jax.Array
    obs_batch: jax.Array  # (n_envs, n_agents, obs_dim)
    env_states: AssemblyState  # Batched environment states
    maddpg_state: MADDPGState  # Algorithm state with buffer
    episode_rewards: jax.Array  # (n_envs,)
    total_coverage: jax.Array  # scalar
    total_collision: jax.Array  # scalar
    done_flag: jax.Array  # bool scalar - whether all envs are done


@struct.dataclass  
class RolloutMetrics:
    """Metrics from a single rollout step (used with lax.scan)."""
    reward: jax.Array
    coverage: jax.Array
    collision: jax.Array


def create_jit_rollout_fn(
    maddpg: MADDPG,
    params: AssemblyParams,
    config: AssemblyTrainConfig,
    vec_step: callable,
):
    """Create a JIT-compiled rollout function.
    
    This function creates a rollout that runs entirely on GPU without Python loop overhead.
    Uses jax.lax.scan for the inner loop.
    
    Args:
        maddpg: MADDPG instance (for network structure)
        params: Environment parameters  
        config: Training configuration
        vec_step: Vectorized step function
        
    Returns:
        jit_rollout_fn: JIT-compiled function that runs a full episode
    """
    n_envs = config.n_parallel_envs
    n_agents = config.n_agents
    max_steps = config.max_steps
    use_prior = config.prior_weight > 0
    
    # Pre-define the prior computation function outside the loop
    @jax.jit
    def compute_priors_batch(positions, velocities, grid_centers, grid_mask, l_cell):
        """Compute prior actions for all environments."""
        def single_env_prior(pos, vel, gc, gm, lc):
            # Compute r_avoid dynamically using MARL-LLM formula
            from assembly_env import compute_r_avoid
            r_avoid = compute_r_avoid(gm, n_agents, lc, params.r_avoid)
            return compute_prior_policy(
                pos, vel, gc, gm, lc,
                r_avoid,
                params.d_sen,
            )
        return jax.vmap(single_env_prior)(positions, velocities, grid_centers, grid_mask, l_cell)
    
    def rollout_step(carry: RolloutCarry, step_idx: int) -> Tuple[RolloutCarry, RolloutMetrics]:
        """Single step of the rollout - will be scanned over."""
        # Split keys
        key, action_key, step_key = random.split(carry.key, 3)
        step_keys = random.split(step_key, n_envs)
        
        # Select actions (batched over envs)
        actions_batch, new_maddpg_state = maddpg.select_actions_batched(
            action_key, carry.maddpg_state, carry.obs_batch, explore=True
        )
        
        # Step environments
        next_obs_batch, new_env_states, rewards_batch, dones_batch, info_batch = vec_step(
            step_keys, carry.env_states, actions_batch
        )
        
        # Compute priors if needed
        if use_prior:
            prior_actions_batch = compute_priors_batch(
                new_env_states.positions,
                new_env_states.velocities,
                new_env_states.grid_centers,
                new_env_states.grid_mask,
                new_env_states.l_cell,
            )
        else:
            prior_actions_batch = None
        
        # Store transitions
        new_maddpg_state = maddpg.store_transitions_batched(
            new_maddpg_state,
            carry.obs_batch,
            actions_batch,
            rewards_batch,
            next_obs_batch,
            dones_batch,
            action_priors_batch=prior_actions_batch,
        )
        
        # Accumulate metrics
        step_reward = jnp.mean(rewards_batch, axis=1)  # (n_envs,)
        step_coverage = jnp.mean(info_batch["coverage_rate"])
        step_collision = jnp.mean(new_env_states.is_colliding)
        
        # Check if done
        all_done = jnp.all(dones_batch[:, 0])
        
        # Create new carry
        new_carry = RolloutCarry(
            key=key,
            obs_batch=next_obs_batch,
            env_states=new_env_states,
            maddpg_state=new_maddpg_state,
            episode_rewards=carry.episode_rewards + step_reward,
            total_coverage=carry.total_coverage + step_coverage,
            total_collision=carry.total_collision + step_collision,
            done_flag=carry.done_flag | all_done,
        )
        
        # Metrics for this step
        metrics = RolloutMetrics(
            reward=jnp.mean(step_reward),
            coverage=step_coverage,
            collision=step_collision,
        )
        
        return new_carry, metrics
    
    @jax.jit
    def jit_rollout(
        key: jax.Array,
        obs_batch: jax.Array,
        env_states: AssemblyState,
        maddpg_state: MADDPGState,
    ) -> Tuple[RolloutCarry, RolloutMetrics]:
        """Run a full episode rollout (JIT-compiled).
        
        Args:
            key: Random key
            obs_batch: Initial observations (n_envs, n_agents, obs_dim)
            env_states: Initial environment states
            maddpg_state: Initial MADDPG state
            
        Returns:
            final_carry: Final state after rollout
            all_metrics: Stacked metrics from all steps
        """
        initial_carry = RolloutCarry(
            key=key,
            obs_batch=obs_batch,
            env_states=env_states,
            maddpg_state=maddpg_state,
            episode_rewards=jnp.zeros(n_envs),
            total_coverage=jnp.array(0.0),
            total_collision=jnp.array(0.0),
            done_flag=jnp.array(False),
        )
        
        # Run the scan over all steps
        final_carry, all_metrics = jax.lax.scan(
            rollout_step,
            initial_carry,
            jnp.arange(max_steps),
        )
        
        return final_carry, all_metrics
    
    return jit_rollout


def run_episode(
    env: AssemblySwarmEnv,
    maddpg: MADDPG,
    training_state: TrainingState,
    params: AssemblyParams,
    config: AssemblyTrainConfig,
    vec_reset: callable,
    vec_step: callable,
    jit_rollout_fn: Optional[callable] = None,
    explore: bool = True,
) -> Tuple[TrainingState, Dict[str, Any]]:
    """Run a single episode across all parallel environments.
    
    Uses JIT-compiled rollout if provided, otherwise falls back to Python loop.
    
    Args:
        env: Environment instance
        maddpg: MADDPG instance
        training_state: Current training state
        params: Environment parameters
        config: Training configuration
        vec_reset: Vectorized reset function
        vec_step: Vectorized step function
        jit_rollout_fn: Optional JIT-compiled rollout function
        explore: Whether to use exploration noise
        
    Returns:
        new_training_state: Updated training state
        metrics: Episode metrics (averaged across parallel envs)
    """
    key = training_state.key
    maddpg_state = training_state.maddpg_state
    n_envs = config.n_parallel_envs
    
    # Reset all parallel environments
    key, reset_key = random.split(key)
    reset_keys = random.split(reset_key, n_envs)
    obs_batch, env_states = vec_reset(reset_keys)
    
    step_start = time.time()
    
    # Use JIT-compiled rollout if available
    if jit_rollout_fn is not None:
        key, rollout_key = random.split(key)
        final_carry, all_metrics = jit_rollout_fn(
            rollout_key, obs_batch, env_states, maddpg_state
        )
        
        # Extract results from carry
        maddpg_state = final_carry.maddpg_state
        env_states = final_carry.env_states
        episode_rewards = final_carry.episode_rewards
        total_coverage = final_carry.total_coverage
        total_collision = final_carry.total_collision
        num_steps = config.max_steps  # Always runs full episode with scan
        
    else:
        # Fallback to Python loop (slower but useful for debugging)
        episode_rewards = jnp.zeros(n_envs)
        total_coverage = jnp.array(0.0)
        total_collision = jnp.array(0.0)
        num_steps = 0
        
        for step in range(config.max_steps):
            key, action_key, step_key = random.split(key, 3)
            step_keys = random.split(step_key, n_envs)
            
            actions_batch, maddpg_state = maddpg.select_actions_batched(
                action_key, maddpg_state, obs_batch, explore=explore
            )
            
            next_obs_batch, env_states, rewards_batch, dones_batch, info_batch = vec_step(
                step_keys, env_states, actions_batch
            )
            
            if config.prior_weight > 0:
                def compute_priors_single_env(positions, velocities, grid_centers, grid_mask, l_cell):
                    # Compute r_avoid dynamically using MARL-LLM formula
                    from assembly_env import compute_r_avoid
                    r_avoid = compute_r_avoid(grid_mask, n_agents, l_cell, params.r_avoid)
                    return compute_prior_policy(
                        positions, velocities, grid_centers, grid_mask, l_cell,
                        r_avoid,
                        params.d_sen,
                    )
                
                prior_actions_batch = jax.vmap(compute_priors_single_env)(
                    env_states.positions,
                    env_states.velocities,
                    env_states.grid_centers,
                    env_states.grid_mask,
                    env_states.l_cell,
                )
            else:
                prior_actions_batch = None
            
            maddpg_state = maddpg.store_transitions_batched(
                maddpg_state,
                obs_batch,
                actions_batch,
                rewards_batch,
                next_obs_batch,
                dones_batch,
                action_priors_batch=prior_actions_batch,
            )
            
            obs_batch = next_obs_batch
            episode_rewards = episode_rewards + jnp.mean(rewards_batch, axis=1)
            total_coverage = total_coverage + jnp.mean(info_batch["coverage_rate"])
            total_collision = total_collision + jnp.mean(env_states.is_colliding)
            num_steps = step + 1
            
            if jnp.all(dones_batch[:, 0]):
                break
    
    step_time = time.time() - step_start
    
    # Update step counters
    total_steps = training_state.total_steps + num_steps * n_envs
    mean_episode_reward = float(jnp.mean(episode_rewards))
    
    # Update noise scale
    noise_scale = maddpg_state.noise_scale
    if config.noise_decay_steps > 0:
        decay_progress = min(1.0, total_steps / config.noise_decay_steps)
        new_noise = config.noise_scale_initial - decay_progress * (
            config.noise_scale_initial - config.noise_scale_final
        )
        maddpg_state = maddpg_state.replace(noise_scale=jnp.array(new_noise))
    
    new_training_state = TrainingState(
        maddpg_state=maddpg_state,
        env_states=env_states,
        key=key,
        episode=training_state.episode + 1,
        total_steps=total_steps,
        best_reward=max(training_state.best_reward, mean_episode_reward),
    )
    
    metrics = {
        "episode_reward": mean_episode_reward,
        "episode_reward_mean": mean_episode_reward / num_steps if num_steps > 0 else 0.0,
        "episode_reward_std": 0.0,
        "coverage_rate": float(total_coverage) / num_steps if num_steps > 0 else 0.0,
        "collision_rate": float(total_collision) / num_steps if num_steps > 0 else 0.0,
        "step_time": step_time,
        "noise_scale": float(maddpg_state.noise_scale),
        "steps_in_episode": num_steps,
        "n_parallel_envs": n_envs,
        "total_transitions": num_steps * n_envs,
    }
    
    return new_training_state, metrics


def run_eval_episode(
    env: AssemblySwarmEnv,
    maddpg: MADDPG,
    maddpg_state: MADDPGState,
    params: AssemblyParams,
    config: AssemblyTrainConfig,
    key: jax.Array,
) -> Tuple[Dict[str, Any], List[AssemblyState]]:
    """Run a single evaluation episode (no exploration noise, no buffer updates).
    
    Collects states for the entire episode for visualization.
    
    Args:
        env: Environment instance
        maddpg: MADDPG instance
        maddpg_state: Current MADDPG state (not modified)
        params: Environment parameters
        config: Training configuration
        key: JAX random key
        
    Returns:
        metrics: Episode metrics
        states: List of environment states for visualization
    """
    # Reset single environment for eval (not parallel - we want one clean trajectory)
    key, reset_key = random.split(key)
    obs, env_state = env.reset(reset_key, params)
    
    # Collect states for visualization
    states = [env_state]
    
    # Episode accumulators
    episode_reward = 0.0
    step_rewards = []
    coverages = []
    collisions = []
    
    for step in range(config.max_steps):
        key, action_key, step_key = random.split(key, 3)
        
        # Select actions WITHOUT exploration noise
        obs_list = [obs[i] for i in range(config.n_agents)]
        actions, _, _ = maddpg.select_actions(
            action_key, maddpg_state, obs_list, explore=False  # No noise!
        )
        
        # Stack actions for environment
        actions_array = jnp.stack(actions)
        
        # Step environment
        next_obs, env_state, rewards, dones, info = env.step(
            step_key, env_state, actions_array, params
        )
        
        # Save state for visualization
        states.append(env_state)
        
        # Update observations
        obs = next_obs
        
        # Accumulate metrics
        step_reward = float(jnp.mean(rewards))
        episode_reward += step_reward
        step_rewards.append(step_reward)
        coverages.append(float(info["coverage_rate"]))
        collisions.append(float(jnp.mean(env_state.is_colliding)))
        
        if dones[0]:
            break
    
    metrics = {
        "eval_reward": episode_reward,
        "eval_coverage": np.mean(coverages),
        "eval_collision": np.mean(collisions),
        "eval_final_coverage": coverages[-1] if coverages else 0.0,
        "eval_steps": step + 1,
    }
    
    return metrics, states


def create_jit_train_step(
    maddpg: MADDPG,
    config: AssemblyTrainConfig,
):
    """Create a JIT-compiled training step function.
    
    This creates a properly JIT-compiled training function that uses lax.scan
    instead of Python loops for maximum performance.
    
    Args:
        maddpg: MADDPG instance
        config: Training configuration
        
    Returns:
        jit_train_step: JIT-compiled training function
    """
    updates_per_step = config.updates_per_step
    warmup_steps = config.warmup_steps
    
    # Get the JIT-compiled update from MADDPG
    # This compiles the entire agent update loop into a single XLA computation
    jit_maddpg_update = maddpg.create_jit_update()
    
    def single_update(carry, _):
        """Single update step for lax.scan."""
        maddpg_state, key = carry
        key, update_key = random.split(key)
        new_state, update_info = jit_maddpg_update(update_key, maddpg_state)
        actor_loss = update_info.get("actor_loss", jnp.array(0.0))
        critic_loss = update_info.get("critic_loss", jnp.array(0.0))
        return (new_state, key), (actor_loss, critic_loss)
    
    @jax.jit
    def jit_train_step(
        maddpg_state: MADDPGState,
        key: jax.Array,
    ) -> Tuple[MADDPGState, Dict[str, Any]]:
        """JIT-compiled training step with lax.scan."""
        buffer_size = maddpg_state.buffer_state.size
        
        def do_updates(state_key):
            state, k = state_key
            (final_state, _), (actor_losses, critic_losses) = jax.lax.scan(
                single_update,
                (state, k),
                jnp.arange(updates_per_step),
            )
            mean_actor_loss = jnp.mean(actor_losses)
            mean_critic_loss = jnp.mean(critic_losses)
            return final_state, mean_actor_loss, mean_critic_loss
        
        def skip_updates(state_key):
            state, _ = state_key
            return state, jnp.array(0.0), jnp.array(0.0)
        
        # Use lax.cond to conditionally skip updates
        can_update = buffer_size >= warmup_steps
        final_state, actor_loss, critic_loss = jax.lax.cond(
            can_update,
            do_updates,
            skip_updates,
            (maddpg_state, key),
        )
        
        info = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "buffer_size": buffer_size,
            "updated": can_update,
        }
        
        return final_state, info
    
    return jit_train_step


def train_step(
    maddpg: MADDPG,
    maddpg_state: MADDPGState,
    key: jax.Array,
    config: AssemblyTrainConfig,
) -> Tuple[MADDPGState, Dict[str, float]]:
    """Perform gradient updates (LEGACY - use create_jit_train_step for speed).
    
    This is kept for backwards compatibility but is slow due to Python loops.
    Use create_jit_train_step() instead for production training.
    
    Args:
        maddpg: MADDPG instance
        maddpg_state: Current MADDPG state
        key: Random key
        config: Training configuration
        
    Returns:
        new_state: Updated MADDPG state
        info: Training metrics
    """
    # Check if buffer has enough samples
    buffer_size = maddpg_state.buffer_state.size
    if buffer_size < config.warmup_steps:
        return maddpg_state, {"buffer_size": int(buffer_size), "updated": False}
    
    # Perform multiple gradient updates
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    
    for _ in range(config.updates_per_step):
        key, update_key = random.split(key)
        maddpg_state, update_info = maddpg.update(update_key, maddpg_state)
        
        total_actor_loss += update_info.get("actor_loss", 0.0)
        total_critic_loss += update_info.get("critic_loss", 0.0)
    
    info = {
        "actor_loss": total_actor_loss / config.updates_per_step,
        "critic_loss": total_critic_loss / config.updates_per_step,
        "buffer_size": int(buffer_size),
        "updated": True,
    }
    
    return maddpg_state, info


# ============================================================================
# Main Training Loop
# ============================================================================

def train(
    config: Optional[AssemblyTrainConfig] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> TrainingState:
    """Main training function.
    
    Args:
        config: Training configuration (uses get_config() if None)
        checkpoint_path: Path to resume from checkpoint
        verbose: Print progress messages
        
    Returns:
        Final training state
    """
    if config is None:
        config = get_config()
    
    # Setup logging and eval directories
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = Path(get_checkpoint_dir(config, run_name))
    log_dir = Path(get_log_dir(config, run_name))
    eval_dir = Path(get_eval_dir(config, run_name)) if config.eval_interval > 0 else None
    
    print("=" * 60)
    print("  JAX MADDPG Training - Assembly Swarm")
    print("=" * 60)
    print(f"Agents: {config.n_agents}")
    print(f"Parallel environments: {config.n_parallel_envs}")
    print(f"Episodes: {config.n_episodes}")
    print(f"Prior weight: {config.prior_weight}")
    print(f"Batch size: {config.batch_size}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Log dir: {log_dir}")
    if eval_dir:
        print(f"Eval dir: {eval_dir}")
        print(f"Eval interval: every {config.eval_interval} episodes")
    print("=" * 60)
    
    # Initialize random key
    key = random.PRNGKey(config.seed)
    
    # Create training state with parallel environments
    training_state, env, maddpg, params, vec_reset, vec_step = create_training_state(config, key)
    
    # Create JIT-compiled rollout function for maximum speed
    print("Creating JIT-compiled rollout function...")
    jit_rollout_fn = create_jit_rollout_fn(maddpg, params, config, vec_step)
    print("JIT rollout function created (will compile on first use)")
    
    # Create JIT-compiled training step
    print("Creating JIT-compiled training step...")
    jit_train_step = create_jit_train_step(maddpg, config)
    print("JIT training step created (will compile on first use)")
    
    # Load checkpoint if resuming
    start_episode = 0
    if checkpoint_path is not None:
        loaded_state, loaded_config = load_checkpoint(checkpoint_path, config)
        training_state = loaded_state
        start_episode = training_state.episode + 1
        print(f"Resuming from episode {start_episode}")
    
    # Setup simple JSON/CSV logging (no TensorBoard required)
    log_file = Path(log_dir) / "training_log.json"
    training_history = []
    
    # Training loop
    print("\nTraining started...")
    training_start = time.time()
    
    for episode in range(start_episode, config.n_episodes):
        # Run episode across all parallel environments (using JIT-compiled rollout)
        episode_start = time.time()
        training_state, episode_metrics = run_episode(
            env, maddpg, training_state, params, config, vec_reset, vec_step,
            jit_rollout_fn=jit_rollout_fn, explore=True
        )
        
        # Perform gradient updates (JIT-compiled)
        train_start = time.time()
        key, train_key = random.split(training_state.key)
        maddpg_state, train_info = jit_train_step(
            training_state.maddpg_state, train_key
        )
        training_state = training_state.replace(
            maddpg_state=maddpg_state,
            key=key,
        )
        train_time = time.time() - train_start
        
        # Convert JAX arrays to Python scalars for logging
        buffer_size_val = int(train_info['buffer_size'])
        actor_loss_val = float(train_info['actor_loss']) if train_info['updated'] else 0.0
        critic_loss_val = float(train_info['critic_loss']) if train_info['updated'] else 0.0
        
        # Calculate elapsed time
        elapsed_time = time.time() - training_start
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = elapsed_time % 60
        
        # Logging
        if episode % config.log_interval == 0 and verbose:
            print(
                f"Episode {episode:5d}/{config.n_episodes} | "
                f"Reward: {episode_metrics['episode_reward']:7.3f} | "
                f"Coverage: {episode_metrics['coverage_rate']:.3f} | "
                f"Collisions: {episode_metrics['collision_rate']:.3f} | "
                f"Noise: {episode_metrics['noise_scale']:.3f} | "
                f"Buffer: {buffer_size_val:6d} | "
                f"Step: {episode_metrics['step_time']:.2f}s | "
                f"Train: {train_time:.2f}s | "
                f"Elapsed: {elapsed_mins}m {elapsed_secs:.1f}s"
            )
        
        # JSON logging (save metrics to file for later analysis)
        if episode % config.log_interval == 0:
            log_entry = {
                "episode": episode,
                "reward_mean": episode_metrics["episode_reward"],
                "reward_std": episode_metrics.get("episode_reward_std", 0.0),
                "coverage_rate": episode_metrics["coverage_rate"],
                "collision_rate": episode_metrics["collision_rate"],
                "noise_scale": episode_metrics["noise_scale"],
                "buffer_size": buffer_size_val,
                "step_time": episode_metrics["step_time"],
                "train_time": train_time,
                "n_parallel_envs": config.n_parallel_envs,
                "total_transitions": episode_metrics.get("total_transitions", 0),
            }
            if train_info.get("updated", False):
                log_entry["actor_loss"] = actor_loss_val
                log_entry["critic_loss"] = critic_loss_val
            training_history.append(log_entry)
            
            # Periodically save to file
            if episode % config.save_interval == 0:
                import json
                with open(log_file, 'w') as f:
                    json.dump(training_history, f, indent=2)
        
        # Checkpointing
        if episode % (config.save_interval * 4) == 0 and episode > 0:
            save_checkpoint(
                training_state, maddpg, config,
                checkpoint_dir / "incremental" / f"checkpoint_ep{episode}.pkl"
            )
        
        # Evaluation (no noise, collect states for visualization)
        if config.eval_interval > 0 and episode % config.eval_interval == 0 and episode > 0:
            key, eval_key = random.split(training_state.key)
            training_state = training_state.replace(key=key)
            
            eval_metrics, eval_states = run_eval_episode(
                env, maddpg, training_state.maddpg_state, params, config, eval_key
            )
            
            if verbose:
                print(
                    f"  [EVAL] Episode {episode} | "
                    f"Reward: {eval_metrics['eval_reward']:7.3f} | "
                    f"Coverage: {eval_metrics['eval_coverage']:.3f} | "
                    f"Final Coverage: {eval_metrics['eval_final_coverage']:.3f}"
                )
            
            # Log eval metrics
            log_entry = {
                "episode": episode,
                "type": "eval",
                **eval_metrics,
            }
            training_history.append(log_entry)
            
            # Save eval video if enabled
            if config.eval_save_video and eval_dir is not None:
                try:
                    from visualize.renderer import create_animation
                    # Create eval directory lazily (only when saving first video)
                    eval_dir.mkdir(parents=True, exist_ok=True)
                    video_path = eval_dir / f"eval_ep{episode}.gif"
                    create_animation(
                        eval_states, params,
                        save_path=str(video_path),
                        fps=config.eval_video_fps,
                        show=False,
                    )
                    if verbose:
                        print(f"  [EVAL] Saved video: {video_path}")
                except ImportError as e:
                    print(f"  [EVAL] Could not save video (matplotlib not available): {e}")
                except Exception as e:
                    print(f"  [EVAL] Error saving video: {e}")
    
    # Final save
    total_time = time.time() - training_start
    print("\n" + "=" * 60)
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best episode reward: {training_state.best_reward:.3f}")
    print("=" * 60)
    
    # Save final training log
    import json
    with open(log_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training log saved to: {log_file}")
    
    save_checkpoint(
        training_state, maddpg, config,
        checkpoint_dir / "final_checkpoint.pkl"
    )
    
    return training_state


def save_checkpoint(
    training_state: TrainingState,
    maddpg: MADDPG,
    config: AssemblyTrainConfig,
    path: Path,
) -> None:
    """Save training checkpoint.
    
    Args:
        training_state: Current training state
        maddpg: MADDPG instance
        config: Training configuration
        path: Path to save checkpoint
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "config": config._asdict(),
        "episode": training_state.episode,
        "total_steps": training_state.total_steps,
        "best_reward": training_state.best_reward,
        "noise_scale": float(training_state.maddpg_state.noise_scale),
        # Save agent parameters
        "agent_params": [
            {
                "actor_params": agent_state.actor_params,
                "critic_params": agent_state.critic_params,
                "target_actor_params": agent_state.target_actor_params,
                "target_critic_params": agent_state.target_critic_params,
            }
            for agent_state in training_state.maddpg_state.agent_states
        ],
    }
    
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    
    print(f"Saved checkpoint: {path}")


def load_checkpoint(
    path: Path,
    config: Optional[AssemblyTrainConfig] = None,
) -> Tuple[TrainingState, AssemblyTrainConfig]:
    """Load training checkpoint.
    
    Args:
        path: Path to checkpoint file
        config: Override config (uses saved config if None)
        
    Returns:
        training_state: Restored training state
        config: Configuration
    """
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    
    if config is None:
        config = AssemblyTrainConfig(**checkpoint["config"])
    
    # Re-create training state
    key = random.PRNGKey(config.seed)
    training_state, env, maddpg, params, vec_reset, vec_step = create_training_state(config, key)
    
    # Restore agent parameters
    agent_states = []
    for i, agent_params in enumerate(checkpoint["agent_params"]):
        agent_state = training_state.maddpg_state.agent_states[i]
        agent_state = agent_state.replace(
            actor_params=agent_params["actor_params"],
            critic_params=agent_params["critic_params"],
            target_actor_params=agent_params["target_actor_params"],
            target_critic_params=agent_params["target_critic_params"],
        )
        agent_states.append(agent_state)
    
    maddpg_state = training_state.maddpg_state.replace(
        agent_states=agent_states,
        noise_scale=jnp.array(checkpoint["noise_scale"]),
    )
    
    training_state = training_state.replace(
        maddpg_state=maddpg_state,
        episode=checkpoint["episode"],
        total_steps=checkpoint["total_steps"],
        best_reward=checkpoint["best_reward"],
    )
    
    print(f"Loaded checkpoint from episode {checkpoint['episode']}")
    
    return training_state, config
