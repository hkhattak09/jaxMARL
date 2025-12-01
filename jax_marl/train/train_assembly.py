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
    
    # Get observation/action dimensions
    obs_dim = compute_observation_dim(params.obs_params)
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


def run_episode(
    env: AssemblySwarmEnv,
    maddpg: MADDPG,
    training_state: TrainingState,
    params: AssemblyParams,
    config: AssemblyTrainConfig,
    vec_reset: callable,
    vec_step: callable,
    explore: bool = True,
) -> Tuple[TrainingState, Dict[str, Any]]:
    """Run a single episode across all parallel environments.
    
    Args:
        env: Environment instance
        maddpg: MADDPG instance
        training_state: Current training state
        params: Environment parameters
        config: Training configuration
        vec_reset: Vectorized reset function
        vec_step: Vectorized step function
        explore: Whether to use exploration noise
        
    Returns:
        new_training_state: Updated training state
        metrics: Episode metrics (averaged across parallel envs)
    """
    key = training_state.key
    maddpg_state = training_state.maddpg_state
    n_envs = config.n_parallel_envs
    
    # Reset all parallel environments
    key, *reset_keys = random.split(key, n_envs + 1)
    reset_keys = jnp.stack(reset_keys)
    obs_batch, env_states = vec_reset(reset_keys)  # (n_envs, n_agents, obs_dim)
    
    # Episode accumulators (per env)
    episode_rewards = jnp.zeros(n_envs)
    all_step_rewards = []
    all_coverages = []
    all_collisions = []
    
    step_start = time.time()
    
    for step in range(config.max_steps):
        key, action_key = random.split(key)
        step_keys = random.split(key, n_envs)
        
        # For each parallel env, select actions for all agents
        # obs_batch shape: (n_envs, n_agents, obs_dim)
        # We'll process all agents across all envs at once
        
        # Flatten across envs for action selection
        # obs_flat: (n_envs * n_agents, obs_dim) - but MADDPG expects list per agent
        # For now, process each env sequentially (can be optimized later)
        
        all_actions = []
        for env_idx in range(n_envs):
            obs_list = [obs_batch[env_idx, i] for i in range(config.n_agents)]
            actions, log_probs, maddpg_state = maddpg.select_actions(
                random.fold_in(action_key, env_idx), maddpg_state, obs_list, explore=explore
            )
            all_actions.append(jnp.stack(actions))  # (n_agents, action_dim)
        
        actions_batch = jnp.stack(all_actions)  # (n_envs, n_agents, action_dim)
        
        # Step all environments in parallel
        next_obs_batch, env_states, rewards_batch, dones_batch, info_batch = vec_step(
            step_keys, env_states, actions_batch
        )
        # rewards_batch: (n_envs, n_agents)
        # dones_batch: (n_envs, n_agents)
        
        # Store transitions from all parallel envs
        for env_idx in range(n_envs):
            obs_list = [obs_batch[env_idx, i] for i in range(config.n_agents)]
            next_obs_list = [next_obs_batch[env_idx, i] for i in range(config.n_agents)]
            actions_list = [actions_batch[env_idx, i] for i in range(config.n_agents)]
            rewards_list = [rewards_batch[env_idx, i:i+1] for i in range(config.n_agents)]
            dones_list = [dones_batch[env_idx, i:i+1] for i in range(config.n_agents)]
            
            # Compute prior actions for regularization if needed
            if config.prior_weight > 0:
                prior_actions = compute_prior_policy(
                    env_states.positions[env_idx],
                    env_states.velocities[env_idx],
                    env_states.grid_centers[env_idx],
                    env_states.grid_mask[env_idx],
                    env_states.l_cell[env_idx],
                    params.reward_params.collision_threshold,
                    params.d_sen,
                )
                prior_list = [prior_actions[i] for i in range(config.n_agents)]
            else:
                prior_list = None
            
            maddpg_state = maddpg.store_transition(
                maddpg_state,
                obs_list,
                actions_list,
                rewards_list,
                next_obs_list,
                dones_list,
                action_priors=prior_list,
            )
        
        # Update observations for next step
        obs_batch = next_obs_batch
        
        # Accumulate metrics across all parallel envs
        step_reward = jnp.mean(rewards_batch)  # Mean across envs and agents
        episode_rewards = episode_rewards + jnp.mean(rewards_batch, axis=1)  # Per env
        all_step_rewards.append(float(step_reward))
        
        # Coverage and collision rates (average across parallel envs)
        coverage = jnp.mean(jnp.array([info_batch["coverage_rate"]]))
        collision = jnp.mean(env_states.is_colliding)
        all_coverages.append(float(coverage))
        all_collisions.append(float(collision))
        
        # Check if all envs are done
        if jnp.all(dones_batch[:, 0]):
            break
    
    step_time = time.time() - step_start
    
    # Update step counters (multiply by n_envs since we ran parallel envs)
    total_steps = training_state.total_steps + (step + 1) * n_envs
    
    # Calculate mean episode reward across parallel envs
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
        "episode_reward_mean": np.mean(all_step_rewards),
        "episode_reward_std": np.std(all_step_rewards),
        "coverage_rate": np.mean(all_coverages),
        "collision_rate": np.mean(all_collisions),
        "step_time": step_time,
        "noise_scale": float(noise_scale),
        "steps_in_episode": step + 1,
        "n_parallel_envs": n_envs,
        "total_transitions": (step + 1) * n_envs,
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


def train_step(
    maddpg: MADDPG,
    maddpg_state: MADDPGState,
    key: jax.Array,
    config: AssemblyTrainConfig,
) -> Tuple[MADDPGState, Dict[str, float]]:
    """Perform gradient updates.
    
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
        # Run episode across all parallel environments
        episode_start = time.time()
        training_state, episode_metrics = run_episode(
            env, maddpg, training_state, params, config, vec_reset, vec_step, explore=True
        )
        
        # Perform gradient updates
        train_start = time.time()
        key, train_key = random.split(training_state.key)
        maddpg_state, train_info = train_step(
            maddpg, training_state.maddpg_state, train_key, config
        )
        training_state = training_state.replace(
            maddpg_state=maddpg_state,
            key=key,
        )
        train_time = time.time() - train_start
        
        # Logging
        if episode % config.log_interval == 0 and verbose:
            print(
                f"Episode {episode:5d}/{config.n_episodes} | "
                f"Reward: {episode_metrics['episode_reward']:7.3f} | "
                f"Coverage: {episode_metrics['coverage_rate']:.3f} | "
                f"Collisions: {episode_metrics['collision_rate']:.3f} | "
                f"Noise: {episode_metrics['noise_scale']:.3f} | "
                f"Buffer: {train_info['buffer_size']:6d} | "
                f"Envs: {config.n_parallel_envs} | "
                f"Step: {episode_metrics['step_time']:.2f}s | "
                f"Train: {train_time:.2f}s"
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
                "buffer_size": train_info["buffer_size"],
                "step_time": episode_metrics["step_time"],
                "train_time": train_time,
                "n_parallel_envs": config.n_parallel_envs,
                "total_transitions": episode_metrics.get("total_transitions", 0),
            }
            if train_info.get("updated", False):
                log_entry["actor_loss"] = float(train_info["actor_loss"])
                log_entry["critic_loss"] = float(train_info["critic_loss"])
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
