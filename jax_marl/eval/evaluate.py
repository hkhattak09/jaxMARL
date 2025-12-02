#!/usr/bin/env python3
"""Evaluation script for Assembly Swarm Environment.

Evaluates trained models on all 8 shapes WITHOUT any transformations:
- No rotation
- No scaling  
- No position shifting

This provides a consistent benchmark to compare model performance across
different training runs.

Usage:
    python -m jax_marl.eval.evaluate --checkpoint path/to/checkpoint.pkl
    
    # Or from Python:
    from jax_marl.eval import evaluate_on_all_shapes, EvalConfig
    
    config = EvalConfig(checkpoint_path="path/to/checkpoint.pkl")
    results = evaluate_on_all_shapes(config)
"""

import os
import sys
import time
import json
import pickle
from typing import Tuple, Dict, Any, Optional, List, NamedTuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "jax_cus_gym"))

# Configuration imports
from cfg import (
    AssemblyTrainConfig,
    get_shape_file_path,
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
)
from shape_loader import load_shapes_from_pickle, ShapeLibrary
from observations import compute_observation_dim


# ============================================================================
# Evaluation Configuration
# ============================================================================

@dataclass
class EvalConfig:
    """Configuration for evaluation.
    
    Attributes:
        checkpoint_path: Path to trained model checkpoint
        shape_file: Path to shapes pickle (None = use default)
        output_dir: Directory to save results (None = auto)
        n_episodes_per_shape: Number of evaluation episodes per shape
        max_steps: Max steps per episode (None = use config default)
        seed: Random seed for evaluation
        save_videos: Whether to save GIF videos
        video_fps: FPS for saved videos
        verbose: Whether to print progress
    """
    checkpoint_path: str
    shape_file: Optional[str] = None
    output_dir: Optional[str] = None
    n_episodes_per_shape: int = 5
    max_steps: Optional[int] = None
    seed: int = 42
    save_videos: bool = True
    video_fps: int = 10
    verbose: bool = True


@dataclass
class ShapeResult:
    """Results for a single shape evaluation."""
    shape_idx: int
    shape_name: str
    n_episodes: int
    
    # Averaged metrics
    mean_reward: float
    std_reward: float
    mean_coverage: float
    std_coverage: float
    mean_final_coverage: float
    std_final_coverage: float
    mean_collision_rate: float
    std_collision_rate: float
    mean_steps: float
    
    # Per-episode details
    episode_rewards: List[float] = field(default_factory=list)
    episode_coverages: List[float] = field(default_factory=list)
    episode_final_coverages: List[float] = field(default_factory=list)
    episode_collision_rates: List[float] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)


@dataclass
class EvalResult:
    """Complete evaluation results."""
    timestamp: str
    checkpoint_path: str
    n_shapes: int
    n_agents: int
    
    # Aggregated metrics
    overall_mean_reward: float
    overall_mean_coverage: float
    overall_mean_final_coverage: float
    overall_mean_collision_rate: float
    
    # Per-shape results
    shape_results: List[ShapeResult] = field(default_factory=list)
    
    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Shape Names
# ============================================================================

SHAPE_NAMES = [
    "Shape_1",
    "Shape_2", 
    "Shape_3",
    "Shape_4",
    "Shape_5",
    "Shape_6",
    "Shape_7",
    "Shape_8",
]


def get_shape_name(idx: int) -> str:
    """Get human-readable shape name."""
    if idx < len(SHAPE_NAMES):
        return SHAPE_NAMES[idx]
    return f"Shape_{idx + 1}"


# ============================================================================
# Core Evaluation Functions
# ============================================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    shape_library: ShapeLibrary,
) -> Tuple[MADDPG, MADDPGState, AssemblyTrainConfig]:
    """Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        shape_library: Shape library to use
        
    Returns:
        maddpg: MADDPG instance
        maddpg_state: Loaded MADDPG state
        config: Training configuration
    """
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    
    # Reconstruct config
    config = AssemblyTrainConfig(**checkpoint["config"])
    
    # Create environment to get dimensions
    env = AssemblySwarmEnv(n_agents=config.n_agents, shape_library=shape_library)
    params = config_to_assembly_params(config)
    
    # Get observation dimension
    key = random.PRNGKey(0)
    obs, _ = env.reset(key, params)
    obs_dim = obs.shape[-1]
    action_dim = 2
    
    # Create MADDPG
    maddpg_config = config_to_maddpg_config(config, obs_dim, action_dim)
    maddpg = MADDPG(maddpg_config)
    maddpg_state = maddpg.init(key)
    
    # Load agent parameters
    agent_states = []
    for i, agent_params in enumerate(checkpoint["agent_params"]):
        agent_state = maddpg_state.agent_states[i]
        agent_state = agent_state.replace(
            actor_params=agent_params["actor_params"],
            critic_params=agent_params["critic_params"],
            target_actor_params=agent_params["target_actor_params"],
            target_critic_params=agent_params["target_critic_params"],
        )
        agent_states.append(agent_state)
    
    maddpg_state = maddpg_state.replace(agent_states=agent_states)
    
    return maddpg, maddpg_state, config


def create_eval_params(config: AssemblyTrainConfig, max_steps: Optional[int] = None) -> AssemblyParams:
    """Create evaluation parameters with NO randomization.
    
    Args:
        config: Training configuration
        max_steps: Override max steps (None = use config)
        
    Returns:
        AssemblyParams with all randomization disabled
    """
    base_params = config_to_assembly_params(config)
    
    # Disable ALL randomization for consistent evaluation
    eval_params = AssemblyParams(
        # Arena
        arena_size=base_params.arena_size,
        
        # Agent
        agent_radius=base_params.agent_radius,
        max_velocity=base_params.max_velocity,
        max_acceleration=base_params.max_acceleration,
        
        # Observation
        k_neighbors=base_params.k_neighbors,
        d_sen=base_params.d_sen,
        
        # Sub-params
        physics=base_params.physics,
        obs_params=base_params.obs_params,
        reward_params=base_params.reward_params,
        
        # Episode
        max_steps=max_steps if max_steps else base_params.max_steps,
        dt=base_params.dt,
        
        # DISABLE ALL RANDOMIZATION
        randomize_shape=False,      # We select shape explicitly
        randomize_rotation=False,   # No rotation
        randomize_scale=False,      # No scaling
        randomize_offset=False,     # No position shift
        
        # Other
        traj_len=base_params.traj_len,
        reward_mode=base_params.reward_mode,
    )
    
    return eval_params


def run_eval_episode_for_shape(
    env: AssemblySwarmEnv,
    maddpg: MADDPG,
    maddpg_state: MADDPGState,
    params: AssemblyParams,
    shape_idx: int,
    key: jax.Array,
    config: AssemblyTrainConfig,
) -> Tuple[Dict[str, Any], List[AssemblyState]]:
    """Run a single evaluation episode on a specific shape.
    
    The shape is loaded directly without any transformation.
    
    Args:
        env: Environment instance
        maddpg: MADDPG instance
        maddpg_state: MADDPG state (not modified)
        params: Evaluation parameters (no randomization)
        shape_idx: Index of shape to evaluate on
        key: JAX random key
        config: Training configuration
        
    Returns:
        metrics: Episode metrics
        states: List of states for visualization
    """
    from shape_loader import get_shape_from_library, apply_shape_transform
    
    # Get shape without any transformation
    base_grid, base_l_cell, base_mask = get_shape_from_library(
        env.shape_library, shape_idx
    )
    
    # Apply identity transform (no rotation, scale=1, offset=0)
    grid_centers = apply_shape_transform(
        base_grid, base_mask, 
        rotation=0.0, 
        scale=1.0, 
        offset=jnp.zeros(2)
    )
    l_cell = base_l_cell
    
    # Initialize agent positions randomly (but shape is fixed)
    key, pos_key = random.split(key)
    half_size = params.arena_size / 2
    positions = random.uniform(
        pos_key,
        shape=(config.n_agents, 2),
        minval=-half_size * 0.9,
        maxval=half_size * 0.9,
    )
    velocities = jnp.zeros((config.n_agents, 2))
    
    # Create initial state with fixed shape
    trajectory = jnp.zeros((params.traj_len, config.n_agents, 2))
    
    env_state = AssemblyState(
        positions=positions,
        velocities=velocities,
        grid_centers=grid_centers,
        grid_mask=base_mask,
        l_cell=l_cell,
        time=0.0,
        step_count=0,
        done=False,
        trajectory=trajectory,
        traj_idx=0,
        shape_idx=shape_idx,
        shape_rotation=0.0,
        shape_scale=1.0,
        shape_offset=jnp.zeros(2),
        occupied_mask=jnp.zeros(env.shape_library.max_n_grid, dtype=bool),
        in_target=jnp.zeros(config.n_agents, dtype=bool),
        is_colliding=jnp.zeros(config.n_agents, dtype=bool),
    )
    
    # Get initial observation
    obs = env.get_obs(env_state, params)
    
    # Collect states for visualization
    states = [env_state]
    
    # Episode accumulators
    episode_reward = 0.0
    step_rewards = []
    coverages = []
    collisions = []
    
    for step in range(params.max_steps):
        key, action_key, step_key = random.split(key, 3)
        
        # Select actions WITHOUT exploration noise
        obs_list = [obs[i] for i in range(config.n_agents)]
        actions, _, _ = maddpg.select_actions(
            action_key, maddpg_state, obs_list, explore=False
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
        "episode_reward": episode_reward,
        "mean_coverage": np.mean(coverages),
        "final_coverage": coverages[-1] if coverages else 0.0,
        "collision_rate": np.mean(collisions),
        "n_steps": step + 1,
    }
    
    return metrics, states


def evaluate_single_shape(
    env: AssemblySwarmEnv,
    maddpg: MADDPG,
    maddpg_state: MADDPGState,
    params: AssemblyParams,
    shape_idx: int,
    config: AssemblyTrainConfig,
    eval_config: EvalConfig,
    key: jax.Array,
) -> ShapeResult:
    """Evaluate model on a single shape over multiple episodes.
    
    Args:
        env: Environment instance
        maddpg: MADDPG instance
        maddpg_state: MADDPG state
        params: Evaluation parameters
        shape_idx: Shape index to evaluate
        config: Training configuration
        eval_config: Evaluation configuration
        key: Random key
        
    Returns:
        ShapeResult with aggregated metrics
    """
    episode_rewards = []
    episode_coverages = []
    episode_final_coverages = []
    episode_collision_rates = []
    episode_steps = []
    
    all_states = []  # For video saving
    
    for ep in range(eval_config.n_episodes_per_shape):
        key, ep_key = random.split(key)
        
        metrics, states = run_eval_episode_for_shape(
            env, maddpg, maddpg_state, params, shape_idx, ep_key, config
        )
        
        episode_rewards.append(metrics["episode_reward"])
        episode_coverages.append(metrics["mean_coverage"])
        episode_final_coverages.append(metrics["final_coverage"])
        episode_collision_rates.append(metrics["collision_rate"])
        episode_steps.append(metrics["n_steps"])
        
        # Save first episode states for video
        if ep == 0:
            all_states = states
    
    # Save video if requested
    if eval_config.save_videos and all_states and eval_config.output_dir:
        save_eval_video(
            states=all_states,
            params=params,
            shape_idx=shape_idx,
            output_dir=eval_config.output_dir,
            fps=eval_config.video_fps,
        )
    
    return ShapeResult(
        shape_idx=shape_idx,
        shape_name=get_shape_name(shape_idx),
        n_episodes=eval_config.n_episodes_per_shape,
        mean_reward=float(np.mean(episode_rewards)),
        std_reward=float(np.std(episode_rewards)),
        mean_coverage=float(np.mean(episode_coverages)),
        std_coverage=float(np.std(episode_coverages)),
        mean_final_coverage=float(np.mean(episode_final_coverages)),
        std_final_coverage=float(np.std(episode_final_coverages)),
        mean_collision_rate=float(np.mean(episode_collision_rates)),
        std_collision_rate=float(np.std(episode_collision_rates)),
        mean_steps=float(np.mean(episode_steps)),
        episode_rewards=episode_rewards,
        episode_coverages=episode_coverages,
        episode_final_coverages=episode_final_coverages,
        episode_collision_rates=episode_collision_rates,
        episode_steps=episode_steps,
    )


def evaluate_on_all_shapes(eval_config: EvalConfig) -> EvalResult:
    """Evaluate model on all shapes without any transformations.
    
    Args:
        eval_config: Evaluation configuration
        
    Returns:
        EvalResult with complete evaluation metrics
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Set up output directory
    if eval_config.output_dir is None:
        checkpoint_dir = Path(eval_config.checkpoint_path).parent
        output_dir = checkpoint_dir / "eval" / timestamp
    else:
        output_dir = Path(eval_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update eval_config with output_dir
    eval_config.output_dir = str(output_dir)
    
    if eval_config.verbose:
        print(f"=" * 60)
        print(f"Evaluation: Fixed Shapes (No Transformations)")
        print(f"=" * 60)
        print(f"Checkpoint: {eval_config.checkpoint_path}")
        print(f"Output dir: {output_dir}")
        print(f"Episodes per shape: {eval_config.n_episodes_per_shape}")
        print()
    
    # Load shapes
    if eval_config.shape_file:
        shape_file = eval_config.shape_file
    else:
        # Try to find default shape file
        current_dir = Path(__file__).parent
        shape_file = str(current_dir.parent.parent.parent / "fig" / "results.pkl")
    
    if eval_config.verbose:
        print(f"Loading shapes from: {shape_file}")
    
    shape_library = load_shapes_from_pickle(shape_file)
    n_shapes = shape_library.n_shapes
    
    if eval_config.verbose:
        print(f"Found {n_shapes} shapes")
        print()
    
    # Load model
    maddpg, maddpg_state, config = load_model_from_checkpoint(
        eval_config.checkpoint_path, shape_library
    )
    
    if eval_config.verbose:
        print(f"Loaded model with {config.n_agents} agents")
        print()
    
    # Create environment and params
    env = AssemblySwarmEnv(n_agents=config.n_agents, shape_library=shape_library)
    params = create_eval_params(config, eval_config.max_steps)
    
    # Initialize random key
    key = random.PRNGKey(eval_config.seed)
    
    # Evaluate each shape
    shape_results = []
    
    for shape_idx in range(n_shapes):
        key, shape_key = random.split(key)
        
        if eval_config.verbose:
            print(f"Evaluating {get_shape_name(shape_idx)} ({shape_idx + 1}/{n_shapes})...")
        
        result = evaluate_single_shape(
            env=env,
            maddpg=maddpg,
            maddpg_state=maddpg_state,
            params=params,
            shape_idx=shape_idx,
            config=config,
            eval_config=eval_config,
            key=shape_key,
        )
        
        shape_results.append(result)
        
        if eval_config.verbose:
            print(f"  Reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}")
            print(f"  Final Coverage: {result.mean_final_coverage:.2%} ± {result.std_final_coverage:.2%}")
            print(f"  Collision Rate: {result.mean_collision_rate:.2%}")
            print()
    
    # Compute overall metrics
    overall_mean_reward = float(np.mean([r.mean_reward for r in shape_results]))
    overall_mean_coverage = float(np.mean([r.mean_coverage for r in shape_results]))
    overall_mean_final_coverage = float(np.mean([r.mean_final_coverage for r in shape_results]))
    overall_mean_collision_rate = float(np.mean([r.mean_collision_rate for r in shape_results]))
    
    # Create result
    eval_result = EvalResult(
        timestamp=timestamp,
        checkpoint_path=eval_config.checkpoint_path,
        n_shapes=n_shapes,
        n_agents=config.n_agents,
        overall_mean_reward=overall_mean_reward,
        overall_mean_coverage=overall_mean_coverage,
        overall_mean_final_coverage=overall_mean_final_coverage,
        overall_mean_collision_rate=overall_mean_collision_rate,
        shape_results=shape_results,
        config=config._asdict(),
    )
    
    # Save results
    save_eval_results(eval_result, output_dir)
    
    if eval_config.verbose:
        print(f"=" * 60)
        print(f"Overall Results (averaged across {n_shapes} shapes):")
        print(f"=" * 60)
        print(f"  Mean Reward:        {overall_mean_reward:.2f}")
        print(f"  Mean Coverage:      {overall_mean_coverage:.2%}")
        print(f"  Mean Final Coverage:{overall_mean_final_coverage:.2%}")
        print(f"  Mean Collision Rate:{overall_mean_collision_rate:.2%}")
        print()
        print(f"Results saved to: {output_dir}")
    
    return eval_result


# ============================================================================
# Saving Functions
# ============================================================================

def save_eval_results(result: EvalResult, output_dir: Path) -> None:
    """Save evaluation results to JSON and pickle."""
    output_dir = Path(output_dir)
    
    # Save as JSON (human-readable)
    json_path = output_dir / "eval_results.json"
    
    # Convert to JSON-serializable format
    json_data = {
        "timestamp": result.timestamp,
        "checkpoint_path": result.checkpoint_path,
        "n_shapes": result.n_shapes,
        "n_agents": result.n_agents,
        "overall_metrics": {
            "mean_reward": result.overall_mean_reward,
            "mean_coverage": result.overall_mean_coverage,
            "mean_final_coverage": result.overall_mean_final_coverage,
            "mean_collision_rate": result.overall_mean_collision_rate,
        },
        "per_shape_results": [
            {
                "shape_idx": r.shape_idx,
                "shape_name": r.shape_name,
                "n_episodes": r.n_episodes,
                "mean_reward": r.mean_reward,
                "std_reward": r.std_reward,
                "mean_coverage": r.mean_coverage,
                "std_coverage": r.std_coverage,
                "mean_final_coverage": r.mean_final_coverage,
                "std_final_coverage": r.std_final_coverage,
                "mean_collision_rate": r.mean_collision_rate,
                "std_collision_rate": r.std_collision_rate,
                "mean_steps": r.mean_steps,
            }
            for r in result.shape_results
        ],
        "config": result.config,
    }
    
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    # Save full results as pickle (includes per-episode data)
    pkl_path = output_dir / "eval_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)


def save_eval_video(
    states: List[AssemblyState],
    params: AssemblyParams,
    shape_idx: int,
    output_dir: str,
    fps: int = 10,
) -> None:
    """Save evaluation episode as GIF video."""
    try:
        from visualize.renderer import create_animation
        
        output_path = Path(output_dir) / f"eval_shape_{shape_idx}.gif"
        
        # create_animation handles saving when save_path is provided
        create_animation(
            states=states,
            params=params,
            save_path=str(output_path),
            fps=fps,
            show=False,
            show_trajectories=True,
            show_grid=True,
        )
        
    except ImportError:
        print(f"Warning: Could not import visualize module, skipping video for shape {shape_idx}")
    except Exception as e:
        print(f"Warning: Could not save video for shape {shape_idx}: {e}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Command-line entry point for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on all shapes without transformations"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--shape-file", "-s",
        type=str,
        default=None,
        help="Path to shapes pickle file (default: use jaxMARL/fig/results.pkl)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for results (default: checkpoint_dir/eval/timestamp)"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=5,
        help="Number of episodes per shape (default: 5)"
    )
    parser.add_argument(
        "--max-steps", "-m",
        type=int,
        default=None,
        help="Max steps per episode (default: use training config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Disable video saving"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Video FPS (default: 10)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    eval_config = EvalConfig(
        checkpoint_path=args.checkpoint,
        shape_file=args.shape_file,
        output_dir=args.output_dir,
        n_episodes_per_shape=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        save_videos=not args.no_videos,
        video_fps=args.fps,
        verbose=not args.quiet,
    )
    
    result = evaluate_on_all_shapes(eval_config)
    
    return result


if __name__ == "__main__":
    main()
