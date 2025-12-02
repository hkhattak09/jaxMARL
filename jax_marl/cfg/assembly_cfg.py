"""Assembly Swarm Environment Configuration for JAX MARL.

Edit the values in AssemblyTrainConfig directly to change training settings.
All parameters are in one place - no need to pass arguments or use presets.

With JAX, you can also run multiple parallel environments using vmap for
faster training (see n_parallel_envs parameter).
"""


# set the training steps to decay to (44% of total training epsiodes)*max_step/epsiode*n_envs in the training config when defining it, it is dependant on number of envs.
# set buffer size for each config it is (1/12 * total epsiodes)*max_steps*n_env, to allow for off policy learning i.e you learn from previosu data as well you dont just forget

from typing import Optional, NamedTuple
from pathlib import Path


class AssemblyTrainConfig(NamedTuple):
    """Training configuration - EDIT VALUES HERE.
    
    Just change the default values below to configure your training run.
    """
    
    # ================== Environment ==================
    n_agents: int = 30                  # Number of agents in swarm
    n_parallel_envs: int = 4            # Number of parallel environments (JAX vmap)
    arena_size: float = 5.0             # Square arena size
    agent_radius: float = 0.1           # Agent collision radius
    max_velocity: float = 0.8           # Max agent speed
    max_acceleration: float = 1.0       # Max acceleration (action scale)
    
    # Observation
    k_neighbors: int = 6                # Number of nearest neighbors to observe
    d_sen: float = 3.0                  # Sensing distance
    include_self_state: bool = True     # Include own position/velocity
    
    # Physics
    dt: float = 0.1                     # Simulation timestep
    
    # Episode
    max_steps: int = 200                # Max steps per episode
    
    # Domain randomization
    randomize_shape: bool = True        # Random target shape each episode
    randomize_rotation: bool = True     # Random rotation
    randomize_scale: bool = True        # Random scaling
    randomize_offset: bool = True       # Random translation
    
    # Reward
    reward_mode: str = "individual"     # "individual", "shared_mean", "shared_max"
    
    # ================== Algorithm (MADDPG) ==================
    hidden_dim: int = 256               # Neural network hidden layer size
    lr_actor: float = 1e-4              # Actor learning rate
    lr_critic: float = 1e-3             # Critic learning rate
    gamma: float = 0.95                 # Discount factor
    tau: float = 0.01                   # Soft target update rate
    
    buffer_size: int = 50000            # Replay buffer capacity
    batch_size: int = 2048              # Training batch size
    warmup_steps: int = 5000            # Steps before training starts
    
    noise_scale_initial: float = 0.9    # Initial exploration noise
    noise_scale_final: float = 0.5      # Final exploration noise
    noise_decay_steps: int = 100000     # Steps to decay noise
    
    update_every: int = 100             # Steps between gradient updates
    updates_per_step: int = 30          # Gradient updates per training step
    
    prior_weight: float = 0.3           # LLM prior regularization (0 = disabled)
    
    # ================== TD3 Enhancements ==================
    use_td3: bool = True                # Enable TD3 (twin critics, delayed updates, smoothing)
    policy_delay: int = 2               # Update actor every N critic updates
    target_noise: float = 0.2           # Stddev of noise for target smoothing
    target_noise_clip: float = 0.5      # Clip range for target noise
    
    # ================== Training ==================
    seed: int = 226                     # Random seed
    n_episodes: int = 3000              # Total training episodes
    
    # Logging
    log_interval: int = 10              # Log every N episodes
    save_interval: int = 100            # Save checkpoint every N episodes
    
    # Evaluation
    eval_interval: int = 50             # Evaluate every N episodes (0 = disabled)
    eval_save_video: bool = True        # Save GIF of eval episodes
    eval_video_fps: int = 10            # FPS for eval videos
    
    # Paths (None = use defaults in jaxMARL/fig/, jax_marl/checkpoints/, etc.)
    shape_file: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None
    eval_dir: Optional[str] = None      # Directory for eval videos


# ============================================================================
# Example Configs (commented out - uncomment to use, or just edit above)
# ============================================================================

config = None

# # -------------------- FAST DEBUG CONFIG --------------------
# # Use this for quick testing and debugging
# config = AssemblyTrainConfig(
#     # Environment
#     n_agents=5,
#     n_parallel_envs=2,
#     arena_size=5.0,
#     agent_radius=0.1,
#     max_velocity=0.8,
#     max_acceleration=1.0,
#     # Observation
#     k_neighbors=4,
#     d_sen=3.0,
#     include_self_state=True,
#     # Physics
#     dt=0.1,
#     # Episode
#     max_steps=50,
#     # Domain randomization
#     randomize_shape=True,
#     randomize_rotation=True,
#     randomize_scale=True,
#     randomize_offset=True,
#     # Reward
#     reward_mode="individual",
#     # Algorithm
#     hidden_dim=128,
#     lr_actor=1e-4,
#     lr_critic=1e-3,
#     gamma=0.95,
#     tau=0.01,
#     buffer_size=833,
#     batch_size=64,
#     warmup_steps=100,
#     noise_scale_initial=0.9,
#     noise_scale_final=0.1,
#     noise_decay_steps=4400,
#     update_every=50,
#     updates_per_step=5,
#     prior_weight=0.3,
#     # Training
#     seed=226,
#     n_episodes=100,
#     log_interval=1,
#     save_interval=5,
#     # Evaluation
#     eval_interval=10,
#     eval_save_video=True,
#     eval_video_fps=10,
#     # Paths
#     shape_file=None,
#     checkpoint_dir=None,
#     log_dir=None,
#     eval_dir=None,
# )

# -------------------- SMALL SCALE EXPERIMENT --------------------
# Use this for quick experiments with fewer agents
# config = AssemblyTrainConfig(
#     # Environment
#     n_agents=10,
#     n_parallel_envs=4,
#     arena_size=5.0,
#     agent_radius=0.1,
#     max_velocity=0.8,
#     max_acceleration=1.0,
#     # Observation
#     k_neighbors=6,
#     d_sen=3.0,
#     include_self_state=True,
#     # Physics
#     dt=0.1,
#     # Episode
#     max_steps=150,
#     # Domain randomization
#     randomize_shape=True,
#     randomize_rotation=True,
#     randomize_scale=True,
#     randomize_offset=True,
#     # Reward
#     reward_mode="individual",
#     # Algorithm
#     hidden_dim=256,
#     lr_actor=1e-4,
#     lr_critic=1e-3,
#     gamma=0.95,
#     tau=0.01,
#     buffer_size=50000,
#     batch_size=512,
#     warmup_steps=2000,
#     noise_scale_initial=0.9,
#     noise_scale_final=0.1,
#     noise_decay_steps=264000,
#     update_every=100,
#     updates_per_step=20,
#     prior_weight=0.3,
#     # Training
#     seed=226,
#     n_episodes=1000,
#     log_interval=1,
#     save_interval=50,
#     eval_interval=25,
#     # Paths
#     shape_file=None,
#     checkpoint_dir=None,
#     log_dir=None,
# )

# # -------------------- FULL SCALE TRAINING --------------------
# # Use this for full training runs
config = AssemblyTrainConfig(
    # Environment
    n_agents=20,
    n_parallel_envs=8,
    arena_size=5.0,
    agent_radius=0.1,
    max_velocity=0.8,
    max_acceleration=1.0,
    # Observation
    k_neighbors=6,
    d_sen=3.0,
    include_self_state=True,
    # Physics
    dt=0.1,
    # Episode
    max_steps=200,
    # Domain randomization
    randomize_shape=True,
    randomize_rotation=True,
    randomize_scale=True,
    randomize_offset=True,
    # Reward
    reward_mode="individual",
    # Algorithm
    hidden_dim=256,
    lr_actor=1e-4,
    lr_critic=1e-3,
    gamma=0.95,
    tau=0.01,
    buffer_size=240000,
    batch_size=2048,
    warmup_steps=50000,
    noise_scale_initial=0.9,
    noise_scale_final=0.5,
    noise_decay_steps=2112000,
    update_every=100,
    updates_per_step=30,
    prior_weight=0.5,
    # Training
    seed=226,
    n_episodes=3000,
    log_interval=10,
    save_interval=100,
    eval_interval=200,
    # Paths
    shape_file=None,
    checkpoint_dir=None,
    log_dir=None,
)


# ============================================================================
# Helper Functions (used by training code - don't modify)
# ============================================================================

def get_config() -> AssemblyTrainConfig:
    """Get the current config. Returns the defaults from the class."""
    if config is not None:
        return config
    else:
        return AssemblyTrainConfig()


def get_shape_file_path(config: AssemblyTrainConfig) -> str:
    """Get path to shape pickle file (default: jaxMARL/fig/results.pkl)."""
    if config.shape_file is not None:
        return config.shape_file
    
    current_dir = Path(__file__).resolve().parent
    jaxmarl_root = current_dir.parent.parent
    return str(jaxmarl_root / "fig" / "results.pkl")


def get_checkpoint_dir(config: AssemblyTrainConfig, run_name: Optional[str] = None) -> str:
    """Get checkpoint directory path."""
    if config.checkpoint_dir is not None:
        base_dir = Path(config.checkpoint_dir)
    else:
        current_dir = Path(__file__).resolve().parent
        base_dir = current_dir.parent / "checkpoints"
    
    if run_name is None:
        from datetime import datetime
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    checkpoint_dir = base_dir / "assembly" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir)


def get_log_dir(config: AssemblyTrainConfig, run_name: Optional[str] = None) -> str:
    """Get log directory path."""
    if config.log_dir is not None:
        base_dir = Path(config.log_dir)
    else:
        current_dir = Path(__file__).resolve().parent
        base_dir = current_dir.parent / "logs"
    
    if run_name is None:
        from datetime import datetime
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    log_dir = base_dir / "assembly" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def get_eval_dir(config: AssemblyTrainConfig, run_name: Optional[str] = None) -> str:
    """Get eval video directory path.
    
    Note: Directory is NOT created here - it will be created lazily when first file is saved.
    """
    if config.eval_dir is not None:
        base_dir = Path(config.eval_dir)
    else:
        current_dir = Path(__file__).resolve().parent
        base_dir = current_dir.parent / "eval_videos"
    
    if run_name is None:
        from datetime import datetime
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    eval_dir = base_dir / "assembly" / run_name
    # Don't create directory here - create lazily when saving files
    return str(eval_dir)


def config_to_maddpg_config(config: AssemblyTrainConfig, obs_dim: int, action_dim: int):
    """Convert to MADDPGConfig for algorithm initialization."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "algo"))
    from maddpg import MADDPGConfig
    
    return MADDPGConfig(
        n_agents=config.n_agents,
        obs_dims=tuple([obs_dim] * config.n_agents),
        action_dims=tuple([action_dim] * config.n_agents),
        hidden_dims=(config.hidden_dim, config.hidden_dim),
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        gamma=config.gamma,
        tau=config.tau,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        warmup_steps=config.warmup_steps,
        noise_scale_initial=config.noise_scale_initial,
        noise_scale_final=config.noise_scale_final,
        noise_decay_steps=config.noise_decay_steps,
        update_every=config.update_every,
        updates_per_step=config.updates_per_step,
        prior_weight=config.prior_weight,
        # TD3 enhancements
        use_td3=config.use_td3,
        policy_delay=config.policy_delay,
        target_noise=config.target_noise,
        target_noise_clip=config.target_noise_clip,
    )


def config_to_assembly_params(config: AssemblyTrainConfig):
    """Convert to AssemblyParams for environment initialization."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "jax_cus_gym"))
    from assembly_env import AssemblyParams
    from observations import ObservationParams
    from rewards import RewardParams
    from physics import PhysicsParams
    
    obs_params = ObservationParams(
        topo_nei_max=config.k_neighbors,
        d_sen=config.d_sen,
        include_self_state=config.include_self_state,
        l_max=config.arena_size / 2,
        vel_max=config.max_velocity,
    )
    
    reward_params = RewardParams(
        reward_mode=config.reward_mode,
        collision_threshold=config.agent_radius * 2,
    )
    
    physics_params = PhysicsParams(
        dt=config.dt,
        agent_radius=config.agent_radius,
        vel_max=config.max_velocity,
    )
    
    return AssemblyParams(
        arena_size=config.arena_size,
        agent_radius=config.agent_radius,
        max_velocity=config.max_velocity,
        max_acceleration=config.max_acceleration,
        k_neighbors=config.k_neighbors,
        d_sen=config.d_sen,
        physics=physics_params,
        obs_params=obs_params,
        reward_params=reward_params,
        max_steps=config.max_steps,
        dt=config.dt,
        randomize_shape=config.randomize_shape,
        randomize_rotation=config.randomize_rotation,
        randomize_scale=config.randomize_scale,
        randomize_offset=config.randomize_offset,
        reward_mode=config.reward_mode,
    )
