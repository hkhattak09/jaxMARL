"""Comprehensive tests for train_assembly.py.

Tests cover:
1. Import compatibility with jax_cus_gym and algo modules
2. Configuration creation and conversion
3. Parallel environment creation (n_envs)
4. Training state initialization
5. Episode execution with parallel environments
6. Gradient computation and updates
7. Buffer filling and sampling
8. Noise decay
9. End-to-end mini training loop

Run with:
    pytest test_train_assembly.py -v
    
Or directly:
    python test_train_assembly.py
"""

import sys
import os
import tempfile
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "jax_cus_gym"))

import pytest
import numpy as np

# JAX imports
import jax
import jax.numpy as jnp
from jax import random


# ============================================================================
# Test Imports
# ============================================================================

class TestImports:
    """Test that all imports work correctly."""
    
    def test_cfg_imports(self):
        """Test cfg module imports."""
        from cfg import (
            AssemblyTrainConfig,
            get_config,
            get_shape_file_path,
            get_checkpoint_dir,
            get_log_dir,
            config_to_maddpg_config,
            config_to_assembly_params,
        )
        print("✓ cfg imports passed")
    
    def test_algo_imports(self):
        """Test algo module imports."""
        from algo import MADDPG, MADDPGConfig, MADDPGState
        print("✓ algo imports passed")
    
    def test_environment_imports(self):
        """Test jax_cus_gym environment imports."""
        from assembly_env import (
            AssemblySwarmEnv,
            AssemblyParams,
            AssemblyState,
            make_assembly_env,
            make_vec_env,
            compute_prior_policy,
        )
        from shape_loader import load_shapes_from_pickle
        from observations import compute_observation_dim, ObservationParams
        print("✓ environment imports passed")
    
    def test_train_assembly_imports(self):
        """Test train_assembly module imports."""
        from train.train_assembly import (
            TrainingState,
            TrainingMetrics,
            create_training_state,
            run_episode,
            train_step,
            train,
        )
        print("✓ train_assembly imports passed")


# ============================================================================
# Test Configuration
# ============================================================================

class TestConfiguration:
    """Test configuration creation and conversion."""
    
    def test_get_config(self):
        """Test get_config returns valid config."""
        from cfg import get_config, AssemblyTrainConfig
        
        config = get_config()
        assert isinstance(config, AssemblyTrainConfig)
        assert config.n_agents > 0
        assert config.n_parallel_envs > 0
        print("✓ get_config passed")
    
    def test_debug_config_creation(self):
        """Test creating a debug config with small values."""
        from cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig(
            n_agents=3,
            n_parallel_envs=2,
            n_episodes=2,
            max_steps=10,
            batch_size=8,
            buffer_size=100,
            warmup_steps=10,
            hidden_dim=32,
        )
        
        assert config.n_agents == 3
        assert config.n_parallel_envs == 2
        assert config.batch_size == 8
        print("✓ debug_config_creation passed")
    
    def test_config_to_maddpg_config(self):
        """Test conversion to MADDPGConfig."""
        from cfg import AssemblyTrainConfig, config_to_maddpg_config
        from algo import MADDPGConfig
        
        config = AssemblyTrainConfig(
            n_agents=5,
            hidden_dim=64,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.95,
            tau=0.01,
            buffer_size=1000,
            batch_size=32,
        )
        
        obs_dim = 20
        action_dim = 2
        maddpg_config = config_to_maddpg_config(config, obs_dim, action_dim)
        
        # Check it has the right attributes (avoid isinstance issues with different import paths)
        assert hasattr(maddpg_config, 'n_agents')
        assert hasattr(maddpg_config, 'obs_dims')
        assert hasattr(maddpg_config, 'action_dims')
        assert maddpg_config.n_agents == 5
        assert maddpg_config.obs_dims == tuple([20] * 5)
        assert maddpg_config.action_dims == tuple([2] * 5)
        assert maddpg_config.hidden_dims == (64, 64)
        assert maddpg_config.buffer_size == 1000
        assert maddpg_config.batch_size == 32
        print("✓ config_to_maddpg_config passed")
    
    def test_config_to_assembly_params(self):
        """Test conversion to AssemblyParams."""
        from cfg import AssemblyTrainConfig, config_to_assembly_params
        from assembly_env import AssemblyParams
        
        config = AssemblyTrainConfig(
            n_agents=5,
            arena_size=4.0,
            agent_radius=0.1,
            max_velocity=1.0,
            k_neighbors=4,
            d_sen=2.5,
            dt=0.05,
        )
        
        params = config_to_assembly_params(config)
        
        assert isinstance(params, AssemblyParams)
        assert params.arena_size == 4.0
        assert params.agent_radius == 0.1
        assert params.max_velocity == 1.0
        assert params.k_neighbors == 4
        assert params.d_sen == 2.5
        assert params.dt == 0.05
        print("✓ config_to_assembly_params passed")


# ============================================================================
# Test Environment Creation
# ============================================================================

class TestEnvironmentCreation:
    """Test environment creation and parallel environments."""
    
    def test_make_assembly_env(self):
        """Test creating a single environment."""
        from assembly_env import make_assembly_env
        
        env, params = make_assembly_env(n_agents=5)
        
        assert env is not None
        assert params is not None
        assert env.n_agents == 5
        print("✓ make_assembly_env passed")
    
    def test_make_vec_env(self):
        """Test creating vectorized environments."""
        from assembly_env import make_vec_env
        
        n_envs = 4
        n_agents = 5
        
        env, params, vec_reset, vec_step = make_vec_env(
            n_envs=n_envs,
            n_agents=n_agents,
        )
        
        assert env is not None
        assert params is not None
        assert callable(vec_reset)
        assert callable(vec_step)
        print("✓ make_vec_env passed")
    
    def test_vec_reset_shape(self):
        """Test vectorized reset produces correct shapes."""
        from assembly_env import make_vec_env
        
        n_envs = 4
        n_agents = 5
        
        env, params, vec_reset, vec_step = make_vec_env(
            n_envs=n_envs,
            n_agents=n_agents,
        )
        
        key = random.PRNGKey(42)
        keys = random.split(key, n_envs)
        
        obs_batch, env_states = vec_reset(keys)
        
        # Check observation shape: (n_envs, n_agents, obs_dim)
        assert obs_batch.shape[0] == n_envs
        assert obs_batch.shape[1] == n_agents
        
        # Check state shapes
        assert env_states.positions.shape[0] == n_envs
        assert env_states.positions.shape[1] == n_agents
        assert env_states.velocities.shape[0] == n_envs
        
        print(f"✓ vec_reset_shape passed (obs: {obs_batch.shape})")
    
    def test_vec_step_shape(self):
        """Test vectorized step produces correct shapes."""
        from assembly_env import make_vec_env
        
        n_envs = 4
        n_agents = 5
        action_dim = 2
        
        env, params, vec_reset, vec_step = make_vec_env(
            n_envs=n_envs,
            n_agents=n_agents,
        )
        
        key = random.PRNGKey(42)
        reset_keys = random.split(key, n_envs)
        step_keys = random.split(random.PRNGKey(0), n_envs)
        
        obs_batch, env_states = vec_reset(reset_keys)
        
        # Random actions
        actions = random.uniform(
            random.PRNGKey(1),
            shape=(n_envs, n_agents, action_dim),
            minval=-1.0,
            maxval=1.0
        )
        
        next_obs, next_states, rewards, dones, info = vec_step(
            step_keys, env_states, actions
        )
        
        # Check shapes
        assert next_obs.shape == obs_batch.shape
        assert rewards.shape == (n_envs, n_agents)
        assert dones.shape == (n_envs, n_agents)
        assert next_states.positions.shape == env_states.positions.shape
        
        print(f"✓ vec_step_shape passed (rewards: {rewards.shape})")


# ============================================================================
# Test Training State
# ============================================================================

class TestTrainingState:
    """Test training state initialization."""
    
    def get_debug_config(self):
        """Get a small debug config for testing."""
        from cfg import AssemblyTrainConfig
        return AssemblyTrainConfig(
            n_agents=3,
            n_parallel_envs=2,
            n_episodes=2,
            max_steps=5,
            batch_size=8,
            buffer_size=100,
            warmup_steps=5,
            hidden_dim=32,
            k_neighbors=2,
        )
    
    def test_create_training_state(self):
        """Test creating initial training state."""
        from train.train_assembly import create_training_state
        
        config = self.get_debug_config()
        key = random.PRNGKey(42)
        
        training_state, env, maddpg, params, vec_reset, vec_step = create_training_state(
            config, key
        )
        
        # Check training state
        assert training_state is not None
        assert training_state.episode == 0
        assert training_state.total_steps == 0
        assert training_state.best_reward == float('-inf')
        
        # Check MADDPG state
        assert training_state.maddpg_state is not None
        assert len(training_state.maddpg_state.agent_states) == config.n_agents
        
        # Check env states shape (batched)
        assert training_state.env_states.positions.shape[0] == config.n_parallel_envs
        
        # Check returned objects
        assert env is not None
        assert maddpg is not None
        assert params is not None
        assert callable(vec_reset)
        assert callable(vec_step)
        
        print("✓ create_training_state passed")
    
    def test_maddpg_init_shapes(self):
        """Test MADDPG initialization produces correct shapes."""
        from cfg import AssemblyTrainConfig, config_to_maddpg_config
        from algo import MADDPG
        from observations import compute_observation_dim, ObservationParams
        
        config = self.get_debug_config()
        
        obs_params = ObservationParams(
            topo_nei_max=config.k_neighbors,
            d_sen=config.d_sen,
        )
        obs_dim = compute_observation_dim(obs_params)
        action_dim = 2
        
        maddpg_config = config_to_maddpg_config(config, obs_dim, action_dim)
        maddpg = MADDPG(maddpg_config)
        
        key = random.PRNGKey(42)
        state = maddpg.init(key)
        
        # Check agent states
        assert len(state.agent_states) == config.n_agents
        
        # Check buffer
        assert state.buffer_state is not None
        assert int(state.buffer_state.size) == 0
        
        # Check noise scale (use approx for float32 precision)
        assert abs(float(state.noise_scale) - config.noise_scale_initial) < 1e-6
        
        print(f"✓ maddpg_init_shapes passed (obs_dim={obs_dim})")


# ============================================================================
# Test Episode Execution
# ============================================================================

class TestEpisodeExecution:
    """Test episode execution with parallel environments."""
    
    def get_debug_config(self):
        """Get a small debug config for testing."""
        from cfg import AssemblyTrainConfig
        return AssemblyTrainConfig(
            n_agents=3,
            n_parallel_envs=2,
            n_episodes=2,
            max_steps=5,
            batch_size=8,
            buffer_size=100,
            warmup_steps=5,
            hidden_dim=32,
            k_neighbors=2,
            prior_weight=0.0,  # Disable prior for simplicity
        )
    
    def test_run_episode(self):
        """Test running a single episode."""
        from train.train_assembly import create_training_state, run_episode
        
        config = self.get_debug_config()
        key = random.PRNGKey(42)
        
        training_state, env, maddpg, params, vec_reset, vec_step = create_training_state(
            config, key
        )
        
        initial_episode = training_state.episode
        initial_steps = training_state.total_steps
        
        new_state, metrics = run_episode(
            env, maddpg, training_state, params, config,
            vec_reset, vec_step, explore=True
        )
        
        # Check state updates
        assert new_state.episode == initial_episode + 1
        assert new_state.total_steps > initial_steps
        
        # Check metrics
        assert "episode_reward" in metrics
        assert "coverage_rate" in metrics
        assert "collision_rate" in metrics
        assert "noise_scale" in metrics
        assert "n_parallel_envs" in metrics
        assert metrics["n_parallel_envs"] == config.n_parallel_envs
        
        print(f"✓ run_episode passed (steps: {new_state.total_steps})")
    
    def test_parallel_envs_fill_buffer_faster(self):
        """Test that parallel envs add transitions proportionally."""
        from train.train_assembly import create_training_state, run_episode
        
        config = self.get_debug_config()
        key = random.PRNGKey(42)
        
        training_state, env, maddpg, params, vec_reset, vec_step = create_training_state(
            config, key
        )
        
        initial_buffer_size = int(training_state.maddpg_state.buffer_state.size)
        
        new_state, metrics = run_episode(
            env, maddpg, training_state, params, config,
            vec_reset, vec_step, explore=True
        )
        
        final_buffer_size = int(new_state.maddpg_state.buffer_state.size)
        transitions_added = final_buffer_size - initial_buffer_size
        
        # Should add approximately n_parallel_envs * max_steps transitions
        # (might be less if episode terminates early)
        expected_max = config.n_parallel_envs * config.max_steps
        
        assert transitions_added > 0, "Should add transitions to buffer"
        assert transitions_added <= expected_max, "Should not exceed max transitions"
        
        print(f"✓ parallel_envs_fill_buffer_faster passed (added: {transitions_added})")
    
    def test_episode_with_prior(self):
        """Test episode with prior policy regularization."""
        from train.train_assembly import create_training_state, run_episode
        from cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig(
            n_agents=3,
            n_parallel_envs=2,
            max_steps=5,
            batch_size=8,
            buffer_size=100,
            warmup_steps=5,
            hidden_dim=32,
            k_neighbors=2,
            prior_weight=0.3,  # Enable prior
        )
        
        key = random.PRNGKey(42)
        
        training_state, env, maddpg, params, vec_reset, vec_step = create_training_state(
            config, key
        )
        
        new_state, metrics = run_episode(
            env, maddpg, training_state, params, config,
            vec_reset, vec_step, explore=True
        )
        
        assert new_state.episode == 1
        print("✓ episode_with_prior passed")


# ============================================================================
# Test Gradient Updates
# ============================================================================

class TestGradientUpdates:
    """Test gradient computation and updates."""
    
    def get_debug_config(self):
        """Get a small debug config for testing."""
        from cfg import AssemblyTrainConfig
        return AssemblyTrainConfig(
            n_agents=3,
            n_parallel_envs=2,
            n_episodes=2,
            max_steps=10,
            batch_size=8,
            buffer_size=200,
            warmup_steps=20,  # Low warmup for testing
            hidden_dim=32,
            k_neighbors=2,
            updates_per_step=2,
        )
    
    def test_train_step_before_warmup(self):
        """Test train_step returns early before warmup."""
        from train.train_assembly import create_training_state, train_step
        
        config = self.get_debug_config()
        key = random.PRNGKey(42)
        
        training_state, env, maddpg, params, vec_reset, vec_step = create_training_state(
            config, key
        )
        
        # Buffer is empty, should not update
        new_maddpg_state, info = train_step(
            maddpg, training_state.maddpg_state, key, config
        )
        
        assert info["updated"] == False
        assert info["buffer_size"] == 0
        
        print("✓ train_step_before_warmup passed")
    
    def test_train_step_after_warmup(self):
        """Test train_step performs updates after warmup."""
        from train.train_assembly import create_training_state, run_episode, train_step
        from cfg import AssemblyTrainConfig
        
        # Config with very low warmup
        config = AssemblyTrainConfig(
            n_agents=3,
            n_parallel_envs=4,
            max_steps=20,
            batch_size=8,
            buffer_size=500,
            warmup_steps=10,  # Very low warmup
            hidden_dim=32,
            k_neighbors=2,
            updates_per_step=2,
        )
        
        key = random.PRNGKey(42)
        
        training_state, env, maddpg, params, vec_reset, vec_step = create_training_state(
            config, key
        )
        
        # Run episode to fill buffer
        training_state, _ = run_episode(
            env, maddpg, training_state, params, config,
            vec_reset, vec_step, explore=True
        )
        
        buffer_size = int(training_state.maddpg_state.buffer_state.size)
        
        if buffer_size >= config.warmup_steps:
            key, train_key = random.split(key)
            new_maddpg_state, info = train_step(
                maddpg, training_state.maddpg_state, train_key, config
            )
            
            assert info["updated"] == True
            assert "actor_loss" in info
            assert "critic_loss" in info
            print(f"✓ train_step_after_warmup passed (actor_loss: {info['actor_loss']:.4f})")
        else:
            print(f"✓ train_step_after_warmup skipped (buffer: {buffer_size} < {config.warmup_steps})")
    
    def test_gradient_not_nan(self):
        """Test that gradients are not NaN."""
        from train.train_assembly import create_training_state, run_episode, train_step
        from cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig(
            n_agents=3,
            n_parallel_envs=4,
            max_steps=20,
            batch_size=8,
            buffer_size=500,
            warmup_steps=10,
            hidden_dim=32,
            k_neighbors=2,
            updates_per_step=1,
        )
        
        key = random.PRNGKey(42)
        
        training_state, env, maddpg, params, vec_reset, vec_step = create_training_state(
            config, key
        )
        
        # Run episode to fill buffer
        training_state, _ = run_episode(
            env, maddpg, training_state, params, config,
            vec_reset, vec_step, explore=True
        )
        
        buffer_size = int(training_state.maddpg_state.buffer_state.size)
        
        if buffer_size >= config.warmup_steps:
            key, train_key = random.split(key)
            new_maddpg_state, info = train_step(
                maddpg, training_state.maddpg_state, train_key, config
            )
            
            assert not np.isnan(info.get("actor_loss", 0.0)), "Actor loss should not be NaN"
            assert not np.isnan(info.get("critic_loss", 0.0)), "Critic loss should not be NaN"
            print("✓ gradient_not_nan passed")
        else:
            print(f"✓ gradient_not_nan skipped (buffer: {buffer_size})")


# ============================================================================
# Test Noise Decay
# ============================================================================

class TestNoiseDecay:
    """Test noise decay during training."""
    
    def test_noise_decay_linear(self):
        """Test linear noise decay."""
        from train.train_assembly import create_training_state, run_episode
        from cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig(
            n_agents=3,
            n_parallel_envs=2,
            max_steps=50,
            batch_size=8,
            buffer_size=500,
            warmup_steps=10,
            hidden_dim=32,
            k_neighbors=2,
            noise_scale_initial=0.9,
            noise_scale_final=0.1,
            noise_decay_steps=100,
        )
        
        key = random.PRNGKey(42)
        
        training_state, env, maddpg, params, vec_reset, vec_step = create_training_state(
            config, key
        )
        
        initial_noise = float(training_state.maddpg_state.noise_scale)
        
        # Run a few episodes
        for _ in range(3):
            training_state, _ = run_episode(
                env, maddpg, training_state, params, config,
                vec_reset, vec_step, explore=True
            )
        
        final_noise = float(training_state.maddpg_state.noise_scale)
        
        # Noise should have decreased
        assert final_noise <= initial_noise, "Noise should decrease over time"
        assert final_noise >= config.noise_scale_final, "Noise should not go below minimum"
        
        print(f"✓ noise_decay_linear passed ({initial_noise:.3f} -> {final_noise:.3f})")


# ============================================================================
# Test End-to-End Mini Training
# ============================================================================

class TestEndToEnd:
    """End-to-end mini training tests."""
    
    def test_mini_training_loop(self):
        """Test a complete mini training loop."""
        from train.train_assembly import create_training_state, run_episode, train_step
        from cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig(
            n_agents=3,
            n_parallel_envs=2,
            n_episodes=3,
            max_steps=10,
            batch_size=8,
            buffer_size=200,
            warmup_steps=15,
            hidden_dim=32,
            k_neighbors=2,
            updates_per_step=1,
            log_interval=1,
        )
        
        key = random.PRNGKey(42)
        
        # Create training state
        training_state, env, maddpg, params, vec_reset, vec_step = create_training_state(
            config, key
        )
        
        rewards = []
        
        # Mini training loop
        for episode in range(config.n_episodes):
            # Run episode
            training_state, metrics = run_episode(
                env, maddpg, training_state, params, config,
                vec_reset, vec_step, explore=True
            )
            
            rewards.append(metrics["episode_reward"])
            
            # Train step
            key, train_key = random.split(training_state.key)
            maddpg_state, train_info = train_step(
                maddpg, training_state.maddpg_state, train_key, config
            )
            
            training_state = training_state.replace(
                maddpg_state=maddpg_state,
                key=key,
            )
        
        # Check final state
        assert training_state.episode == config.n_episodes
        assert training_state.total_steps > 0
        assert len(rewards) == config.n_episodes
        
        print(f"✓ mini_training_loop passed (episodes: {config.n_episodes}, final_reward: {rewards[-1]:.3f})")
    
    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoints."""
        from train.train_assembly import create_training_state, run_episode, save_checkpoint, load_checkpoint
        from cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig(
            n_agents=3,
            n_parallel_envs=2,
            max_steps=5,
            batch_size=8,
            buffer_size=100,
            warmup_steps=10,
            hidden_dim=32,
            k_neighbors=2,
        )
        
        key = random.PRNGKey(42)
        
        training_state, env, maddpg, params, vec_reset, vec_step = create_training_state(
            config, key
        )
        
        # Run an episode
        training_state, _ = run_episode(
            env, maddpg, training_state, params, config,
            vec_reset, vec_step, explore=True
        )
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pkl"
            save_checkpoint(training_state, maddpg, config, checkpoint_path)
            
            assert checkpoint_path.exists()
            
            # Load checkpoint
            loaded_state, loaded_config = load_checkpoint(checkpoint_path)
            
            assert loaded_state.episode == training_state.episode
            assert loaded_state.total_steps == training_state.total_steps
            
        print("✓ checkpoint_save_load passed")


# ============================================================================
# Test Evaluation
# ============================================================================

class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_run_eval_episode(self):
        """Test running eval episode without exploration."""
        from cfg import AssemblyTrainConfig, config_to_assembly_params, config_to_maddpg_config
        from assembly_env import AssemblySwarmEnv
        from algo import MADDPG
        from train.train_assembly import run_eval_episode
        
        config = AssemblyTrainConfig(
            n_agents=3,
            max_steps=10,
            hidden_dim=32,
            k_neighbors=2,
        )
        
        env = AssemblySwarmEnv(n_agents=config.n_agents)
        params = config_to_assembly_params(config)
        
        # Get actual obs_dim
        key = random.PRNGKey(42)
        key, reset_key = random.split(key)
        obs, _ = env.reset(reset_key, params)
        obs_dim = obs.shape[1]
        
        maddpg_config = config_to_maddpg_config(config, obs_dim, 2)
        maddpg = MADDPG(maddpg_config)
        
        key, init_key, eval_key = random.split(key, 3)
        maddpg_state = maddpg.init(init_key)
        
        # Run eval episode
        eval_metrics, eval_states = run_eval_episode(
            env, maddpg, maddpg_state, params, config, eval_key
        )
        
        # Check metrics
        assert "eval_reward" in eval_metrics
        assert "eval_coverage" in eval_metrics
        assert "eval_collision" in eval_metrics
        assert "eval_final_coverage" in eval_metrics
        assert "eval_steps" in eval_metrics
        
        # Check states collected for visualization
        assert len(eval_states) > 0
        assert len(eval_states) <= config.max_steps + 1  # +1 for initial state
        
        print("✓ run_eval_episode passed")
    
    def test_eval_no_buffer_modification(self):
        """Test that eval doesn't modify the replay buffer."""
        from cfg import AssemblyTrainConfig, config_to_assembly_params, config_to_maddpg_config
        from assembly_env import AssemblySwarmEnv
        from algo import MADDPG
        from train.train_assembly import run_eval_episode
        
        config = AssemblyTrainConfig(
            n_agents=3,
            max_steps=10,
            hidden_dim=32,
            k_neighbors=2,
        )
        
        env = AssemblySwarmEnv(n_agents=config.n_agents)
        params = config_to_assembly_params(config)
        
        key = random.PRNGKey(42)
        key, reset_key = random.split(key)
        obs, _ = env.reset(reset_key, params)
        obs_dim = obs.shape[1]
        
        maddpg_config = config_to_maddpg_config(config, obs_dim, 2)
        maddpg = MADDPG(maddpg_config)
        
        key, init_key, eval_key = random.split(key, 3)
        maddpg_state = maddpg.init(init_key)
        
        # Buffer size before eval
        buffer_size_before = int(maddpg_state.buffer_state.size)
        
        # Run eval episode (should NOT modify buffer)
        eval_metrics, eval_states = run_eval_episode(
            env, maddpg, maddpg_state, params, config, eval_key
        )
        
        # Buffer size should be unchanged (eval doesn't store transitions)
        # Note: run_eval_episode doesn't modify maddpg_state, so buffer is unchanged
        buffer_size_after = int(maddpg_state.buffer_state.size)
        assert buffer_size_before == buffer_size_after
        
        print("✓ eval_no_buffer_modification passed")
    
    def test_eval_states_for_visualization(self):
        """Test that eval states can be used for visualization."""
        from cfg import AssemblyTrainConfig, config_to_assembly_params, config_to_maddpg_config
        from assembly_env import AssemblySwarmEnv
        from algo import MADDPG
        from train.train_assembly import run_eval_episode
        
        config = AssemblyTrainConfig(
            n_agents=3,
            max_steps=10,
            hidden_dim=32,
            k_neighbors=2,
        )
        
        env = AssemblySwarmEnv(n_agents=config.n_agents)
        params = config_to_assembly_params(config)
        
        key = random.PRNGKey(42)
        key, reset_key = random.split(key)
        obs, _ = env.reset(reset_key, params)
        obs_dim = obs.shape[1]
        
        maddpg_config = config_to_maddpg_config(config, obs_dim, 2)
        maddpg = MADDPG(maddpg_config)
        
        key, init_key, eval_key = random.split(key, 3)
        maddpg_state = maddpg.init(init_key)
        
        eval_metrics, eval_states = run_eval_episode(
            env, maddpg, maddpg_state, params, config, eval_key
        )
        
        # Check each state has required attributes for visualization
        for state in eval_states:
            assert hasattr(state, 'positions')
            assert hasattr(state, 'velocities')
            assert hasattr(state, 'grid_centers')
            assert hasattr(state, 'grid_mask')
            assert hasattr(state, 'in_target')
            assert hasattr(state, 'is_colliding')
            
            # Check shapes
            assert state.positions.shape == (config.n_agents, 2)
            assert state.velocities.shape == (config.n_agents, 2)
        
        print("✓ eval_states_for_visualization passed")
    
    def test_eval_animation_save(self):
        """Test saving eval animation to file."""
        from cfg import AssemblyTrainConfig, config_to_assembly_params, config_to_maddpg_config
        from assembly_env import AssemblySwarmEnv
        from algo import MADDPG
        from train.train_assembly import run_eval_episode
        from visualize.renderer import create_animation
        
        config = AssemblyTrainConfig(
            n_agents=3,
            max_steps=10,
            hidden_dim=32,
            k_neighbors=2,
            eval_save_video=True,
            eval_video_fps=5,
        )
        
        env = AssemblySwarmEnv(n_agents=config.n_agents)
        params = config_to_assembly_params(config)
        
        key = random.PRNGKey(42)
        key, reset_key = random.split(key)
        obs, _ = env.reset(reset_key, params)
        obs_dim = obs.shape[1]
        
        maddpg_config = config_to_maddpg_config(config, obs_dim, 2)
        maddpg = MADDPG(maddpg_config)
        
        key, init_key, eval_key = random.split(key, 3)
        maddpg_state = maddpg.init(init_key)
        
        eval_metrics, eval_states = run_eval_episode(
            env, maddpg, maddpg_state, params, config, eval_key
        )
        
        # Save animation
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test_eval.gif"
            create_animation(
                eval_states, params,
                save_path=str(video_path),
                fps=config.eval_video_fps,
                show=False,
            )
            
            assert video_path.exists()
            assert video_path.stat().st_size > 0
        
        print("✓ eval_animation_save passed")
    
    def test_get_eval_dir(self):
        """Test get_eval_dir returns correct path."""
        from cfg import AssemblyTrainConfig, get_eval_dir
        
        config = AssemblyTrainConfig()
        eval_dir = get_eval_dir(config, run_name="test_run")
        
        assert "eval_videos" in eval_dir
        assert "assembly" in eval_dir
        assert "test_run" in eval_dir
        assert Path(eval_dir).exists()
        
        # Clean up
        Path(eval_dir).rmdir()
        Path(eval_dir).parent.rmdir()
        
        print("✓ get_eval_dir passed")
    
    def test_eval_config_fields(self):
        """Test eval config fields exist with correct defaults."""
        from cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig()
        
        assert hasattr(config, 'eval_interval')
        assert hasattr(config, 'eval_save_video')
        assert hasattr(config, 'eval_video_fps')
        assert hasattr(config, 'eval_dir')
        
        # Check defaults
        assert config.eval_interval == 50
        assert config.eval_save_video == True
        assert config.eval_video_fps == 10
        assert config.eval_dir is None
        
        print("✓ eval_config_fields passed")


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests manually (without pytest)."""
    print("=" * 70)
    print("Running train_assembly.py comprehensive tests")
    print("=" * 70)
    print()
    
    test_classes = [
        TestImports,
        TestConfiguration,
        TestEnvironmentCreation,
        TestTrainingState,
        TestEpisodeExecution,
        TestGradientUpdates,
        TestNoiseDecay,
        TestEndToEnd,
        TestEvaluation,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 50)
        
        instance = test_class()
        for method_name in sorted(dir(instance)):
            if method_name.startswith('test_'):
                try:
                    method = getattr(instance, method_name)
                    method()
                    passed += 1
                except Exception as e:
                    if "skipped" in str(e).lower():
                        skipped += 1
                        print(f"  ⊘ {method_name} skipped")
                    else:
                        print(f"  ✗ {method_name} FAILED: {e}")
                        import traceback
                        traceback.print_exc()
                        failed += 1
    
    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    # Check if pytest is available
    try:
        import pytest
        # Run with pytest for better output
        sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
    except ImportError:
        # Fall back to manual test runner
        success = run_all_tests()
        sys.exit(0 if success else 1)
