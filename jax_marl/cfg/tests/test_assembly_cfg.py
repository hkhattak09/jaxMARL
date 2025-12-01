"""Tests for assembly_cfg.py - Assembly environment configuration.

Run with: 
    pytest test_assembly_cfg.py -v
    
Or directly:
    python test_assembly_cfg.py
"""

import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ============================================================================
# Test AssemblyTrainConfig
# ============================================================================

class TestAssemblyTrainConfig:
    """Tests for AssemblyTrainConfig NamedTuple."""
    
    def test_default_config_creation(self):
        """Test creating configuration with default values."""
        from assembly_cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig()
        
        # Environment defaults
        assert config.n_agents == 30
        assert config.n_parallel_envs == 4
        assert config.arena_size == 5.0
        assert config.agent_radius == 0.1
        assert config.max_velocity == 0.8
        assert config.max_acceleration == 1.0
        
        # Observation defaults
        assert config.k_neighbors == 6
        assert config.d_sen == 3.0
        assert config.include_self_state == True
        
        # Physics
        assert config.dt == 0.1
        
        # Episode
        assert config.max_steps == 200
        
        # Domain randomization
        assert config.randomize_shape == True
        assert config.randomize_rotation == True
        assert config.randomize_scale == True
        assert config.randomize_offset == True
        
        # Reward
        assert config.reward_mode == "individual"
        
        # Algorithm
        assert config.hidden_dim == 256
        assert config.lr_actor == 1e-4
        assert config.lr_critic == 1e-3
        assert config.gamma == 0.95
        assert config.tau == 0.01
        assert config.buffer_size == 50000
        assert config.batch_size == 2048
        assert config.warmup_steps == 5000
        assert config.noise_scale_initial == 0.9
        assert config.noise_scale_final == 0.1
        assert config.noise_decay_steps == 100000
        assert config.update_every == 100
        assert config.updates_per_step == 30
        assert config.prior_weight == 0.3
        
        # Training
        assert config.seed == 226
        assert config.n_episodes == 3000
        assert config.log_interval == 10
        assert config.save_interval == 100
        assert config.eval_interval == 50
        
        # Paths (default to None)
        assert config.shape_file is None
        assert config.checkpoint_dir is None
        assert config.log_dir is None
        
        print("✓ Default config creation passed")
    
    def test_config_is_namedtuple(self):
        """Test that config is a NamedTuple (immutable, hashable)."""
        from assembly_cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig()
        
        # Should be a tuple subclass
        assert isinstance(config, tuple)
        
        # Should have _fields attribute
        assert hasattr(config, '_fields')
        assert 'n_agents' in config._fields
        assert 'batch_size' in config._fields
        
        # Should be hashable
        hash(config)
        
        print("✓ Config is NamedTuple passed")
    
    def test_config_immutability(self):
        """Test that config cannot be modified after creation."""
        from assembly_cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig()
        
        # NamedTuples should not allow modification
        with pytest.raises(AttributeError):
            config.n_agents = 50
        
        print("✓ Config immutability passed")
    
    def test_config_with_custom_values(self):
        """Test creating config with custom values."""
        from assembly_cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig(
            n_agents=50,
            n_parallel_envs=8,
            batch_size=512,
            n_episodes=1000,
            lr_actor=5e-4,
            prior_weight=0.0,
        )
        
        # Custom values
        assert config.n_agents == 50
        assert config.n_parallel_envs == 8
        assert config.batch_size == 512
        assert config.n_episodes == 1000
        assert config.lr_actor == 5e-4
        assert config.prior_weight == 0.0
        
        # Unchanged defaults
        assert config.hidden_dim == 256
        assert config.gamma == 0.95
        assert config.arena_size == 5.0
        
        print("✓ Config with custom values passed")
    
    def test_config_replace(self):
        """Test using _replace to create modified config."""
        from assembly_cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig()
        
        # Create new config with some values changed
        new_config = config._replace(n_agents=100, batch_size=256)
        
        # Original unchanged
        assert config.n_agents == 30
        assert config.batch_size == 2048
        
        # New config has changes
        assert new_config.n_agents == 100
        assert new_config.batch_size == 256
        
        # Other values preserved
        assert new_config.hidden_dim == 256
        assert new_config.gamma == 0.95
        
        print("✓ Config _replace passed")
    
    def test_config_asdict(self):
        """Test converting config to dictionary."""
        from assembly_cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig()
        config_dict = config._asdict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['n_agents'] == 30
        assert config_dict['batch_size'] == 2048
        assert len(config_dict) == len(config._fields)
        
        print("✓ Config _asdict passed")
    
    def test_config_field_types(self):
        """Test that config fields have correct types."""
        from assembly_cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig()
        
        # Integer fields
        assert isinstance(config.n_agents, int)
        assert isinstance(config.n_parallel_envs, int)
        assert isinstance(config.hidden_dim, int)
        assert isinstance(config.buffer_size, int)
        assert isinstance(config.batch_size, int)
        assert isinstance(config.seed, int)
        
        # Float fields
        assert isinstance(config.arena_size, float)
        assert isinstance(config.lr_actor, float)
        assert isinstance(config.gamma, float)
        assert isinstance(config.prior_weight, float)
        
        # Boolean fields
        assert isinstance(config.include_self_state, bool)
        assert isinstance(config.randomize_shape, bool)
        
        # String fields
        assert isinstance(config.reward_mode, str)
        
        print("✓ Config field types passed")


# ============================================================================
# Test get_config Function
# ============================================================================

class TestGetConfig:
    """Tests for get_config() function."""
    
    def test_get_config_returns_default(self):
        """Test that get_config returns default config."""
        from assembly_cfg import get_config, AssemblyTrainConfig
        
        config = get_config()
        default = AssemblyTrainConfig()
        
        assert isinstance(config, AssemblyTrainConfig)
        assert config == default
        
        print("✓ get_config returns default passed")


# ============================================================================
# Test Path Utilities
# ============================================================================

class TestPathUtilities:
    """Tests for path utility functions."""
    
    def test_get_shape_file_path_default(self):
        """Test default shape file path."""
        from assembly_cfg import get_config, get_shape_file_path
        
        config = get_config()
        path = get_shape_file_path(config)
        
        assert isinstance(path, str)
        assert path.endswith("results.pkl")
        assert "fig" in path
        
        print("✓ get_shape_file_path default passed")
    
    def test_get_shape_file_path_custom(self):
        """Test custom shape file path."""
        from assembly_cfg import AssemblyTrainConfig, get_shape_file_path
        
        custom_path = "/tmp/custom_shapes.pkl"
        config = AssemblyTrainConfig(shape_file=custom_path)
        path = get_shape_file_path(config)
        
        assert path == custom_path
        
        print("✓ get_shape_file_path custom passed")
    
    def test_get_checkpoint_dir_creates_directory(self):
        """Test that get_checkpoint_dir creates the directory."""
        from assembly_cfg import AssemblyTrainConfig, get_checkpoint_dir
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AssemblyTrainConfig(checkpoint_dir=tmpdir)
            checkpoint_dir = get_checkpoint_dir(config, run_name="test_run")
            
            assert Path(checkpoint_dir).exists()
            assert "test_run" in checkpoint_dir
            assert "assembly" in checkpoint_dir
        
        print("✓ get_checkpoint_dir creates directory passed")
    
    def test_get_checkpoint_dir_auto_run_name(self):
        """Test that get_checkpoint_dir generates run name if not provided."""
        from assembly_cfg import AssemblyTrainConfig, get_checkpoint_dir
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AssemblyTrainConfig(checkpoint_dir=tmpdir)
            checkpoint_dir = get_checkpoint_dir(config)  # No run_name
            
            assert Path(checkpoint_dir).exists()
            # Should contain date-formatted name
            assert "assembly" in checkpoint_dir
        
        print("✓ get_checkpoint_dir auto run_name passed")
    
    def test_get_log_dir_creates_directory(self):
        """Test that get_log_dir creates the directory."""
        from assembly_cfg import AssemblyTrainConfig, get_log_dir
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AssemblyTrainConfig(log_dir=tmpdir)
            log_dir = get_log_dir(config, run_name="test_run")
            
            assert Path(log_dir).exists()
            assert "test_run" in log_dir
            assert "assembly" in log_dir
        
        print("✓ get_log_dir creates directory passed")


# ============================================================================
# Test Config Conversion Functions
# ============================================================================

class TestConfigConversion:
    """Tests for config conversion functions."""
    
    def test_config_to_maddpg_config(self):
        """Test converting to MADDPGConfig."""
        from assembly_cfg import AssemblyTrainConfig, config_to_maddpg_config
        
        config = AssemblyTrainConfig(
            n_agents=10,
            hidden_dim=128,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.99,
            tau=0.005,
            buffer_size=10000,
            batch_size=256,
            prior_weight=0.5,
        )
        
        obs_dim = 20
        action_dim = 2
        
        maddpg_config = config_to_maddpg_config(config, obs_dim, action_dim)
        
        assert maddpg_config.n_agents == 10
        assert maddpg_config.obs_dims == tuple([20] * 10)
        assert maddpg_config.action_dims == tuple([2] * 10)
        assert maddpg_config.hidden_dims == (128, 128)
        assert maddpg_config.lr_actor == 1e-4
        assert maddpg_config.lr_critic == 1e-3
        assert maddpg_config.gamma == 0.99
        assert maddpg_config.tau == 0.005
        assert maddpg_config.buffer_size == 10000
        assert maddpg_config.batch_size == 256
        assert maddpg_config.prior_weight == 0.5
        
        print("✓ config_to_maddpg_config passed")
    
    def test_config_to_assembly_params(self):
        """Test converting to AssemblyParams."""
        from assembly_cfg import AssemblyTrainConfig, config_to_assembly_params
        
        config = AssemblyTrainConfig(
            n_agents=10,
            arena_size=4.0,
            agent_radius=0.15,
            max_velocity=1.0,
            max_acceleration=2.0,
            k_neighbors=4,
            d_sen=2.5,
            dt=0.05,
            max_steps=100,
            randomize_shape=False,
            reward_mode="shared_mean",
        )
        
        params = config_to_assembly_params(config)
        
        assert params.arena_size == 4.0
        assert params.agent_radius == 0.15
        assert params.max_velocity == 1.0
        assert params.max_acceleration == 2.0
        assert params.k_neighbors == 4
        assert params.d_sen == 2.5
        assert params.dt == 0.05
        assert params.max_steps == 100
        assert params.randomize_shape == False
        assert params.reward_mode == "shared_mean"
        
        # Check nested params
        assert params.obs_params.topo_nei_max == 4
        assert params.obs_params.d_sen == 2.5
        assert params.reward_params.reward_mode == "shared_mean"
        assert params.physics.dt == 0.05
        
        print("✓ config_to_assembly_params passed")


# ============================================================================
# Test Value Constraints
# ============================================================================

class TestValueConstraints:
    """Tests for sensible value constraints."""
    
    def test_positive_values(self):
        """Test that values that should be positive are positive."""
        from assembly_cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig()
        
        assert config.n_agents > 0
        assert config.n_parallel_envs > 0
        assert config.arena_size > 0
        assert config.agent_radius > 0
        assert config.max_velocity > 0
        assert config.dt > 0
        assert config.max_steps > 0
        assert config.hidden_dim > 0
        assert config.buffer_size > 0
        assert config.batch_size > 0
        
        print("✓ Positive values passed")
    
    def test_probability_ranges(self):
        """Test that probability-like values are in valid range."""
        from assembly_cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig()
        
        # Gamma should be in [0, 1]
        assert 0 <= config.gamma <= 1
        
        # Tau should be small positive
        assert 0 < config.tau <= 1
        
        # Prior weight should be in [0, 1]
        assert 0 <= config.prior_weight <= 1
        
        print("✓ Probability ranges passed")
    
    def test_noise_scale_decay(self):
        """Test noise scale values make sense."""
        from assembly_cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig()
        
        # Initial noise should be >= final noise
        assert config.noise_scale_initial >= config.noise_scale_final
        
        # Both should be positive
        assert config.noise_scale_initial > 0
        assert config.noise_scale_final >= 0
        
        # Decay steps should be positive
        assert config.noise_decay_steps > 0
        
        print("✓ Noise scale decay passed")
    
    def test_reward_mode_valid(self):
        """Test reward mode is valid."""
        from assembly_cfg import AssemblyTrainConfig
        
        config = AssemblyTrainConfig()
        
        valid_modes = ["individual", "shared_mean", "shared_max"]
        assert config.reward_mode in valid_modes
        
        print("✓ Reward mode valid passed")


# ============================================================================
# Run Tests
# ============================================================================

def run_all_tests():
    """Run all tests manually (without pytest)."""
    print("=" * 60)
    print("Running assembly_cfg.py tests")
    print("=" * 60)
    print()
    
    test_classes = [
        TestAssemblyTrainConfig,
        TestGetConfig,
        TestPathUtilities,
        TestConfigConversion,
        TestValueConstraints,
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    method = getattr(instance, method_name)
                    method()
                    passed += 1
                except Exception as e:
                    print(f"✗ {method_name} FAILED: {e}")
                    failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    # Check if pytest is available
    try:
        import pytest
        # Run with pytest for better output
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        # Fall back to manual test runner
        success = run_all_tests()
        sys.exit(0 if success else 1)
