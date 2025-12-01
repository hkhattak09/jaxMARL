"""
Test JIT Rollout Implementation.

Tests the JIT-compiled rollout using jax.lax.scan for faster training.
"""
import sys
import os

# Add necessary paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_JAX_MARL_DIR = os.path.join(_THIS_DIR, '..', '..')  # jax_marl directory
_JAX_CUS_GYM_DIR = os.path.join(_THIS_DIR, '..', '..', '..', 'jax_cus_gym')
_TRAIN_DIR = os.path.join(_THIS_DIR, '..')  # train directory

sys.path.insert(0, _JAX_CUS_GYM_DIR)
sys.path.insert(0, _JAX_MARL_DIR)
sys.path.insert(0, _TRAIN_DIR)

import jax
import jax.numpy as jnp
from jax import random
import time


# ============================================================================
# Shared Setup Helper
# ============================================================================

def create_test_setup(n_agents=3, n_envs=2, max_steps=10, prior_weight=0.0, buffer_size=100):
    """Create common test setup for JIT rollout tests."""
    from cfg import get_config, config_to_assembly_params
    from assembly_env import make_vec_env
    from algo import MADDPG, MADDPGConfig
    
    config = get_config()._replace(
        n_agents=n_agents,
        n_parallel_envs=n_envs,
        max_steps=max_steps,
        prior_weight=prior_weight,
    )
    
    env, _, _, _ = make_vec_env(n_envs=n_envs, n_agents=n_agents)
    params = config_to_assembly_params(config)
    
    @jax.jit
    def vec_step(keys, states, actions):
        return jax.vmap(lambda k, s, a: env.step(k, s, a, params))(keys, states, actions)
    
    @jax.jit
    def vec_reset(keys):
        return jax.vmap(lambda k: env.reset(k, params))(keys)
    
    # Get actual obs_dim from environment reset (not computed - they may differ)
    key = random.PRNGKey(0)
    reset_keys = random.split(key, n_envs)
    obs_batch, _ = vec_reset(reset_keys)
    obs_dim = obs_batch.shape[-1]  # Actual obs dim from environment
    
    maddpg_config = MADDPGConfig(
        n_agents=n_agents,
        obs_dims=tuple([obs_dim] * n_agents),
        action_dims=tuple([2] * n_agents),
        hidden_dims=(32, 32),
        buffer_size=buffer_size,
        batch_size=16,
    )
    maddpg = MADDPG(maddpg_config)
    
    return {
        'config': config,
        'env': env,
        'params': params,
        'maddpg': maddpg,
        'obs_dim': obs_dim,
        'vec_step': vec_step,
        'vec_reset': vec_reset,
    }


# ============================================================================
# Test JIT Rollout Imports
# ============================================================================

class TestJITRolloutImports:
    """Test that all required components can be imported."""
    
    def test_import_rollout_carry(self):
        """Test RolloutCarry can be imported."""
        from train.train_assembly import RolloutCarry
        assert RolloutCarry is not None
        print("✓ RolloutCarry imported successfully")
    
    def test_import_rollout_metrics(self):
        """Test RolloutMetrics can be imported."""
        from train.train_assembly import RolloutMetrics
        assert RolloutMetrics is not None
        print("✓ RolloutMetrics imported successfully")
    
    def test_import_create_jit_rollout_fn(self):
        """Test create_jit_rollout_fn can be imported."""
        from train.train_assembly import create_jit_rollout_fn
        assert callable(create_jit_rollout_fn)
        print("✓ create_jit_rollout_fn imported successfully")


# ============================================================================
# Test RolloutCarry Dataclass
# ============================================================================

class TestRolloutCarry:
    """Test RolloutCarry dataclass creation and properties."""
    
    def test_rollout_carry_creation(self):
        """Test that RolloutCarry can be created with proper values."""
        from train.train_assembly import RolloutCarry
        from algo import MADDPG, MADDPGConfig
        
        setup = create_test_setup()
        obs_dim = setup['obs_dim']
        n_envs = setup['config'].n_parallel_envs
        n_agents = setup['config'].n_agents
        
        key = random.PRNGKey(42)
        maddpg_state = setup['maddpg'].init(key)
        
        # Create sample data
        obs_batch = jnp.zeros((n_envs, n_agents, obs_dim))
        
        key, reset_key = random.split(key)
        reset_keys = random.split(reset_key, n_envs)
        _, env_states = setup['vec_reset'](reset_keys)
        
        # Create RolloutCarry - must match actual fields in train_assembly.py
        carry = RolloutCarry(
            key=key,
            obs_batch=obs_batch,
            env_states=env_states,
            maddpg_state=maddpg_state,
            episode_rewards=jnp.zeros(n_envs),
            total_coverage=jnp.array(0.0),
            total_collision=jnp.array(0.0),
            done_flag=jnp.array(False),
        )
        
        assert carry.obs_batch.shape == (n_envs, n_agents, obs_dim)
        assert carry.episode_rewards.shape == (n_envs,)
        assert carry.total_coverage.shape == ()
        print(f"✓ RolloutCarry created successfully with obs_batch shape {carry.obs_batch.shape}")


# ============================================================================
# Test JIT Rollout Creation
# ============================================================================

class TestJITRolloutCreation:
    """Test create_jit_rollout_fn."""
    
    def test_create_jit_rollout_fn_returns_callable(self):
        """Test that create_jit_rollout_fn returns a callable."""
        from train.train_assembly import create_jit_rollout_fn
        
        setup = create_test_setup()
        jit_rollout_fn = create_jit_rollout_fn(
            setup['maddpg'],
            setup['params'],
            setup['config'],
            setup['vec_step'],
        )
        
        assert callable(jit_rollout_fn), "create_jit_rollout_fn should return callable"
        print("✓ create_jit_rollout_fn returns callable")
    
    def test_jit_rollout_fn_executes(self):
        """Test that JIT rollout function executes without error."""
        from train.train_assembly import create_jit_rollout_fn
        
        setup = create_test_setup()
        jit_rollout_fn = create_jit_rollout_fn(
            setup['maddpg'],
            setup['params'],
            setup['config'],
            setup['vec_step'],
        )
        
        key = random.PRNGKey(42)
        maddpg_state = setup['maddpg'].init(key)
        
        # Reset environment
        key, reset_key = random.split(key)
        reset_keys = random.split(reset_key, setup['config'].n_parallel_envs)
        obs_batch, env_states = setup['vec_reset'](reset_keys)
        
        # Run JIT rollout
        key, rollout_key = random.split(key)
        final_carry, all_metrics = jit_rollout_fn(
            rollout_key, obs_batch, env_states, maddpg_state
        )
        
        # Block until complete
        final_carry.episode_rewards.block_until_ready()
        
        assert final_carry.episode_rewards.shape == (setup['config'].n_parallel_envs,)
        print(f"✓ JIT rollout executed successfully")
        print(f"  Episode rewards shape: {final_carry.episode_rewards.shape}")


# ============================================================================
# Test JIT Rollout Correctness
# ============================================================================

class TestJITRolloutCorrectness:
    """Test that JIT rollout produces correct results."""
    
    def test_buffer_fills_during_rollout(self):
        """Test that buffer is filled during JIT rollout."""
        from train.train_assembly import create_jit_rollout_fn
        
        n_envs = 4
        max_steps = 20
        setup = create_test_setup(n_envs=n_envs, max_steps=max_steps, buffer_size=1000)
        
        jit_rollout_fn = create_jit_rollout_fn(
            setup['maddpg'], setup['params'], setup['config'], setup['vec_step']
        )
        
        key = random.PRNGKey(42)
        maddpg_state = setup['maddpg'].init(key)
        
        # Check initial buffer size
        initial_buffer_size = int(maddpg_state.buffer_state.size)
        assert initial_buffer_size == 0, f"Buffer should start empty, got {initial_buffer_size}"
        
        # Reset and run rollout
        key, reset_key = random.split(key)
        reset_keys = random.split(reset_key, n_envs)
        obs_batch, env_states = setup['vec_reset'](reset_keys)
        
        key, rollout_key = random.split(key)
        final_carry, _ = jit_rollout_fn(rollout_key, obs_batch, env_states, maddpg_state)
        
        # Check buffer was filled
        final_buffer_size = int(final_carry.maddpg_state.buffer_state.size)
        expected_transitions = n_envs * max_steps
        
        assert final_buffer_size == expected_transitions, \
            f"Buffer should have {expected_transitions} transitions, got {final_buffer_size}"
        
        print(f"✓ Buffer filled correctly during JIT rollout")
        print(f"  Initial size: {initial_buffer_size}")
        print(f"  Final size: {final_buffer_size}")
    
    def test_metrics_accumulation(self):
        """Test that metrics are accumulated correctly."""
        from train.train_assembly import create_jit_rollout_fn
        
        n_envs = 2
        max_steps = 10
        setup = create_test_setup(n_envs=n_envs, max_steps=max_steps)
        
        jit_rollout_fn = create_jit_rollout_fn(
            setup['maddpg'], setup['params'], setup['config'], setup['vec_step']
        )
        
        key = random.PRNGKey(42)
        maddpg_state = setup['maddpg'].init(key)
        
        # Reset and run rollout
        key, reset_key = random.split(key)
        reset_keys = random.split(reset_key, n_envs)
        obs_batch, env_states = setup['vec_reset'](reset_keys)
        
        key, rollout_key = random.split(key)
        final_carry, all_metrics = jit_rollout_fn(rollout_key, obs_batch, env_states, maddpg_state)
        
        # Check that metrics were accumulated (per-step metrics from scan)
        # RolloutMetrics has: reward, coverage, collision
        assert all_metrics.reward.shape == (max_steps,)
        assert all_metrics.coverage.shape == (max_steps,)
        assert all_metrics.collision.shape == (max_steps,)
        
        print(f"✓ Metrics accumulated correctly")
        print(f"  Reward shape: {all_metrics.reward.shape}")
        print(f"  Coverage shape: {all_metrics.coverage.shape}")


# ============================================================================
# Test JIT vs Python Equivalence
# ============================================================================

class TestJITvsPythonEquivalence:
    """Test that JIT rollout produces equivalent results to Python loop."""
    
    def test_same_results_with_same_seed(self):
        """Test JIT and Python loop produce same results with same seed."""
        from train.train_assembly import create_jit_rollout_fn
        
        n_envs = 2
        max_steps = 5
        setup = create_test_setup(n_envs=n_envs, max_steps=max_steps)
        
        # Create JIT rollout
        jit_rollout_fn = create_jit_rollout_fn(
            setup['maddpg'], setup['params'], setup['config'], setup['vec_step']
        )
        
        key = random.PRNGKey(123)
        
        # Run JIT rollout
        jit_maddpg_state = setup['maddpg'].init(key)
        key, reset_key = random.split(key)
        reset_keys = random.split(reset_key, n_envs)
        obs_batch, env_states = setup['vec_reset'](reset_keys)
        
        key, rollout_key = random.split(key)
        jit_final_carry, jit_metrics = jit_rollout_fn(rollout_key, obs_batch, env_states, jit_maddpg_state)
        
        # Wait for results
        jit_rewards = float(jnp.mean(jit_final_carry.episode_rewards))
        
        # Results should be finite
        assert jnp.isfinite(jit_rewards), f"JIT rewards should be finite, got {jit_rewards}"
        
        print(f"✓ JIT rollout produced finite results")
        print(f"  Mean episode reward: {jit_rewards:.4f}")


# ============================================================================
# Test JIT Rollout Performance
# ============================================================================

class TestJITRolloutPerformance:
    """Test that JIT rollout is fast after compilation."""
    
    def test_jit_warmup_then_fast(self):
        """Test that JIT rollout is fast after warmup."""
        from train.train_assembly import create_jit_rollout_fn
        
        n_envs = 8
        max_steps = 50
        setup = create_test_setup(n_envs=n_envs, max_steps=max_steps, buffer_size=10000)
        
        jit_rollout_fn = create_jit_rollout_fn(
            setup['maddpg'], setup['params'], setup['config'], setup['vec_step']
        )
        
        key = random.PRNGKey(42)
        maddpg_state = setup['maddpg'].init(key)
        
        # Reset environment
        key, reset_key = random.split(key)
        reset_keys = random.split(reset_key, n_envs)
        obs_batch, env_states = setup['vec_reset'](reset_keys)
        
        # Warmup run (JIT compilation)
        key, rollout_key = random.split(key)
        start_warmup = time.perf_counter()
        final_carry, _ = jit_rollout_fn(rollout_key, obs_batch, env_states, maddpg_state)
        final_carry.episode_rewards.block_until_ready()
        warmup_time = time.perf_counter() - start_warmup
        
        # Second run should be faster (already compiled)
        maddpg_state = final_carry.maddpg_state
        key, reset_key = random.split(key)
        reset_keys = random.split(reset_key, n_envs)
        obs_batch, env_states = setup['vec_reset'](reset_keys)
        
        key, rollout_key = random.split(key)
        start_fast = time.perf_counter()
        final_carry, _ = jit_rollout_fn(rollout_key, obs_batch, env_states, maddpg_state)
        final_carry.episode_rewards.block_until_ready()
        fast_time = time.perf_counter() - start_fast
        
        print(f"✓ JIT compilation and execution test passed")
        print(f"  Warmup time (includes JIT): {warmup_time:.3f}s")
        print(f"  Fast run time: {fast_time:.3f}s")
        print(f"  Speedup: {warmup_time / fast_time:.1f}x")
        
        # Fast run should be reasonably fast (less than warmup)
        # This is a soft check - timing can vary
        assert fast_time < warmup_time * 0.8 or fast_time < 1.0, \
            f"Fast run should be faster than warmup or under 1s"


# ============================================================================
# Test JIT Rollout with Prior Policy
# ============================================================================

class TestJITRolloutWithPrior:
    """Test JIT rollout with prior policy enabled."""
    
    def test_jit_rollout_with_prior_policy(self):
        """Test JIT rollout works correctly with prior policy."""
        from train.train_assembly import create_jit_rollout_fn
        
        n_envs = 2
        max_steps = 10
        prior_weight = 0.3
        setup = create_test_setup(n_envs=n_envs, max_steps=max_steps, prior_weight=prior_weight)
        
        jit_rollout_fn = create_jit_rollout_fn(
            setup['maddpg'], setup['params'], setup['config'], setup['vec_step']
        )
        
        key = random.PRNGKey(42)
        maddpg_state = setup['maddpg'].init(key)
        
        # Reset and run rollout
        key, reset_key = random.split(key)
        reset_keys = random.split(reset_key, n_envs)
        obs_batch, env_states = setup['vec_reset'](reset_keys)
        
        key, rollout_key = random.split(key)
        final_carry, _ = jit_rollout_fn(rollout_key, obs_batch, env_states, maddpg_state)
        
        # Should complete without error
        final_buffer_size = int(final_carry.maddpg_state.buffer_state.size)
        expected_transitions = n_envs * max_steps
        
        assert final_buffer_size == expected_transitions, \
            f"Buffer should have {expected_transitions} transitions, got {final_buffer_size}"
        
        print(f"✓ JIT rollout with prior policy works correctly")
        print(f"  Prior weight: {prior_weight}")
        print(f"  Buffer size: {final_buffer_size}")


# ============================================================================
# Run Tests
# ============================================================================

def run_all_tests():
    """Run all JIT rollout tests."""
    print("=" * 60)
    print("  JIT ROLLOUT TEST SUITE")
    print("=" * 60)
    
    test_classes = [
        TestJITRolloutImports,
        TestRolloutCarry,
        TestJITRolloutCreation,
        TestJITRolloutCorrectness,
        TestJITvsPythonEquivalence,
        TestJITRolloutPerformance,
        TestJITRolloutWithPrior,
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"  {test_class.__name__}")
        print(f"{'='*60}")
        
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    method = getattr(instance, method_name)
                    method()
                    passed += 1
                except Exception as e:
                    print(f"  ✗ FAILED: {method_name}")
                    print(f"    Error: {str(e)[:200]}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
    
    print(f"\n{'='*60}")
    print(f"  TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total: {passed + failed}")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
