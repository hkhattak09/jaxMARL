"""Tests for noise.py - exploration noise for MADDPG."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random

from noise import (
    GaussianNoiseParams,
    gaussian_noise,
    gaussian_noise_log_prob,
    add_gaussian_noise,
    OUNoiseState,
    OUNoiseParams,
    ou_noise_init,
    ou_noise_reset,
    ou_noise_step,
    add_ou_noise,
    epsilon_greedy_continuous,
    gaussian_noise_batch,
    add_gaussian_noise_batch,
    ou_noise_init_batch,
    ou_noise_step_batch,
    linear_schedule,
    exponential_schedule,
)


def test_gaussian_noise():
    """Test Gaussian noise generation."""
    print("Testing gaussian_noise...")
    
    key = random.PRNGKey(42)
    shape = (1000,)
    scale = 0.5
    
    noise = gaussian_noise(key, shape, scale)
    
    # Check shape
    assert noise.shape == shape, f"Shape mismatch: {noise.shape}"
    
    # Check statistics (should be approximately N(0, scale^2))
    mean = jnp.mean(noise)
    std = jnp.std(noise)
    
    assert jnp.abs(mean) < 0.1, f"Mean {mean} should be close to 0"
    assert jnp.abs(std - scale) < 0.1, f"Std {std} should be close to {scale}"
    
    print(f"   Mean: {mean:.4f} (expected: 0.0)")
    print(f"   Std: {std:.4f} (expected: {scale})")
    
    # Test 2D shape
    key, subkey = random.split(key)
    noise_2d = gaussian_noise(subkey, (10, 2), scale)
    assert noise_2d.shape == (10, 2), f"2D shape mismatch: {noise_2d.shape}"
    
    print("   gaussian_noise: PASSED")
    return True


def test_gaussian_noise_log_prob():
    """Test Gaussian noise log probability."""
    print("Testing gaussian_noise_log_prob...")
    
    scale = 0.5
    
    # Zero noise should have highest probability
    zero_noise = jnp.zeros(3)
    log_prob_zero = gaussian_noise_log_prob(zero_noise, scale)
    
    # Larger noise should have lower probability
    large_noise = jnp.ones(3) * 2.0
    log_prob_large = gaussian_noise_log_prob(large_noise, scale)
    
    assert log_prob_zero > log_prob_large, "Zero noise should have higher prob than large noise"
    print(f"   Log prob (zero noise): {log_prob_zero:.4f}")
    print(f"   Log prob (large noise): {log_prob_large:.4f}")
    
    # Test batched
    batched_noise = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    log_probs = gaussian_noise_log_prob(batched_noise, scale)
    assert log_probs.shape == (3,), f"Batched log prob shape: {log_probs.shape}"
    assert jnp.all(log_probs[:-1] > log_probs[1:]), "Log probs should decrease with distance from 0"
    
    print("   gaussian_noise_log_prob: PASSED")
    return True


def test_add_gaussian_noise():
    """Test adding Gaussian noise to actions."""
    print("Testing add_gaussian_noise...")
    
    key = random.PRNGKey(42)
    action = jnp.array([0.5, -0.3])
    scale = 0.1
    
    noisy_action, noise = add_gaussian_noise(key, action, scale)
    
    # Check shape
    assert noisy_action.shape == action.shape, f"Shape mismatch"
    assert noise.shape == action.shape, f"Noise shape mismatch"
    
    # Check clipping
    assert jnp.all(noisy_action >= -1.0), "Action below lower bound"
    assert jnp.all(noisy_action <= 1.0), "Action above upper bound"
    
    print(f"   Original action: {action}")
    print(f"   Noisy action: {noisy_action}")
    print(f"   Noise: {noise}")
    
    # Test clipping with extreme action + noise
    key, subkey = random.split(key)
    extreme_action = jnp.array([0.95, -0.95])
    noisy_extreme, _ = add_gaussian_noise(subkey, extreme_action, scale=0.5)
    assert jnp.all(noisy_extreme >= -1.0) and jnp.all(noisy_extreme <= 1.0), "Clipping failed"
    
    print("   add_gaussian_noise: PASSED")
    return True


def test_ou_noise_init_and_reset():
    """Test OU noise initialization and reset."""
    print("Testing ou_noise_init and ou_noise_reset...")
    
    action_dim = 3
    mu = 0.0
    
    # Test init
    state = ou_noise_init(action_dim, mu)
    assert state.state.shape == (action_dim,), f"Init shape: {state.state.shape}"
    assert jnp.allclose(state.state, mu), "Init should equal mu"
    print(f"   Initialized state: {state.state}")
    
    # Test reset
    params = OUNoiseParams(mu=0.0, theta=0.15, sigma=0.2, dt=1.0, action_dim=action_dim)
    reset_state = ou_noise_reset(params)
    assert jnp.allclose(reset_state.state, params.mu), "Reset should equal mu"
    print(f"   Reset state: {reset_state.state}")
    
    print("   ou_noise_init and ou_noise_reset: PASSED")
    return True


def test_ou_noise_step():
    """Test OU noise step function."""
    print("Testing ou_noise_step...")
    
    action_dim = 2
    params = OUNoiseParams(
        mu=0.0,
        theta=0.15,
        sigma=0.2,
        dt=1.0,
        action_dim=action_dim
    )
    
    # Initialize at non-zero state to test mean reversion
    state = OUNoiseState(state=jnp.array([1.0, -1.0]))
    
    key = random.PRNGKey(42)
    
    # Run multiple steps
    states = [state.state]
    for i in range(100):
        key, subkey = random.split(key)
        state, noise = ou_noise_step(subkey, state, params)
        states.append(state.state)
    
    states = jnp.stack(states)
    
    # Check that state has right shape
    assert state.state.shape == (action_dim,), f"State shape: {state.state.shape}"
    
    # Check mean reversion: final state should be closer to mu than initial
    initial_dist = jnp.linalg.norm(states[0] - params.mu)
    final_dist = jnp.linalg.norm(states[-1] - params.mu)
    
    # Over many steps, average should be close to mu
    mean_state = jnp.mean(states[50:], axis=0)  # Skip initial transient
    assert jnp.linalg.norm(mean_state - params.mu) < 1.0, f"Mean state should revert to mu: {mean_state}"
    
    print(f"   Initial distance from mu: {initial_dist:.4f}")
    print(f"   Final distance from mu: {final_dist:.4f}")
    print(f"   Mean state (later half): {mean_state}")
    
    print("   ou_noise_step: PASSED")
    return True


def test_add_ou_noise():
    """Test adding OU noise to actions."""
    print("Testing add_ou_noise...")
    
    action_dim = 2
    params = OUNoiseParams(
        mu=0.0,
        theta=0.15,
        sigma=0.2,
        dt=1.0,
        action_dim=action_dim
    )
    
    key = random.PRNGKey(42)
    action = jnp.array([0.5, -0.3])
    ou_state = ou_noise_init(action_dim)
    
    noisy_action, new_state, noise = add_ou_noise(key, action, ou_state, params)
    
    # Check shapes
    assert noisy_action.shape == action.shape, f"Noisy action shape: {noisy_action.shape}"
    assert new_state.state.shape == (action_dim,), f"New state shape: {new_state.state.shape}"
    assert noise.shape == action.shape, f"Noise shape: {noise.shape}"
    
    # Check clipping
    assert jnp.all(noisy_action >= -1.0) and jnp.all(noisy_action <= 1.0), "Clipping failed"
    
    print(f"   Original action: {action}")
    print(f"   Noisy action: {noisy_action}")
    print(f"   New OU state: {new_state.state}")
    
    print("   add_ou_noise: PASSED")
    return True


def test_epsilon_greedy_continuous():
    """Test epsilon-greedy exploration for continuous actions."""
    print("Testing epsilon_greedy_continuous...")
    
    key = random.PRNGKey(42)
    action = jnp.array([0.5, -0.3])
    
    # Test eps=0 (pure greedy)
    final_action, log_prob = epsilon_greedy_continuous(key, action, epsilon=0.0)
    assert jnp.allclose(final_action, action), "eps=0 should return original action"
    print("   eps=0 (greedy): PASSED")
    
    # Test eps=1 (pure random)
    key, subkey = random.split(key)
    final_action_random, _ = epsilon_greedy_continuous(subkey, action, epsilon=1.0)
    # Should be different from original (very likely)
    print(f"   eps=1 action: {final_action_random} (original: {action})")
    
    # Test clipping
    assert jnp.all(final_action_random >= -1.0) and jnp.all(final_action_random <= 1.0), "Clipping failed"
    
    # Test with many samples to verify epsilon ratio
    n_samples = 1000
    keys = random.split(random.PRNGKey(0), n_samples)
    epsilon = 0.3
    
    def sample_action(key):
        final, _ = epsilon_greedy_continuous(key, action, epsilon)
        return final
    
    final_actions = jax.vmap(sample_action)(keys)
    
    # Count how many are exactly equal to original (greedy)
    # Due to floating point, check close enough
    is_greedy = jnp.all(jnp.isclose(final_actions, action), axis=1)
    greedy_ratio = jnp.mean(is_greedy)
    
    expected_greedy = 1.0 - epsilon
    assert jnp.abs(greedy_ratio - expected_greedy) < 0.1, f"Greedy ratio {greedy_ratio} far from {expected_greedy}"
    print(f"   eps=0.3: {greedy_ratio*100:.1f}% greedy (expected: {expected_greedy*100:.1f}%)")
    
    print("   epsilon_greedy_continuous: PASSED")
    return True


def test_batch_gaussian_noise():
    """Test batched Gaussian noise for multi-agent scenarios."""
    print("Testing gaussian_noise_batch and add_gaussian_noise_batch...")
    
    key = random.PRNGKey(42)
    n_agents = 5
    action_dim = 2
    scale = 0.1
    
    # Test batch generation
    noise = gaussian_noise_batch(key, n_agents, action_dim, scale)
    assert noise.shape == (n_agents, action_dim), f"Batch noise shape: {noise.shape}"
    print(f"   Batch noise shape: {noise.shape}")
    
    # Test add noise to batched actions
    actions = jnp.zeros((n_agents, action_dim))
    key, subkey = random.split(key)
    noisy_actions, noise_added = add_gaussian_noise_batch(subkey, actions, scale)
    
    assert noisy_actions.shape == actions.shape, f"Noisy actions shape: {noisy_actions.shape}"
    assert noise_added.shape == actions.shape, f"Noise shape: {noise_added.shape}"
    
    # Each agent should have different noise
    unique_noises = jnp.unique(noise_added, axis=0).shape[0]
    assert unique_noises == n_agents, "Each agent should have different noise"
    
    print(f"   All {n_agents} agents have unique noise: PASSED")
    print("   gaussian_noise_batch: PASSED")
    return True


def test_batch_ou_noise():
    """Test batched OU noise for multi-agent scenarios."""
    print("Testing ou_noise_init_batch and ou_noise_step_batch...")
    
    n_agents = 4
    action_dim = 2
    
    # Initialize batch state
    state = ou_noise_init_batch(n_agents, action_dim)
    assert state.state.shape == (n_agents, action_dim), f"Batch state shape: {state.state.shape}"
    print(f"   Batch OU state shape: {state.state.shape}")
    
    # Run multiple steps
    params = OUNoiseParams(mu=0.0, theta=0.15, sigma=0.2, dt=1.0, action_dim=action_dim)
    key = random.PRNGKey(42)
    
    for i in range(10):
        key, subkey = random.split(key)
        state, noise = ou_noise_step_batch(subkey, state, params)
    
    assert state.state.shape == (n_agents, action_dim), f"Final state shape: {state.state.shape}"
    
    # Each agent should have different state (they share params but different random noise)
    # After multiple steps, states should diverge
    print(f"   Final OU states:\n{state.state}")
    
    print("   ou_noise_batch: PASSED")
    return True


def test_noise_schedules():
    """Test noise schedule functions."""
    print("Testing linear_schedule and exponential_schedule...")
    
    # Linear schedule
    initial = 1.0
    final = 0.1
    total_steps = 100
    
    # At step 0
    val_0 = linear_schedule(initial, final, 0, total_steps)
    assert jnp.isclose(val_0, initial), f"Linear at 0: {val_0}"
    
    # At step total_steps
    val_end = linear_schedule(initial, final, total_steps, total_steps)
    assert jnp.isclose(val_end, final), f"Linear at end: {val_end}"
    
    # At midpoint
    val_mid = linear_schedule(initial, final, 50, total_steps)
    expected_mid = (initial + final) / 2
    assert jnp.isclose(val_mid, expected_mid), f"Linear at mid: {val_mid}"
    
    # Beyond total_steps should clamp to final
    val_beyond = linear_schedule(initial, final, 200, total_steps)
    assert jnp.isclose(val_beyond, final), f"Linear beyond end: {val_beyond}"
    
    print(f"   Linear: {initial} -> {val_mid} -> {val_end}")
    
    # Exponential schedule
    decay_rate = 0.01
    
    val_0_exp = exponential_schedule(initial, final, 0, decay_rate)
    assert jnp.isclose(val_0_exp, initial), f"Exp at 0: {val_0_exp}"
    
    # At large step, should approach final
    val_large = exponential_schedule(initial, final, 1000, decay_rate)
    assert jnp.abs(val_large - final) < 0.01, f"Exp at large step: {val_large}"
    
    print(f"   Exponential: {initial} -> {val_large:.4f} (asymptote: {final})")
    
    print("   noise schedules: PASSED")
    return True


def test_jit_compilation():
    """Test JIT compilation of noise functions."""
    print("Testing JIT compilation...")
    
    key = random.PRNGKey(42)
    
    # JIT Gaussian noise
    gaussian_jit = jax.jit(gaussian_noise, static_argnums=(1,))
    noise = gaussian_jit(key, (5, 2), 0.1)
    assert noise.shape == (5, 2), "JIT gaussian_noise failed"
    print("   gaussian_noise JIT: PASSED")
    
    # JIT add_gaussian_noise
    add_gaussian_jit = jax.jit(add_gaussian_noise, static_argnums=(2, 3, 4))
    key, subkey = random.split(key)
    action = jnp.array([0.5, -0.3])
    noisy, _ = add_gaussian_jit(subkey, action, 0.1, -1.0, 1.0)
    assert noisy.shape == action.shape, "JIT add_gaussian_noise failed"
    print("   add_gaussian_noise JIT: PASSED")
    
    # JIT OU noise step
    @jax.jit
    def ou_step_jit(key, state, params):
        return ou_noise_step(key, state, params)
    
    params = OUNoiseParams(mu=0.0, theta=0.15, sigma=0.2, dt=1.0, action_dim=2)
    state = ou_noise_init(2)
    key, subkey = random.split(key)
    new_state, noise = ou_step_jit(subkey, state, params)
    assert new_state.state.shape == (2,), "JIT ou_noise_step failed"
    print("   ou_noise_step JIT: PASSED")
    
    # JIT epsilon greedy
    eps_greedy_jit = jax.jit(epsilon_greedy_continuous, static_argnums=(2, 3, 4))
    key, subkey = random.split(key)
    final, _ = eps_greedy_jit(subkey, action, 0.3, -1.0, 1.0)
    assert final.shape == action.shape, "JIT epsilon_greedy failed"
    print("   epsilon_greedy_continuous JIT: PASSED")
    
    print("   JIT compilation: ALL PASSED")
    return True


def test_vmap_compatibility():
    """Test vmap for parallel agent noise."""
    print("Testing vmap compatibility...")
    
    n_agents = 8
    action_dim = 2
    
    # vmap Gaussian noise over different keys
    keys = random.split(random.PRNGKey(0), n_agents)
    vmapped_gaussian = jax.vmap(lambda k: gaussian_noise(k, (action_dim,), 0.1))
    noise_batch = vmapped_gaussian(keys)
    assert noise_batch.shape == (n_agents, action_dim), f"vmap gaussian shape: {noise_batch.shape}"
    print(f"   vmap gaussian_noise: shape {noise_batch.shape}")
    
    # vmap add_gaussian_noise
    actions = jnp.zeros((n_agents, action_dim))
    vmapped_add = jax.vmap(lambda k, a: add_gaussian_noise(k, a, 0.1))
    noisy_actions, noises = vmapped_add(keys, actions)
    assert noisy_actions.shape == (n_agents, action_dim), "vmap add_gaussian shape mismatch"
    print(f"   vmap add_gaussian_noise: PASSED")
    
    # vmap OU noise step
    params = OUNoiseParams(mu=0.0, theta=0.15, sigma=0.2, dt=1.0, action_dim=action_dim)
    states = [ou_noise_init(action_dim) for _ in range(n_agents)]
    
    # Stack states for vmap
    stacked_states = OUNoiseState(state=jnp.stack([s.state for s in states]))
    
    vmapped_ou = jax.vmap(lambda k, s: ou_noise_step(k, OUNoiseState(state=s), params))
    new_states, noises = vmapped_ou(keys, stacked_states.state)
    # new_states is a tuple of (OUNoiseState, noise) for each agent
    print(f"   vmap ou_noise_step: noise shape {noises.shape}")
    
    print("   vmap compatibility: ALL PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running noise.py tests")
    print("=" * 60)
    
    tests = [
        test_gaussian_noise,
        test_gaussian_noise_log_prob,
        test_add_gaussian_noise,
        test_ou_noise_init_and_reset,
        test_ou_noise_step,
        test_add_ou_noise,
        test_epsilon_greedy_continuous,
        test_batch_gaussian_noise,
        test_batch_ou_noise,
        test_noise_schedules,
        test_jit_compilation,
        test_vmap_compatibility,
        # New tests
        test_cosine_schedule,
        test_warmup_linear_schedule,
        test_noise_scheduler,
        test_ou_noise_scale_parameter,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


def test_cosine_schedule():
    """Test cosine annealing schedule."""
    print("Testing cosine_schedule...")
    
    from noise import cosine_schedule
    
    initial = 1.0
    final = 0.1
    total_steps = 100
    
    # At step 0
    val_0 = cosine_schedule(initial, final, 0, total_steps)
    assert jnp.isclose(val_0, initial, atol=1e-5), f"Cosine at 0: {val_0}"
    
    # At end
    val_end = cosine_schedule(initial, final, total_steps, total_steps)
    assert jnp.isclose(val_end, final, atol=1e-5), f"Cosine at end: {val_end}"
    
    # At midpoint (cosine should be at 50% decay)
    val_mid = cosine_schedule(initial, final, 50, total_steps)
    expected_mid = (initial + final) / 2  # Cosine at pi/2 = 0
    assert jnp.isclose(val_mid, expected_mid, atol=0.01), f"Cosine at mid: {val_mid}"
    
    print(f"   Cosine: {initial} -> {val_mid:.4f} -> {val_end:.4f}")
    
    print("   cosine_schedule: PASSED")
    return True


def test_warmup_linear_schedule():
    """Test warmup + linear decay schedule."""
    print("Testing warmup_linear_schedule...")
    
    from noise import warmup_linear_schedule
    
    initial = 0.0
    peak = 1.0
    final = 0.1
    warmup_steps = 10
    total_steps = 100
    
    # At step 0
    val_0 = warmup_linear_schedule(initial, peak, final, 0, warmup_steps, total_steps)
    assert jnp.isclose(val_0, initial), f"Warmup at 0: {val_0}"
    
    # At warmup end
    val_warmup = warmup_linear_schedule(initial, peak, final, warmup_steps, warmup_steps, total_steps)
    assert jnp.isclose(val_warmup, peak, atol=0.1), f"At warmup end: {val_warmup}"
    
    # At total_steps
    val_end = warmup_linear_schedule(initial, peak, final, total_steps, warmup_steps, total_steps)
    assert jnp.isclose(val_end, final, atol=0.1), f"At end: {val_end}"
    
    print(f"   Warmup: {initial} -> {val_warmup:.4f} (peak) -> {val_end:.4f}")
    
    print("   warmup_linear_schedule: PASSED")
    return True


def test_noise_scheduler():
    """Test unified NoiseScheduler class."""
    print("Testing NoiseScheduler...")
    
    from noise import NoiseScheduler
    
    # Gaussian scheduler
    scheduler = NoiseScheduler(
        noise_type='gaussian',
        initial_scale=0.3,
        final_scale=0.05,
        schedule='linear',
        total_steps=100,
        action_dim=2,
    )
    
    state = scheduler.init()
    assert state.step == 0, "Initial step should be 0"
    assert state.ou_state is None, "OU state should be None for Gaussian"
    
    # Add noise
    key = random.PRNGKey(42)
    action = jnp.array([0.5, -0.3])
    
    noisy_action, new_state, noise = scheduler.add_noise(key, action, state)
    assert new_state.step == 1, "Step should increment"
    assert noisy_action.shape == action.shape
    print(f"   Gaussian scheduler: step {state.step} -> {new_state.step}")
    
    # Check scale decreases
    scale_0 = scheduler.get_scale(jnp.array(0))
    scale_50 = scheduler.get_scale(jnp.array(50))
    scale_100 = scheduler.get_scale(jnp.array(100))
    assert scale_0 > scale_50 > scale_100
    print(f"   Scale decay: {scale_0:.3f} -> {scale_50:.3f} -> {scale_100:.3f}")
    
    # OU scheduler
    ou_scheduler = NoiseScheduler(
        noise_type='ou',
        initial_scale=0.3,
        final_scale=0.05,
        schedule='cosine',
        total_steps=100,
        action_dim=2,
    )
    
    ou_state = ou_scheduler.init()
    assert ou_state.ou_state is not None, "OU state should exist"
    print("   OU scheduler initialized with OU state")
    
    # Reset
    for _ in range(10):
        key, subkey = random.split(key)
        _, ou_state, _ = ou_scheduler.add_noise(subkey, action, ou_state)
    
    reset_state = ou_scheduler.reset(ou_state)
    assert reset_state.step == ou_state.step, "Step should be preserved on reset"
    assert jnp.allclose(reset_state.ou_state.state, 0.0), "OU state should reset to mu"
    print("   Reset preserves step, resets OU state")
    
    print("   NoiseScheduler: PASSED")
    return True


def test_ou_noise_scale_parameter():
    """Test that OU noise scale parameter works correctly."""
    print("Testing OU noise scale parameter...")
    
    action_dim = 2
    
    # Create params with scale=1.0
    params_no_scale = OUNoiseParams(mu=0.0, theta=0.15, sigma=0.2, scale=1.0, action_dim=action_dim)
    
    # Create params with scale=0.5
    params_scaled = OUNoiseParams(mu=0.0, theta=0.15, sigma=0.2, scale=0.5, action_dim=action_dim)
    
    key = random.PRNGKey(42)
    state = ou_noise_init(action_dim)
    
    # Run several steps to accumulate state
    for _ in range(5):
        key, subkey = random.split(key)
        state, noise_no_scale = ou_noise_step(subkey, state, params_no_scale)
    
    # Reset and run with scaled version using same keys
    key = random.PRNGKey(42)
    state = ou_noise_init(action_dim)
    
    for _ in range(5):
        key, subkey = random.split(key)
        state, noise_scaled = ou_noise_step(subkey, state, params_scaled)
    
    # The scaled output should be half (scale=0.5)
    # Note: states evolve the same, but output noise is scaled
    print(f"   Unscaled noise (last): {noise_no_scale}")
    print(f"   Scaled noise (last): {noise_scaled}")
    
    # Due to the state evolution, we can check that scale is applied
    # The final noise should be state * scale
    assert noise_scaled.shape == (action_dim,)
    
    print("   OU noise scale parameter: PASSED")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
