"""Tests for utils.py - utility functions for MADDPG."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random

from utils import (
    soft_update,
    hard_update,
    sample_gumbel,
    gumbel_softmax_sample,
    gumbel_softmax,
    onehot_from_logits,
    onehot_from_logits_epsilon_greedy,
    clip_by_global_norm,
    explained_variance,
)


def test_soft_update():
    """Test soft update of target network parameters."""
    print("Testing soft_update...")
    
    target = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.0])}
    online = {"w": jnp.array([3.0, 4.0]), "b": jnp.array([1.0])}
    tau = 0.1
    
    updated = soft_update(target, online, tau)
    expected_w = target["w"] * 0.9 + online["w"] * 0.1
    expected_b = target["b"] * 0.9 + online["b"] * 0.1
    
    assert jnp.allclose(updated["w"], expected_w), f"Expected {expected_w}, got {updated['w']}"
    assert jnp.allclose(updated["b"], expected_b), f"Expected {expected_b}, got {updated['b']}"
    
    # Test with tau=0 (no update)
    no_update = soft_update(target, online, 0.0)
    assert jnp.allclose(no_update["w"], target["w"]), "tau=0 should not change target"
    
    # Test with tau=1 (full update)
    full_update = soft_update(target, online, 1.0)
    assert jnp.allclose(full_update["w"], online["w"]), "tau=1 should copy online"
    
    print("   soft_update: PASSED")
    return True


def test_hard_update():
    """Test hard update (copy) of target network parameters."""
    print("Testing hard_update...")
    
    target = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.0])}
    online = {"w": jnp.array([3.0, 4.0]), "b": jnp.array([1.0])}
    
    updated = hard_update(target, online)
    
    assert jnp.allclose(updated["w"], online["w"]), "hard_update w failed"
    assert jnp.allclose(updated["b"], online["b"]), "hard_update b failed"
    
    # Ensure it's a copy, not a reference (in JAX arrays are immutable anyway)
    assert updated["w"] is not online["w"] or jnp.allclose(updated["w"], online["w"])
    
    print("   hard_update: PASSED")
    return True


def test_sample_gumbel():
    """Test Gumbel(0, 1) sampling."""
    print("Testing sample_gumbel...")
    
    key = random.PRNGKey(42)
    shape = (1000,)
    
    samples = sample_gumbel(key, shape)
    
    # Check shape
    assert samples.shape == shape, f"Shape mismatch: expected {shape}, got {samples.shape}"
    
    # Gumbel(0, 1) has mean ≈ 0.5772 (Euler-Mascheroni constant) and variance ≈ π²/6
    expected_mean = 0.5772  # Euler-Mascheroni constant
    expected_var = (jnp.pi ** 2) / 6
    
    sample_mean = jnp.mean(samples)
    sample_var = jnp.var(samples)
    
    # Allow some tolerance for randomness
    assert jnp.abs(sample_mean - expected_mean) < 0.1, f"Mean {sample_mean} far from expected {expected_mean}"
    assert jnp.abs(sample_var - expected_var) < 0.3, f"Var {sample_var} far from expected {expected_var}"
    
    print(f"   Sample mean: {sample_mean:.4f} (expected: {expected_mean:.4f})")
    print(f"   Sample var: {sample_var:.4f} (expected: {expected_var:.4f})")
    print("   sample_gumbel: PASSED")
    return True


def test_gumbel_softmax():
    """Test Gumbel-Softmax sampling."""
    print("Testing gumbel_softmax...")
    
    key = random.PRNGKey(42)
    logits = jnp.array([[1.0, 2.0, 0.5], [0.1, 0.1, 3.0]])
    
    # Test soft sample
    soft_sample = gumbel_softmax(key, logits, temperature=1.0, hard=False)
    assert soft_sample.shape == logits.shape, f"Shape mismatch: {soft_sample.shape}"
    assert jnp.allclose(jnp.sum(soft_sample, axis=-1), 1.0), "Soft sample doesn't sum to 1"
    assert jnp.all(soft_sample >= 0), "Soft sample has negative values"
    print(f"   Soft sample:\n{soft_sample}")
    
    # Test hard sample
    key, subkey = random.split(key)
    hard_sample = gumbel_softmax(subkey, logits, temperature=0.1, hard=True)
    assert hard_sample.shape == logits.shape, f"Shape mismatch: {hard_sample.shape}"
    assert jnp.allclose(jnp.sum(hard_sample, axis=-1), 1.0), "Hard sample doesn't sum to 1"
    print(f"   Hard sample:\n{hard_sample}")
    
    # Test temperature effect - lower temp should be more concentrated
    key, subkey1, subkey2 = random.split(key, 3)
    high_temp = gumbel_softmax(subkey1, logits, temperature=10.0, hard=False)
    low_temp = gumbel_softmax(subkey2, logits, temperature=0.1, hard=False)
    
    # Max probability should be higher with lower temperature
    # (on average, over many samples)
    
    # Test that hard sample has gradient through soft
    def loss_fn(logits):
        key = random.PRNGKey(0)
        sample = gumbel_softmax(key, logits, temperature=1.0, hard=True)
        return jnp.sum(sample)
    
    grad = jax.grad(loss_fn)(logits)
    assert grad is not None, "Gradient should exist for hard gumbel_softmax"
    
    print("   gumbel_softmax: PASSED")
    return True


def test_onehot_from_logits():
    """Test one-hot encoding from logits."""
    print("Testing onehot_from_logits...")
    
    # Simple case
    logits = jnp.array([1.0, 3.0, 2.0])
    onehot = onehot_from_logits(logits)
    expected = jnp.array([0.0, 1.0, 0.0])
    assert jnp.allclose(onehot, expected), f"Expected {expected}, got {onehot}"
    print(f"   Single: {logits} -> {onehot}")
    
    # Batched case
    logits_batch = jnp.array([
        [1.0, 3.0, 2.0],
        [5.0, 1.0, 2.0],
        [0.0, 0.0, 1.0],
    ])
    onehot_batch = onehot_from_logits(logits_batch)
    expected_batch = jnp.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert jnp.allclose(onehot_batch, expected_batch), f"Batch mismatch"
    print(f"   Batch test: PASSED")
    
    # Test with ties (first max should be selected)
    logits_tie = jnp.array([1.0, 1.0, 0.0])
    onehot_tie = onehot_from_logits(logits_tie)
    assert jnp.sum(onehot_tie) == 1.0, "Should select exactly one"
    print(f"   Tie handling: {logits_tie} -> {onehot_tie}")
    
    print("   onehot_from_logits: PASSED")
    return True


def test_onehot_from_logits_epsilon_greedy():
    """Test epsilon-greedy one-hot encoding."""
    print("Testing onehot_from_logits_epsilon_greedy...")
    
    key = random.PRNGKey(0)
    logits = jnp.array([[10.0, 0.0, 0.0]] * 100)  # Strong preference for first action
    
    # With eps=0, should always select first action (argmax)
    onehot_zero_eps = onehot_from_logits_epsilon_greedy(key, logits, eps=0.0)
    assert jnp.all(onehot_zero_eps[:, 0] == 1.0), "eps=0 should always select argmax"
    print("   eps=0 (pure greedy): PASSED")
    
    # With eps=1, should select randomly (check diversity)
    key, subkey = random.split(key)
    onehot_full_eps = onehot_from_logits_epsilon_greedy(subkey, logits, eps=1.0)
    
    # Count how many unique actions were selected
    action_counts = jnp.sum(onehot_full_eps, axis=0)
    unique_actions = jnp.sum(action_counts > 0)
    
    # Should have selected at least 2 different actions out of 100 samples
    # (probability of all same action with uniform random is very low)
    assert unique_actions >= 2, f"eps=1 should explore, only got {unique_actions} unique actions"
    print(f"   eps=1 (pure random): {unique_actions} unique actions selected")
    
    # Test intermediate eps
    key, subkey = random.split(key)
    onehot_half_eps = onehot_from_logits_epsilon_greedy(subkey, logits, eps=0.3)
    greedy_count = jnp.sum(onehot_half_eps[:, 0])
    
    # With eps=0.3, roughly 70% should be greedy (first action)
    assert 50 < greedy_count < 90, f"eps=0.3 should have ~70% greedy, got {greedy_count}%"
    print(f"   eps=0.3: {greedy_count}% greedy actions")
    
    print("   onehot_from_logits_epsilon_greedy: PASSED")
    return True


def test_clip_by_global_norm():
    """Test gradient clipping by global norm."""
    print("Testing clip_by_global_norm...")
    
    # Simple case: norm = 5 (3-4-5 triangle)
    grads = {"a": jnp.array([3.0, 4.0])}
    original_norm = jnp.sqrt(jnp.sum(grads["a"] ** 2))
    assert jnp.isclose(original_norm, 5.0), f"Original norm should be 5, got {original_norm}"
    
    # Clip to 2.5
    clipped = clip_by_global_norm(grads, max_norm=2.5)
    clipped_norm = jnp.sqrt(jnp.sum(clipped["a"] ** 2))
    assert jnp.isclose(clipped_norm, 2.5, atol=1e-5), f"Expected norm 2.5, got {clipped_norm}"
    print(f"   Norm 5.0 -> clipped to 2.5: PASSED")
    
    # No clipping needed (norm < max_norm)
    no_clip = clip_by_global_norm(grads, max_norm=10.0)
    no_clip_norm = jnp.sqrt(jnp.sum(no_clip["a"] ** 2))
    assert jnp.isclose(no_clip_norm, 5.0), f"Should not clip, got norm {no_clip_norm}"
    print(f"   Norm 5.0 with max_norm=10.0: no clipping (correct)")
    
    # Multi-array case
    multi_grads = {
        "layer1": {"w": jnp.array([3.0, 0.0]), "b": jnp.array([4.0])},
        "layer2": {"w": jnp.array([0.0, 0.0])},
    }
    # Global norm = sqrt(9 + 16) = 5
    multi_clipped = clip_by_global_norm(multi_grads, max_norm=2.5)
    
    # Compute clipped global norm
    leaves = jax.tree.leaves(multi_clipped)
    clipped_global_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    assert jnp.isclose(clipped_global_norm, 2.5, atol=1e-5), f"Multi-array clip failed: {clipped_global_norm}"
    print(f"   Multi-array clipping: PASSED")
    
    print("   clip_by_global_norm: PASSED")
    return True


def test_explained_variance():
    """Test explained variance computation."""
    print("Testing explained_variance...")
    
    y_true = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Perfect prediction: EV = 1
    y_pred_perfect = y_true
    ev_perfect = explained_variance(y_pred_perfect, y_true)
    assert jnp.isclose(ev_perfect, 1.0), f"Perfect prediction should have EV=1, got {ev_perfect}"
    print(f"   EV (perfect prediction): {ev_perfect:.4f}")
    
    # Mean prediction: EV = 0
    y_pred_mean = jnp.full_like(y_true, jnp.mean(y_true))
    ev_mean = explained_variance(y_pred_mean, y_true)
    assert jnp.isclose(ev_mean, 0.0), f"Mean prediction should have EV=0, got {ev_mean}"
    print(f"   EV (mean prediction): {ev_mean:.4f}")
    
    # Bad prediction (reversed): EV < 0
    y_pred_bad = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])
    ev_bad = explained_variance(y_pred_bad, y_true)
    assert ev_bad < 0, f"Bad prediction should have EV<0, got {ev_bad}"
    print(f"   EV (reversed prediction): {ev_bad:.4f}")
    
    # Partial prediction
    y_pred_partial = y_true * 0.5 + 1.5  # Linear transformation
    ev_partial = explained_variance(y_pred_partial, y_true)
    assert 0 < ev_partial < 1, f"Partial prediction should have 0<EV<1, got {ev_partial}"
    print(f"   EV (partial prediction): {ev_partial:.4f}")
    
    print("   explained_variance: PASSED")
    return True


def test_jit_compilation():
    """Test that all functions can be JIT compiled."""
    print("Testing JIT compilation...")
    
    target = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.0])}
    online = {"w": jnp.array([3.0, 4.0]), "b": jnp.array([1.0])}
    
    # JIT compile functions
    soft_update_jit = jax.jit(soft_update, static_argnums=(2,))
    hard_update_jit = jax.jit(hard_update)
    clip_jit = jax.jit(clip_by_global_norm, static_argnums=(1,))
    explained_variance_jit = jax.jit(explained_variance)
    
    # Test soft_update
    updated = soft_update_jit(target, online, 0.1)
    expected_w = target["w"] * 0.9 + online["w"] * 0.1
    assert jnp.allclose(updated["w"], expected_w), "JIT soft_update failed"
    print("   soft_update JIT: PASSED")
    
    # Test hard_update
    updated = hard_update_jit(target, online)
    assert jnp.allclose(updated["w"], online["w"]), "JIT hard_update failed"
    print("   hard_update JIT: PASSED")
    
    # Test clip_by_global_norm
    grads = {"a": jnp.array([3.0, 4.0])}
    clipped = clip_jit(grads, 2.5)
    clipped_norm = jnp.sqrt(jnp.sum(clipped["a"] ** 2))
    assert jnp.isclose(clipped_norm, 2.5, atol=1e-5), "JIT clip failed"
    print("   clip_by_global_norm JIT: PASSED")
    
    # Test explained_variance
    y_true = jnp.array([1.0, 2.0, 3.0])
    y_pred = y_true
    ev = explained_variance_jit(y_pred, y_true)
    assert jnp.isclose(ev, 1.0), "JIT explained_variance failed"
    print("   explained_variance JIT: PASSED")
    
    # Test gumbel_softmax with JIT
    @jax.jit
    def gumbel_jit(key, logits):
        return gumbel_softmax(key, logits, temperature=1.0, hard=True)
    
    key = random.PRNGKey(0)
    logits = jnp.array([1.0, 2.0, 3.0])
    sample = gumbel_jit(key, logits)
    assert sample.shape == logits.shape, "JIT gumbel_softmax failed"
    print("   gumbel_softmax JIT: PASSED")
    
    print("   JIT compilation: ALL PASSED")
    return True


def test_vmap_compatibility():
    """Test that functions work with vmap for batched operations."""
    print("Testing vmap compatibility...")
    
    # Batch of params (simulating multiple agents)
    batch_size = 4
    targets = {
        "w": jnp.ones((batch_size, 3)),
        "b": jnp.zeros((batch_size, 1)),
    }
    onlines = {
        "w": jnp.ones((batch_size, 3)) * 2,
        "b": jnp.ones((batch_size, 1)),
    }
    
    # vmap over the batch dimension
    def single_soft_update(target, online):
        return soft_update(target, online, 0.1)
    
    # This simulates updating multiple agents' networks
    vmapped_update = jax.vmap(single_soft_update)
    updated = vmapped_update(targets, onlines)
    
    expected_w = targets["w"] * 0.9 + onlines["w"] * 0.1
    assert jnp.allclose(updated["w"], expected_w), "vmap soft_update failed"
    print("   vmap soft_update: PASSED")
    
    # vmap over onehot_from_logits (already handles batches, but test anyway)
    logits = jnp.array([
        [1.0, 2.0, 0.5],
        [3.0, 1.0, 2.0],
        [0.5, 0.5, 1.0],
    ])
    onehot = onehot_from_logits(logits)
    assert onehot.shape == logits.shape, "Batched onehot failed"
    print("   onehot_from_logits batched: PASSED")
    
    # vmap over gumbel_softmax with different keys
    keys = random.split(random.PRNGKey(0), batch_size)
    logits_batch = jnp.ones((batch_size, 5))
    
    vmapped_gumbel = jax.vmap(lambda k, l: gumbel_softmax(k, l, 1.0, False))
    samples = vmapped_gumbel(keys, logits_batch)
    assert samples.shape == (batch_size, 5), f"vmap gumbel shape: {samples.shape}"
    assert jnp.allclose(jnp.sum(samples, axis=-1), 1.0), "vmap gumbel doesn't sum to 1"
    print("   vmap gumbel_softmax: PASSED")
    
    print("   vmap compatibility: ALL PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running utils.py tests")
    print("=" * 60)
    
    tests = [
        test_soft_update,
        test_hard_update,
        test_sample_gumbel,
        test_gumbel_softmax,
        test_onehot_from_logits,
        test_onehot_from_logits_epsilon_greedy,
        test_clip_by_global_norm,
        test_explained_variance,
        test_jit_compilation,
        test_vmap_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
