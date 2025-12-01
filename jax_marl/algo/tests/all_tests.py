#!/usr/bin/env python
"""Run all tests for the MADDPG algorithm implementation.

This script runs all test modules and provides a summary of results.

Usage:
    python tests/all_tests.py
    
Or run individual test files:
    python tests/test_utils.py
    python tests/test_noise.py
    python tests/test_networks.py
    python tests/test_buffers.py
    python tests/test_agents.py
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f" {text}")
    print(f"{char * 70}")


def run_test_file(test_file: str) -> tuple:
    """Run a single test file and return (success, duration, output)."""
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    
    duration = time.time() - start_time
    success = result.returncode == 0
    output = result.stdout + result.stderr
    
    return success, duration, output


def extract_test_counts(output: str) -> tuple:
    """Extract passed/failed counts from test output."""
    # Look for "Results: X passed, Y failed" pattern
    import re
    match = re.search(r'Results:\s*(\d+)\s*passed,\s*(\d+)\s*failed', output)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Also check for pytest format "X passed"
    match = re.search(r'(\d+)\s*passed', output)
    if match:
        passed = int(match.group(1))
        failed_match = re.search(r'(\d+)\s*failed', output)
        failed = int(failed_match.group(1)) if failed_match else 0
        return passed, failed
    
    return 0, 0


def main():
    """Run all tests."""
    print_header("MADDPG ALGORITHM - FULL TEST SUITE")
    print(f"\n  Testing JAX implementation of Multi-Agent DDPG")
    print(f"  Python: {sys.version.split()[0]}")
    
    # Try to get JAX version
    try:
        import jax
        print(f"  JAX: {jax.__version__}")
        print(f"  Devices: {jax.devices()}")
    except ImportError:
        print("  JAX: Not installed!")
    
    # Define test files in order of dependency
    test_files = [
        ("Utilities", "tests/test_utils.py"),
        ("Noise", "tests/test_noise.py"),
        ("Networks", "tests/test_networks.py"),
        ("Replay Buffers", "tests/test_buffers.py"),
        ("DDPG Agents", "tests/test_agents.py"),
        ("MADDPG Algorithm", "tests/test_maddpg.py"),
    ]
    
    results = {}
    total_duration = 0
    total_passed = 0
    total_failed = 0
    
    for name, test_file in test_files:
        print_header(f"Running: {name}", "-")
        
        # Check if file exists
        test_path = Path(__file__).parent.parent / test_file
        if not test_path.exists():
            print(f"  ⚠ Test file not found: {test_file}")
            results[name] = (False, 0, "File not found", 0, 0)
            continue
        
        success, duration, output = run_test_file(str(test_path))
        passed, failed = extract_test_counts(output)
        results[name] = (success, duration, output, passed, failed)
        total_duration += duration
        total_passed += passed
        total_failed += failed
        
        # Show brief status
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status} - {passed} tests ({duration:.2f}s)")
        
        # If failed, show output
        if not success:
            print("\n  Output (last 30 lines):")
            for line in output.split('\n')[-30:]:
                print(f"    {line}")
    
    # Summary
    print_header("TEST SUMMARY")
    
    modules_passed = sum(1 for s, _, _, _, _ in results.values() if s)
    modules_total = len(results)
    
    print(f"\n  {'Module':<20} {'Status':<10} {'Tests':<12} {'Time':<10}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10}")
    
    for name, (success, duration, _, passed, failed) in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        tests_str = f"{passed}/{passed + failed}"
        print(f"  {name:<20} {status:<10} {tests_str:<12} {duration:.2f}s")
    
    print(f"\n  {'-'*54}")
    print(f"  {'TOTAL':<20} {modules_passed}/{modules_total:<7} {total_passed}/{total_passed + total_failed:<9} {total_duration:.2f}s")
    
    # Final verdict
    if modules_passed == modules_total and total_failed == 0:
        print_header("ALL TESTS PASSED! ✓")
        print("""
  The MADDPG JAX implementation is fully functional!
  
  Modules tested:
  ─────────────────────────────────────────────────
  
  • utils.py - Core utility functions
    - soft_update, hard_update (target network updates)
    - gumbel_softmax, onehot_from_logits (discrete actions)
    - clip_by_global_norm (gradient clipping)
    - explained_variance, normalize_advantages, huber_loss
    
  • noise.py - Exploration noise
    - GaussianNoise with log probability
    - Ornstein-Uhlenbeck noise with scale parameter
    - Noise schedules (linear, exponential, cosine)
    - NoiseScheduler unified interface
    - Batched noise for multi-agent
    
  • networks.py - Neural network architectures
    - Actor (continuous), ActorDiscrete (discrete)
    - Critic, CriticTwin (for TD3-style updates)
    - ActorCritic combined network
    - Layer normalization and dropout support
    - Multi-agent network creation utilities
    
  • buffers.py - Experience replay
    - ReplayBuffer with JIT-compatible operations
    - PerAgentReplayBuffer for MADDPG
    - Log probabilities and action priors storage
    - Sample without replacement
    
  • agents.py - DDPG agent implementation
    - DDPGAgentState (immutable, functional)
    - Action selection with noise
    - Critic update (TD learning)
    - Actor update (policy gradient)
    - Target network soft updates
    - Multi-agent utilities
  
  • maddpg.py - MADDPG algorithm coordinator
    - MADDPGState for complete training state
    - Centralized training, decentralized execution
    - Action prior regularization support
    - Noise scheduling during training
    - Save/load functionality
    
  ─────────────────────────────────────────────────
  
  Quick Start:
  ─────────────────────────────────────────────────
    from maddpg import make_maddpg
    
    # Create MADDPG
    maddpg = make_maddpg(n_agents=3, obs_dims=10, action_dims=2)
    state = maddpg.init(key)
    
    # Select actions
    actions, log_probs, state = maddpg.select_actions(key, state, observations)
    
    # Store transition
    state = maddpg.store_transition(state, obs, actions, rewards, next_obs, dones)
    
    # Update
    state, info = maddpg.update(key, state)
""")
        return 0
    else:
        print_header("SOME TESTS FAILED")
        print(f"\n  {modules_total - modules_passed} module(s) failed")
        print(f"  {total_failed} individual test(s) failed")
        print(f"\n  Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
