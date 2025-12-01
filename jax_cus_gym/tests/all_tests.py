#!/usr/bin/env python
"""Run all tests for the JAX Assembly Swarm Environment.

This script runs all test modules and provides a summary of results.

Usage:
    python tests/all_tests.py
    
Or run individual test files:
    python tests/test_spaces.py
    python tests/test_environment.py
    python tests/test_physics.py
    python tests/test_observations.py
    python tests/test_rewards.py
    python tests/test_assembly_env.py
    python tests/test_maddpg_wrapper.py
    python tests/test_e2e_maddpg.py
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


def main():
    """Run all tests."""
    print_header("JAX ASSEMBLY SWARM ENVIRONMENT - FULL TEST SUITE")
    
    # Define test files in order of dependency
    test_files = [
        ("Spaces", "tests/test_spaces.py"),
        ("Environment Base", "tests/test_environment.py"),
        ("Physics", "tests/test_physics.py"),
        ("Observations", "tests/test_observations.py"),
        ("Rewards", "tests/test_rewards.py"),
        ("Assembly Environment", "tests/test_assembly_env.py"),
        ("MADDPG Wrapper", "tests/test_maddpg_wrapper.py"),
        ("End-to-End MADDPG", "tests/test_e2e_maddpg.py"),
        ("Visualization", "tests/test_visualizer.py"),
    ]
    
    results = {}
    total_duration = 0
    
    for name, test_file in test_files:
        print_header(f"Running: {name}", "-")
        
        # Check if file exists
        test_path = Path(__file__).parent.parent / test_file
        if not test_path.exists():
            print(f"  ⚠ Test file not found: {test_file}")
            results[name] = (False, 0, "File not found")
            continue
        
        success, duration, output = run_test_file(str(test_path))
        results[name] = (success, duration, output)
        total_duration += duration
        
        # Show brief status
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status} ({duration:.2f}s)")
        
        # If failed, show output
        if not success:
            print("\n  Output:")
            for line in output.split('\n')[-20:]:  # Last 20 lines
                print(f"    {line}")
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for s, _, _ in results.values() if s)
    total = len(results)
    
    print(f"\n  {'Module':<25} {'Status':<10} {'Time':<10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    
    for name, (success, duration, _) in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name:<25} {status:<10} {duration:.2f}s")
    
    print(f"\n  {'-'*47}")
    print(f"  {'TOTAL':<25} {passed}/{total:<7} {total_duration:.2f}s")
    
    # Final verdict
    if passed == total:
        print_header("ALL TESTS PASSED! ✓")
        print("""
  The JAX Assembly Swarm Environment is fully functional!
  
  Features tested:
  • Core spaces and environment base classes
  • Physics simulation (collisions, forces, integration)
  • Observations (neighbors, grid, target)
  • Rewards (entering, collision, exploration)
  • Assembly environment with all features:
    - Shape loading from pickle files
    - Domain randomization
    - Trajectory tracking
    - Prior policy
    - Occupied grid tracking
    - Reward sharing modes
  • MADDPG wrapper for multi-agent RL
  • End-to-end training pipeline
  
  Quick Start:
    from assembly_env import make_assembly_env, make_vec_env
    
    # Single environment
    env, params = make_assembly_env(n_agents=10)
    obs, state = env.reset(key, params)
    obs, state, rewards, dones, info = env.step(key, state, actions, params)
    
    # Vectorized environments
    env, params, vec_reset, vec_step = make_vec_env(n_envs=32, n_agents=10)
""")
        return 0
    else:
        print_header("SOME TESTS FAILED")
        print(f"\n  {total - passed} test module(s) failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
