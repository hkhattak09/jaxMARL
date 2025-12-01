#!/usr/bin/env python
"""Run all tests for the configuration module.

This script runs all test modules and provides a summary of results.

Usage:
    python tests/all_tests.py
    
Or run individual test files:
    python tests/test_assembly_cfg.py
    python tests/test_llm_cfg.py
    python tests/test_preprocess_shapes.py
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
    import re
    
    # Look for "Results: X passed, Y failed" pattern
    match = re.search(r'Results:\s*(\d+)\s*passed,\s*(\d+)\s*failed', output)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    return 0, 0


def main():
    """Run all tests."""
    print_header("CONFIGURATION MODULE - FULL TEST SUITE")
    print(f"\n  Testing JAX MARL configuration utilities")
    print(f"  Python: {sys.version.split()[0]}")
    
    # Test files to run
    test_dir = Path(__file__).parent
    test_files = [
        test_dir / "test_assembly_cfg.py",
        test_dir / "test_llm_cfg.py",
        test_dir / "test_preprocess_shapes.py",
    ]
    
    # Filter to existing files
    test_files = [f for f in test_files if f.exists()]
    
    if not test_files:
        print("\n  ERROR: No test files found!")
        return False
    
    print(f"  Found {len(test_files)} test files")
    
    # Run each test file
    results = {}
    total_passed = 0
    total_failed = 0
    total_duration = 0.0
    
    for test_file in test_files:
        test_name = test_file.stem
        print_header(f"Running: {test_name}", "-")
        
        success, duration, output = run_test_file(str(test_file))
        passed, failed = extract_test_counts(output)
        
        results[test_name] = {
            "success": success,
            "duration": duration,
            "passed": passed,
            "failed": failed,
            "output": output,
        }
        
        total_passed += passed
        total_failed += failed
        total_duration += duration
        
        # Print abbreviated output
        lines = output.strip().split("\n")
        for line in lines:
            if "✓" in line or "✗" in line or "PASSED" in line or "FAILED" in line:
                print(f"  {line}")
        
        status = "PASSED" if success else "FAILED"
        print(f"\n  Status: {status} ({passed} passed, {failed} failed) in {duration:.2f}s")
    
    # Summary
    print_header("TEST SUMMARY")
    
    all_success = all(r["success"] for r in results.values())
    
    print("\n  Module Results:")
    print("  " + "-" * 50)
    for name, result in results.items():
        status = "✓" if result["success"] else "✗"
        print(f"  {status} {name}: {result['passed']} passed, {result['failed']} failed ({result['duration']:.2f}s)")
    
    print("  " + "-" * 50)
    print(f"\n  Total: {total_passed} passed, {total_failed} failed")
    print(f"  Duration: {total_duration:.2f}s")
    
    if all_success:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print("\n  ❌ SOME TESTS FAILED")
        
        # Show failed test details
        for name, result in results.items():
            if not result["success"]:
                print(f"\n  Failed module: {name}")
                print("  Output:")
                for line in result["output"].split("\n")[-20:]:
                    print(f"    {line}")
    
    return all_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
