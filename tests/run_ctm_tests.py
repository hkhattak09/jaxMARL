"""Run all CTM tests.

Usage:
    python tests/run_ctm_tests.py              # run all, verbose
    python tests/run_ctm_tests.py -x           # stop on first failure
    python tests/run_ctm_tests.py -k encoder   # filter by name

Internally delegates to pytest so the exit code reflects pass/fail.
"""

import sys
import pytest
from pathlib import Path

# All CTM test modules in order (fast → slow)
CTM_TEST_FILES = [
    "jax_marl/algo/tests/test_ctm_validator.py",  # pure Python, no JAX — fastest
    "jax_marl/algo/tests/test_ctm_reorder.py",    # pure JAX, no model init
    "jax_marl/algo/tests/test_ctm_buffer.py",     # buffer ops
    "jax_marl/algo/tests/test_ctm_encoder.py",    # TrajectoryEncoder forward
    "jax_marl/algo/tests/test_ctm_critic.py",     # CTMCritic forward
    "jax_marl/algo/tests/test_ctm_loss.py",       # loss functions
    "jax_marl/algo/tests/test_ctm_update.py",     # gradient updates + D8
]

REPO_ROOT = Path(__file__).parent.parent


def main():
    # Forward any CLI args (e.g. -x, -k, -s) to pytest
    extra_args = sys.argv[1:]

    test_paths = [str(REPO_ROOT / f) for f in CTM_TEST_FILES]

    args = [
        *test_paths,
        "-v",           # verbose output
        "--tb=short",   # concise tracebacks
        *extra_args,
    ]

    print("=" * 60)
    print("Running CTM test suite")
    print("=" * 60)
    for f in CTM_TEST_FILES:
        print(f"  {f}")
    print("=" * 60)

    exit_code = pytest.main(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
