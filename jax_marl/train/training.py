#!/usr/bin/env python3
"""Training script for Assembly Swarm Environment.

To configure training, edit the values directly in:
    cfg/assembly_cfg.py

Then run:
    python training.py

To resume from a checkpoint, set CHECKPOINT_PATH below.
"""

import os
# Must be set before JAX initializes (before any jax import).
# Uses CUDA virtual memory API instead of BFC allocator — eliminates
# fragmentation-induced OOMs when peak memory spikes during compilation.
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_assembly import train
from cfg import get_config


# ============================================================================
# Resume Training (set path to resume, or None to start fresh)
# ============================================================================
CHECKPOINT_PATH = None  # e.g., "checkpoints/assembly/2025-12-01_14-30-00/final_checkpoint.pkl"


# ============================================================================
# Run Training
# ============================================================================

if __name__ == "__main__":
    # Get config (edit values in cfg/assembly_cfg.py)
    config = get_config()
    
    print(f"Training with {config.n_agents} agents, {config.n_parallel_envs} parallel envs")
    
    if CHECKPOINT_PATH:
        print(f"Resuming from: {CHECKPOINT_PATH}")
    
    # Train (pass checkpoint_path to resume training)
    train(config, checkpoint_path=CHECKPOINT_PATH)
