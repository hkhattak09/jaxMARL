#!/usr/bin/env python3
"""Run evaluation on a trained checkpoint.

Configure the settings below, then run:
    python run_eval.py

Or use command-line arguments:
    python run_eval.py --checkpoint path/to/checkpoint.pkl -e 10
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.evaluate import evaluate_on_all_shapes, EvalConfig


# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# Path to the trained checkpoint file
# Example: "checkpoints/assembly/2025-12-01_14-30-00/final_checkpoint.pkl"
CHECKPOINT_PATH = None  # Set this to your checkpoint path, or use --checkpoint

# Path to shapes pickle file (None = use default jaxMARL/fig/results.pkl)
SHAPE_FILE = None

# Output directory for results (None = creates automatically in checkpoint_dir/eval/timestamp/)
# If you set this, results will go to: OUTPUT_DIR/
OUTPUT_DIR = None

# Number of evaluation episodes per shape
N_EPISODES_PER_SHAPE = 5

# Max steps per episode (None = use training config default)
MAX_STEPS = None

# Random seed for reproducibility
SEED = 42

# Whether to save GIF videos of each shape
SAVE_VIDEOS = True

# Video frames per second
VIDEO_FPS = 10

# Print progress to console
VERBOSE = True


# ============================================================================
# Run Evaluation
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    # Check if command-line arguments provided
    parser = argparse.ArgumentParser(description="Evaluate trained model on all shapes")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--shape-file", "-s", type=str, default=None, help="Path to shapes pickle")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--episodes", "-e", type=int, default=None, help="Episodes per shape")
    parser.add_argument("--max-steps", "-m", type=int, default=None, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-videos", action="store_true", help="Disable video saving")
    parser.add_argument("--fps", type=int, default=None, help="Video FPS")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    # Use command-line args if provided, otherwise use config values above
    checkpoint_path = args.checkpoint or CHECKPOINT_PATH
    shape_file = args.shape_file or SHAPE_FILE
    output_dir = args.output_dir or OUTPUT_DIR
    n_episodes = args.episodes if args.episodes is not None else N_EPISODES_PER_SHAPE
    max_steps = args.max_steps if args.max_steps is not None else MAX_STEPS
    seed = args.seed if args.seed is not None else SEED
    save_videos = not args.no_videos and SAVE_VIDEOS
    video_fps = args.fps if args.fps is not None else VIDEO_FPS
    verbose = not args.quiet and VERBOSE
    
    # Validate checkpoint path
    if checkpoint_path is None:
        print("ERROR: No checkpoint path specified!")
        print()
        print("Either:")
        print("  1. Set CHECKPOINT_PATH in this file, or")
        print("  2. Use: python run_eval.py --checkpoint path/to/checkpoint.pkl")
        sys.exit(1)
    
    # Create config
    eval_config = EvalConfig(
        checkpoint_path=checkpoint_path,
        shape_file=shape_file,
        output_dir=output_dir,
        n_episodes_per_shape=n_episodes,
        max_steps=max_steps,
        seed=seed,
        save_videos=save_videos,
        video_fps=video_fps,
        verbose=verbose,
    )
    
    # Run evaluation
    result = evaluate_on_all_shapes(eval_config)
    
    print()
    print(f"Evaluation complete! Results saved to: {eval_config.output_dir}")
