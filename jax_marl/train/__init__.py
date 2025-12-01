"""Training modules for JAX MARL.

This package provides training scripts for multi-agent reinforcement
learning using JAX-accelerated environments and algorithms.

Available trainers:
    - train_assembly: Training for the Assembly Swarm environment
"""

from .train_assembly import (
    train,
    TrainingState,
    TrainingMetrics,
    create_training_state,
)

__all__ = [
    "train",
    "TrainingState", 
    "TrainingMetrics",
    "create_training_state",
]
