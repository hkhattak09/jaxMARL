"""Evaluation module for Assembly Swarm Environment.

This module provides tools to evaluate trained models on fixed target shapes
without any domain randomization (no rotation, scaling, or position shifting).
"""

from .evaluate import (
    evaluate_on_all_shapes,
    evaluate_single_shape,
    EvalConfig,
    EvalResult,
    ShapeResult,
)

__all__ = [
    "evaluate_on_all_shapes",
    "evaluate_single_shape",
    "EvalConfig",
    "EvalResult",
    "ShapeResult",
]
