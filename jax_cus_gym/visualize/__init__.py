"""Visualization module for Assembly Swarm Environment.

This module provides visualization utilities for the JAX-based assembly
swarm environment. Visualization is deliberately kept separate from the
core environment to maintain JIT compatibility.

Compatible with WSL2 - uses Agg backend by default for file output.
For interactive display, set DISPLAY environment variable for X11 forwarding
or use WSLg.

Usage:
    from visualize import AssemblyVisualizer, render_frame, create_animation
    
    # Single frame
    fig = render_frame(state, params)
    
    # Save frame to file
    fig.savefig("frame.png")
    
    # Animation from trajectory
    visualizer = AssemblyVisualizer(env, params)
    visualizer.animate(state_sequence, save_path="episode.gif")
    
    # Save frames as individual images
    from visualize.renderer import save_frames_as_images
    save_frames_as_images(states, params, "output_frames/")
"""

from visualize.renderer import (
    AssemblyVisualizer,
    render_frame,
    render_trajectory,
    create_animation,
    show_frame,
    save_frames_as_images,
)

__all__ = [
    "AssemblyVisualizer",
    "render_frame",
    "render_trajectory",
    "create_animation",
    "show_frame",
    "save_frames_as_images",
]
