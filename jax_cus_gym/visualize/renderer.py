"""Renderer for Assembly Swarm Environment visualization.

Provides functions to render environment states as matplotlib figures
and create animations from state sequences.

Compatible with WSL2 - uses Agg backend by default for file output,
but can use TkAgg or other backends for interactive display.
"""

import os
import sys
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

# Set matplotlib backend before importing pyplot
# For WSL2 compatibility, use Agg by default (file output)
# Set MPLBACKEND=TkAgg for interactive display if X11 forwarding is set up
if 'DISPLAY' not in os.environ and sys.platform != 'win32':
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.colors as mcolors


def render_frame(
    state,
    params,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 8),
    show_velocities: bool = True,
    show_trajectories: bool = True,
    show_grid: bool = True,
    show_target_overlay: bool = False,
    binary_image: Optional[np.ndarray] = None,
    shape_bounds: Optional[np.ndarray] = None,
    show_ids: bool = True,
    show_metrics: bool = True,
    agent_colors: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Render a single frame of the environment state.
    
    Args:
        state: AssemblyState object containing current environment state
        params: AssemblyParams object with environment parameters
        ax: Matplotlib axes to draw on (creates new figure if None)
        figsize: Figure size if creating new figure
        show_velocities: Whether to draw velocity arrows
        show_trajectories: Whether to draw agent trajectories
        show_grid: Whether to show target grid cells
        show_target_overlay: Whether to overlay binary target image
        binary_image: Binary image of target shape (for overlay)
        shape_bounds: [x_min, x_max, y_min, y_max] for image extent
        show_ids: Whether to show agent ID numbers
        show_metrics: Whether to show coverage and collision metrics
        agent_colors: Custom colors for agents (n_agents,) array
        title: Custom title (default: shows simulation time)
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    # Convert JAX arrays to numpy for plotting
    positions = np.array(state.positions)
    velocities = np.array(state.velocities)
    grid_centers = np.array(state.grid_centers)
    grid_mask = np.array(state.grid_mask)
    in_target = np.array(state.in_target)
    is_colliding = np.array(state.is_colliding)
    trajectory = np.array(state.trajectory)
    
    n_agents = positions.shape[0]
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    ax.clear()
    
    # Set up arena boundaries
    arena_size = params.arena_size
    half_arena = arena_size / 2
    
    # Draw arena boundary
    boundary = plt.Rectangle(
        (-half_arena, -half_arena), arena_size, arena_size,
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(boundary)
    
    # Draw target shape
    if show_grid:
        valid_grid = grid_centers[grid_mask.astype(bool)]
        ax.scatter(
            valid_grid[:, 0], valid_grid[:, 1],
            s=20, c='blue', marker='.', alpha=0.3, label='Target cells'
        )
    
    # Draw binary image overlay
    if show_target_overlay and binary_image is not None and shape_bounds is not None:
        ax.imshow(
            binary_image, cmap='gray', origin='lower', aspect='equal', alpha=0.1,
            extent=[shape_bounds[0], shape_bounds[1], shape_bounds[2], shape_bounds[3]]
        )
    
    # Set up agent colors
    if agent_colors is None:
        # Color based on status: green=in target, red=colliding, blue=normal
        colors = []
        for i in range(n_agents):
            if is_colliding[i]:
                colors.append('red')
            elif in_target[i]:
                colors.append('green')
            else:
                colors.append('dodgerblue')
    else:
        colors = agent_colors
    
    # Draw trajectories
    if show_trajectories:
        traj_idx = int(state.traj_idx)
        traj_len = trajectory.shape[0]
        
        for i in range(n_agents):
            # Get trajectory in correct order (circular buffer)
            if state.step_count >= traj_len:
                ordered_traj = np.concatenate([
                    trajectory[traj_idx:, i, :],
                    trajectory[:traj_idx, i, :]
                ], axis=0)
            else:
                ordered_traj = trajectory[:state.step_count, i, :]
            
            if len(ordered_traj) > 1:
                ax.plot(
                    ordered_traj[:, 0], ordered_traj[:, 1],
                    linestyle='-', color=colors[i], alpha=0.3, linewidth=1
                )
    
    # Draw agents as circles
    agent_radius = params.agent_radius
    for i in range(n_agents):
        circle = Circle(
            (positions[i, 0], positions[i, 1]),
            agent_radius,
            facecolor=colors[i],
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(circle)
        
        # Draw agent ID
        if show_ids:
            ax.text(
                positions[i, 0], positions[i, 1],
                str(i), fontsize=6, ha='center', va='center', color='white'
            )
    
    # Draw velocity arrows
    if show_velocities:
        speed_scale = 0.3  # Scale factor for arrow length
        for i in range(n_agents):
            speed = np.linalg.norm(velocities[i])
            if speed > 0.01:  # Only draw if moving
                ax.arrow(
                    positions[i, 0], positions[i, 1],
                    velocities[i, 0] * speed_scale,
                    velocities[i, 1] * speed_scale,
                    head_width=0.05, head_length=0.02,
                    fc=colors[i], ec='black', linewidth=0.3, alpha=0.7
                )
    
    # Set axis limits with padding
    padding = 0.2
    ax.set_xlim(-half_arena - padding, half_arena + padding)
    ax.set_ylim(-half_arena - padding, half_arena + padding)
    ax.set_aspect('equal')
    
    # Labels and grid
    ax.set_xlabel('X position [m]')
    ax.set_ylabel('Y position [m]')
    ax.grid(True, alpha=0.3)
    
    # Title
    if title is None:
        title = f'Time: {state.time:.2f}s (Step {state.step_count})'
    ax.set_title(title)
    
    # Metrics text
    if show_metrics:
        n_in_target = int(np.sum(in_target))
        n_colliding = int(np.sum(is_colliding))
        coverage = n_in_target / n_agents * 100
        
        metrics_text = (
            f'In target: {n_in_target}/{n_agents} ({coverage:.1f}%)\n'
            f'Colliding: {n_colliding}'
        )
        ax.text(
            0.02, 0.98, metrics_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    fig.tight_layout()
    return fig, ax


def render_trajectory(
    states: List,
    params,
    interval: int = 100,
    figsize: Tuple[int, int] = (8, 8),
    **render_kwargs,
) -> List[np.ndarray]:
    """Render a sequence of states as frames.
    
    Args:
        states: List of AssemblyState objects
        params: AssemblyParams object
        interval: Time between frames in ms (for reference)
        figsize: Figure size
        **render_kwargs: Additional arguments passed to render_frame
        
    Returns:
        List of RGB arrays (one per frame)
    """
    frames = []
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for state in states:
        render_frame(state, params, ax=ax, **render_kwargs)
        
        # Convert to RGB array (compatible with newer matplotlib)
        fig.canvas.draw()
        # Use buffer_rgba() which works in modern matplotlib
        buf = np.asarray(fig.canvas.buffer_rgba())
        # Convert RGBA to RGB
        frame = buf[:, :, :3].copy()
        frames.append(frame)
    
    plt.close(fig)
    return frames


def create_animation(
    states: List,
    params,
    save_path: Optional[str] = None,
    fps: int = 10,
    figsize: Tuple[int, int] = (8, 8),
    show: bool = False,
    **render_kwargs,
) -> animation.FuncAnimation:
    """Create an animation from a sequence of states.
    
    Args:
        states: List of AssemblyState objects
        params: AssemblyParams object
        save_path: Path to save animation (e.g., "episode.gif" or "episode.mp4")
        fps: Frames per second
        figsize: Figure size
        show: Whether to display the animation
        **render_kwargs: Additional arguments passed to render_frame
        
    Returns:
        matplotlib animation object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    def init():
        ax.clear()
        return []
    
    def update(frame_idx):
        render_frame(states[frame_idx], params, ax=ax, **render_kwargs)
        return []
    
    interval = 1000 // fps  # Convert fps to ms interval
    
    anim = animation.FuncAnimation(
        fig, update,
        frames=len(states),
        init_func=init,
        interval=interval,
        blit=False
    )
    
    if save_path is not None:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=fps)
        else:
            anim.save(save_path, fps=fps)
        print(f"Animation saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return anim


def show_frame(
    state,
    params,
    block: bool = True,
    **render_kwargs,
) -> None:
    """Display a frame interactively.
    
    For WSL2: Requires X11 forwarding (e.g., VcXsrv, XMing) or WSLg.
    Set DISPLAY environment variable if needed.
    
    Args:
        state: AssemblyState object
        params: AssemblyParams object
        block: Whether to block until window is closed
        **render_kwargs: Additional arguments passed to render_frame
    """
    fig, ax = render_frame(state, params, **render_kwargs)
    plt.show(block=block)


def save_frames_as_images(
    states: List,
    params,
    output_dir: str,
    prefix: str = "frame",
    fmt: str = "png",
    **render_kwargs,
) -> List[str]:
    """Save each state as a separate image file.
    
    Useful for creating videos with external tools or for debugging.
    
    Args:
        states: List of AssemblyState objects
        params: AssemblyParams object
        output_dir: Directory to save images
        prefix: Filename prefix
        fmt: Image format (png, jpg, etc.)
        **render_kwargs: Additional arguments passed to render_frame
        
    Returns:
        List of saved file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    paths = []
    for i, state in enumerate(states):
        fig, ax = render_frame(state, params, **render_kwargs)
        path = os.path.join(output_dir, f"{prefix}_{i:04d}.{fmt}")
        fig.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        paths.append(path)
    
    return paths


class AssemblyVisualizer:
    """Visualizer class for Assembly Swarm Environment.
    
    Provides a stateful interface for rendering and animating environment
    states. Compatible with the gymnax visualizer pattern.
    
    Example:
        ```python
        # Collect states during rollout
        states = []
        obs, state = env.reset(key, params)
        states.append(state)
        
        for _ in range(100):
            obs, state, _, _, _ = env.step(key, state, actions, params)
            states.append(state)
        
        # Visualize
        vis = AssemblyVisualizer(env, params)
        vis.animate(states, save_path="episode.gif")
        ```
    """
    
    def __init__(
        self,
        env,
        params,
        figsize: Tuple[int, int] = (8, 8),
        binary_images: Optional[List[np.ndarray]] = None,
        shape_bounds: Optional[List[np.ndarray]] = None,
    ):
        """Initialize the visualizer.
        
        Args:
            env: AssemblySwarmEnv instance
            params: AssemblyParams object
            figsize: Default figure size
            binary_images: List of binary images for each shape (for overlay)
            shape_bounds: List of [x_min, x_max, y_min, y_max] for each shape
        """
        self.env = env
        self.params = params
        self.figsize = figsize
        self.binary_images = binary_images
        self.shape_bounds = shape_bounds
        
        # Default render options
        self.render_kwargs = {
            'show_velocities': True,
            'show_trajectories': True,
            'show_grid': True,
            'show_target_overlay': False,
            'show_ids': True,
            'show_metrics': True,
        }
    
    def render(
        self,
        state,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Render a single state.
        
        Args:
            state: AssemblyState to render
            ax: Axes to draw on
            **kwargs: Override default render options
            
        Returns:
            fig, ax
        """
        render_opts = {**self.render_kwargs, **kwargs}
        
        # Add binary image if available
        if self.binary_images is not None and render_opts.get('show_target_overlay', False):
            shape_idx = int(state.shape_idx)
            if shape_idx < len(self.binary_images):
                render_opts['binary_image'] = self.binary_images[shape_idx]
                render_opts['shape_bounds'] = self.shape_bounds[shape_idx]
        
        return render_frame(
            state, self.params, ax=ax, figsize=self.figsize, **render_opts
        )
    
    def animate(
        self,
        states: List,
        save_path: Optional[str] = None,
        fps: int = 10,
        show: bool = False,
        **kwargs,
    ) -> animation.FuncAnimation:
        """Create animation from state sequence.
        
        Args:
            states: List of AssemblyState objects
            save_path: Where to save animation
            fps: Frames per second
            show: Whether to display
            **kwargs: Override default render options
            
        Returns:
            Animation object
        """
        render_opts = {**self.render_kwargs, **kwargs}
        
        return create_animation(
            states, self.params,
            save_path=save_path,
            fps=fps,
            figsize=self.figsize,
            show=show,
            **render_opts
        )
    
    def save_frame(
        self,
        state,
        path: str,
        dpi: int = 150,
        **kwargs,
    ):
        """Save a single frame to file.
        
        Args:
            state: AssemblyState to render
            path: Output file path (e.g., "frame.png")
            dpi: Resolution
            **kwargs: Override default render options
        """
        fig, ax = self.render(state, **kwargs)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Frame saved to {path}")


def collect_episode_states(
    env,
    params,
    key,
    policy_fn=None,
    max_steps: Optional[int] = None,
    policy_uses_state: bool = False,
) -> List:
    """Helper function to collect states during an episode.
    
    Args:
        env: AssemblySwarmEnv instance
        params: AssemblyParams
        key: JAX random key
        policy_fn: Policy function. Signature depends on policy_uses_state:
                   - If False: (key, obs) -> actions
                   - If True: (key, state) -> actions
                   If None, uses zero actions.
        max_steps: Maximum steps (default: params.max_steps)
        policy_uses_state: If True, policy_fn receives state instead of obs
        
    Returns:
        List of AssemblyState objects
    """
    import jax.numpy as jnp
    from jax import random
    
    if max_steps is None:
        max_steps = params.max_steps
    
    states = []
    
    key, reset_key = random.split(key)
    obs, state = env.reset(reset_key, params)
    states.append(state)
    
    for _ in range(max_steps):
        key, action_key, step_key = random.split(key, 3)
        
        if policy_fn is not None:
            if policy_uses_state:
                actions = policy_fn(action_key, state)
            else:
                actions = policy_fn(action_key, obs)
        else:
            actions = jnp.zeros((env.n_agents, 2))
        
        obs, state, rewards, dones, info = env.step(step_key, state, actions, params)
        states.append(state)
        
        if state.done:
            break
    
    return states
