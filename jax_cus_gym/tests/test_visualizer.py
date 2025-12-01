"""Tests for the visualization module.

Run with: python tests/test_visualizer.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import tempfile
import os


def get_fig_shapes_path() -> str:
    """Get path to the actual shapes file in the fig directory."""
    workspace_root = Path(__file__).parent.parent.parent
    fig_path = workspace_root / "fig" / "results.pkl"
    if fig_path.exists():
        return str(fig_path)
    return None


def get_animations_dir() -> str:
    """Get path to the animations output directory."""
    animations_dir = Path(__file__).parent.parent / "animations"
    animations_dir.mkdir(exist_ok=True)
    return str(animations_dir)


FIG_SHAPES_PATH = get_fig_shapes_path()
ANIMATIONS_DIR = get_animations_dir()


def test_render_frame():
    """Test rendering a single frame."""
    print("Testing render_frame...")
    
    from assembly_env import make_assembly_env
    from visualize import render_frame
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    env, params = make_assembly_env(n_agents=10, **kwargs)
    
    key = random.PRNGKey(42)
    obs, state = env.reset(key, params)
    
    # Take a few steps to get some trajectory
    for _ in range(10):
        key, step_key = random.split(key)
        actions = random.uniform(step_key, (10, 2), minval=-1, maxval=1)
        obs, state, _, _, _ = env.step(step_key, state, actions, params)
    
    # Render frame
    fig, ax = render_frame(state, params)
    
    assert fig is not None
    assert ax is not None
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
    
    fig.savefig(temp_path)
    assert os.path.exists(temp_path)
    assert os.path.getsize(temp_path) > 0
    
    os.unlink(temp_path)
    plt.close(fig)
    
    print("  ✓ render_frame passed")


def test_render_frame_options():
    """Test render_frame with different options."""
    print("Testing render_frame options...")
    
    from assembly_env import make_assembly_env
    from visualize import render_frame
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    env, params = make_assembly_env(n_agents=6, **kwargs)
    
    key = random.PRNGKey(0)
    obs, state = env.reset(key, params)
    
    # Test with all options disabled
    fig, ax = render_frame(
        state, params,
        show_velocities=False,
        show_trajectories=False,
        show_grid=False,
        show_ids=False,
        show_metrics=False,
        title="Custom Title"
    )
    
    assert ax.get_title() == "Custom Title"
    plt.close(fig)
    
    # Test with custom figure size
    fig, ax = render_frame(state, params, figsize=(12, 12))
    assert fig.get_figwidth() == 12
    plt.close(fig)
    
    print("  ✓ render_frame options passed")


def test_render_trajectory():
    """Test rendering a trajectory as frames."""
    print("Testing render_trajectory...")
    
    from assembly_env import make_assembly_env
    from visualize import render_trajectory
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    env, params = make_assembly_env(n_agents=5, max_steps=50, **kwargs)
    
    # Collect states
    key = random.PRNGKey(123)
    states = []
    
    key, reset_key = random.split(key)
    obs, state = env.reset(reset_key, params)
    states.append(state)
    
    for _ in range(20):
        key, step_key = random.split(key)
        actions = random.uniform(step_key, (5, 2), minval=-1, maxval=1)
        obs, state, _, _, _ = env.step(step_key, state, actions, params)
        states.append(state)
    
    # Render frames
    frames = render_trajectory(states[:5], params)  # Just 5 frames for speed
    
    assert len(frames) == 5
    assert all(f.shape[2] == 3 for f in frames)  # RGB
    assert all(f.dtype == 'uint8' for f in frames)
    
    print(f"  Frame shape: {frames[0].shape}")
    print("  ✓ render_trajectory passed")


def test_create_animation():
    """Test creating an animation."""
    print("Testing create_animation...")
    
    from assembly_env import make_assembly_env
    from visualize import create_animation
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    env, params = make_assembly_env(n_agents=5, max_steps=50, **kwargs)
    
    # Collect states
    key = random.PRNGKey(456)
    states = []
    
    key, reset_key = random.split(key)
    obs, state = env.reset(reset_key, params)
    states.append(state)
    
    for _ in range(15):
        key, step_key = random.split(key)
        actions = random.uniform(step_key, (5, 2), minval=-1, maxval=1)
        obs, state, _, _, _ = env.step(step_key, state, actions, params)
        states.append(state)
    
    # Create animation (without saving)
    anim = create_animation(states, params, fps=5, show=False)
    
    assert anim is not None
    
    # Test saving to GIF in animations directory
    save_path = os.path.join(ANIMATIONS_DIR, "test_random_policy.gif")
    
    anim = create_animation(states[:10], params, save_path=save_path, fps=5)
    
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0
    
    print("  ✓ create_animation passed")


def test_visualizer_class():
    """Test AssemblyVisualizer class."""
    print("Testing AssemblyVisualizer class...")
    
    from assembly_env import make_assembly_env
    from visualize import AssemblyVisualizer
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    env, params = make_assembly_env(n_agents=8, **kwargs)
    
    # Create visualizer
    vis = AssemblyVisualizer(env, params)
    
    key = random.PRNGKey(789)
    obs, state = env.reset(key, params)
    
    # Test render
    fig, ax = vis.render(state)
    assert fig is not None
    plt.close(fig)
    
    # Test render with custom options
    fig, ax = vis.render(state, show_velocities=False)
    plt.close(fig)
    
    # Test save_frame
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
    
    vis.save_frame(state, temp_path)
    assert os.path.exists(temp_path)
    os.unlink(temp_path)
    
    print("  ✓ AssemblyVisualizer class passed")


def test_collect_episode_states():
    """Test collecting states during episode."""
    print("Testing collect_episode_states...")
    
    from assembly_env import make_assembly_env
    from visualize.renderer import collect_episode_states
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    env, params = make_assembly_env(n_agents=5, max_steps=30, **kwargs)
    
    key = random.PRNGKey(999)
    
    # Collect with no policy (zero actions)
    states = collect_episode_states(env, params, key, max_steps=20)
    
    assert len(states) > 0
    assert len(states) <= 21  # Initial + up to 20 steps
    
    # Collect with random policy
    def random_policy(key, obs):
        return random.uniform(key, (5, 2), minval=-1, maxval=1)
    
    states = collect_episode_states(env, params, key, policy_fn=random_policy, max_steps=15)
    
    assert len(states) > 0
    
    print(f"  Collected {len(states)} states")
    print("  ✓ collect_episode_states passed")


def test_with_prior_policy():
    """Test visualization with prior policy."""
    print("Testing visualization with prior policy...")
    
    from assembly_env import make_assembly_env
    from assembly_env import compute_prior_policy
    from visualize import AssemblyVisualizer, create_animation
    from visualize.renderer import collect_episode_states
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    env, params = make_assembly_env(n_agents=10, **kwargs)
    
    # Define prior policy using the actual compute_prior_policy function
    def prior_policy(key, state):
        # Use the actual prior policy from assembly_env
        # compute_prior_policy needs: positions, velocities, grid_centers, grid_mask, l_cell, r_avoid, d_sen
        # r_avoid is stored as collision_threshold in reward_params
        actions = compute_prior_policy(
            positions=state.positions,
            velocities=state.velocities,
            grid_centers=state.grid_centers,
            grid_mask=state.grid_mask,
            l_cell=state.l_cell,
            r_avoid=params.reward_params.collision_threshold,
            d_sen=params.d_sen,
        )
        return actions
    
    key = random.PRNGKey(42)
    
    # Collect states - use policy_uses_state=True since prior_policy needs state
    states = collect_episode_states(
        env, params, key, 
        policy_fn=prior_policy, 
        max_steps=100,
        policy_uses_state=True
    )
    
    # Create visualization
    vis = AssemblyVisualizer(env, params)
    
    # Save final frame to animations directory
    frame_path = os.path.join(ANIMATIONS_DIR, "prior_policy_final_frame.png")
    vis.save_frame(states[-1], frame_path, show_trajectories=True)
    assert os.path.exists(frame_path)
    print(f"  Saved final frame to: {frame_path}")
    
    # Save animation to animations directory
    anim_path = os.path.join(ANIMATIONS_DIR, "prior_policy_episode.gif")
    create_animation(states, params, save_path=anim_path, fps=10, show_trajectories=True)
    assert os.path.exists(anim_path)
    print(f"  Saved animation to: {anim_path}")
    
    print(f"  Visualized {len(states)} steps")
    print("  ✓ visualization with prior policy passed")


def run_all_tests():
    """Run all visualization tests."""
    print("\n" + "="*60)
    print("VISUALIZATION MODULE TESTS")
    print("="*60)
    
    if FIG_SHAPES_PATH:
        print(f"\nUsing actual shapes from: {FIG_SHAPES_PATH}\n")
    else:
        print("\nUsing procedural shapes\n")
    
    tests = [
        ("Render Frame", test_render_frame),
        ("Render Frame Options", test_render_frame_options),
        ("Render Trajectory", test_render_trajectory),
        ("Create Animation", test_create_animation),
        ("Visualizer Class", test_visualizer_class),
        ("Collect Episode States", test_collect_episode_states),
        ("With Prior Policy", test_with_prior_policy),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = True
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*60)
        print("ALL VISUALIZATION TESTS PASSED! ✓")
        print("="*60)
        return True
    else:
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
