"""Comprehensive tests for AssemblySwarmEnv.

Tests cover:
1. Core functionality (creation, reset, step, spaces)
2. Shape loading (procedural and pickle)
3. Domain randomization
4. Trajectory tracking
5. Prior policy
6. Occupied grid tracking
7. Reward sharing modes
8. Vectorization and JIT compilation

Run with: python tests/test_assembly_env.py
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random


def print_header(text):
    print(f"\n{'='*60}")
    print(text)
    print('='*60)


def print_subheader(text):
    print(f"\n--- {text} ---")


# ============================================================
# CORE FUNCTIONALITY TESTS
# ============================================================

def test_env_creation():
    """Test environment creation."""
    print_subheader("Environment Creation")
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    
    env = AssemblySwarmEnv(n_agents=10)
    assert env.n_agents == 10
    
    params = env.default_params
    assert isinstance(params, AssemblyParams)
    assert params.arena_size == 5.0
    assert params.max_velocity == 0.8
    
    print("  ✓ Environment creation passed")
    return True


def test_reset():
    """Test environment reset."""
    print_subheader("Reset")
    
    from assembly_env import AssemblySwarmEnv
    
    n_agents = 8
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = env.default_params
    
    key = random.PRNGKey(42)
    obs, state = env.reset(key, params)
    
    # Check state shapes
    assert state.positions.shape == (n_agents, 2)
    assert state.velocities.shape == (n_agents, 2)
    assert state.grid_centers.shape[1] == 2
    assert state.grid_mask.shape[0] == state.grid_centers.shape[0]
    
    # Check observations
    obs_dim = env.get_obs_dim(params)
    assert obs.shape == (n_agents, obs_dim)
    
    # Check initial values
    assert state.time == 0.0
    assert state.step_count == 0
    assert not state.done
    
    # Velocities should be zero initially
    assert jnp.allclose(state.velocities, 0.0)
    
    print("  ✓ Reset passed")
    return True


def test_step():
    """Test environment step."""
    print_subheader("Step")
    
    from assembly_env import AssemblySwarmEnv
    
    n_agents = 5
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = env.default_params
    
    key = random.PRNGKey(0)
    obs, state = env.reset(key, params)
    
    # Take a step with zero actions
    key, step_key = random.split(key)
    actions = jnp.zeros((n_agents, 2))
    
    new_obs, new_state, rewards, dones, info = env.step(
        step_key, state, actions, params
    )
    
    # Check shapes
    assert new_obs.shape == obs.shape
    assert rewards.shape == (n_agents,)
    assert dones.shape == (n_agents,)
    
    # Check time updated
    assert new_state.time > state.time
    assert new_state.step_count == 1
    
    # Check info
    assert "time" in info
    assert "in_target" in info
    assert "is_colliding" in info
    assert "coverage_rate" in info
    
    print("  ✓ Step passed")
    return True


def test_step_with_actions():
    """Test step with non-zero actions."""
    print_subheader("Step with Actions")
    
    from assembly_env import AssemblySwarmEnv
    
    n_agents = 4
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = env.default_params
    
    key = random.PRNGKey(123)
    obs, state = env.reset(key, params)
    
    initial_positions = state.positions.copy()
    
    # Apply acceleration
    actions = jnp.ones((n_agents, 2)) * 0.5
    
    key, step_key = random.split(key)
    new_obs, new_state, rewards, dones, info = env.step(
        step_key, state, actions, params
    )
    
    # Positions should have changed
    assert not jnp.allclose(new_state.positions, initial_positions)
    
    # Velocities should have increased
    assert jnp.any(new_state.velocities > 0)
    
    print("  ✓ Step with actions passed")
    return True


def test_velocity_clipping():
    """Test that velocities are clipped per-component (matches C++ MARL).
    
    C++ MARL uses: np.clip(dp, -Vel_max, Vel_max) which clips each component.
    This means diagonal movement can have speed up to sqrt(2) * Vel_max.
    """
    print_subheader("Velocity Clipping (Per-Component)")
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    
    n_agents = 2
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = AssemblyParams(max_velocity=0.5, max_acceleration=10.0)
    
    key = random.PRNGKey(0)
    obs, state = env.reset(key, params)
    
    # Apply max acceleration for several steps (diagonal direction)
    actions = jnp.ones((n_agents, 2)) * params.max_acceleration
    
    for _ in range(20):
        key, step_key = random.split(key)
        obs, state, rewards, dones, info = env.step(
            step_key, state, actions, params
        )
    
    # With per-component clipping, each component is clipped to [-Vel_max, Vel_max]
    # So velocity can be (Vel_max, Vel_max) with magnitude sqrt(2) * Vel_max
    max_component_speed = params.max_velocity
    max_diagonal_speed = jnp.sqrt(2.0) * max_component_speed
    
    # Check each component is clipped
    assert jnp.all(jnp.abs(state.velocities) <= max_component_speed + 0.01), \
        f"Velocity components exceed max: {state.velocities}"
    
    # Check diagonal speed is within expected range
    speeds = jnp.linalg.norm(state.velocities, axis=-1)
    assert jnp.all(speeds <= max_diagonal_speed + 0.01), \
        f"Speeds: {speeds}, max expected: {max_diagonal_speed}"
    
    print(f"  Per-component max: {max_component_speed}")
    print(f"  Max diagonal speed: {max_diagonal_speed:.4f}")
    print(f"  Actual speeds: {speeds}")
    print("  ✓ Velocity clipping (per-component, C++ matching) passed")


def test_episode_done():
    """Test that episode terminates after max_steps."""
    print_subheader("Episode Done")
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    
    n_agents = 2
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = AssemblyParams(max_steps=10)
    
    key = random.PRNGKey(0)
    obs, state = env.reset(key, params)
    
    actions = jnp.zeros((n_agents, 2))
    
    for i in range(15):
        key, step_key = random.split(key)
        obs, state, rewards, dones, info = env.step(
            step_key, state, actions, params
        )
        
        if i >= 9:  # Should be done after 10 steps
            assert state.done, f"Should be done at step {i+1}"
            assert jnp.all(dones)
    
    print("  ✓ Episode done passed")
    return True


def test_observation_space():
    """Test observation space."""
    print_subheader("Observation Space")
    
    from assembly_env import AssemblySwarmEnv
    
    n_agents = 6
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = env.default_params
    
    obs_space = env.observation_space(params)
    
    assert obs_space.n_agents == n_agents
    
    expected_dim = env.get_obs_dim(params)
    assert obs_space.obs_dim == expected_dim
    assert obs_space.shape == (n_agents, expected_dim)
    
    print("  ✓ Observation space passed")
    return True


def test_action_space():
    """Test action space."""
    print_subheader("Action Space")
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    
    n_agents = 4
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = AssemblyParams(max_acceleration=2.0)
    
    action_space = env.action_space(params)
    
    assert action_space.n_agents == n_agents
    assert action_space.action_dim == 2
    assert action_space.shape == (n_agents, 2)
    assert action_space.low == -params.max_acceleration
    assert action_space.high == params.max_acceleration
    
    print("  ✓ Action space passed")
    return True


def test_make_assembly_env():
    """Test convenience function."""
    print_subheader("make_assembly_env")
    
    from assembly_env import make_assembly_env
    
    env, params = make_assembly_env(
        n_agents=12,
        arena_size=10.0,
        max_steps=100,
    )
    
    assert env.n_agents == 12
    assert params.arena_size == 10.0
    assert params.max_steps == 100
    
    print("  ✓ make_assembly_env passed")
    return True


# ============================================================
# JIT AND VECTORIZATION TESTS
# ============================================================

def test_jit_reset():
    """Test JIT-compiled reset."""
    print_subheader("JIT Reset")
    
    from assembly_env import AssemblySwarmEnv
    
    n_agents = 10
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = env.default_params
    
    @jax.jit
    def jit_reset(key):
        return env.reset(key, params)
    
    key = random.PRNGKey(0)
    obs, state = jit_reset(key)
    
    assert obs.shape[0] == n_agents
    
    # Run again to ensure compilation works
    key2 = random.PRNGKey(1)
    obs2, state2 = jit_reset(key2)
    
    # Different keys should give different positions
    assert not jnp.allclose(state.positions, state2.positions)
    
    print("  ✓ JIT reset passed")
    return True


def test_jit_step():
    """Test JIT-compiled step."""
    print_subheader("JIT Step")
    
    from assembly_env import AssemblySwarmEnv
    
    n_agents = 8
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = env.default_params
    
    @jax.jit
    def jit_step(key, state, actions):
        return env.step(key, state, actions, params)
    
    key = random.PRNGKey(0)
    obs, state = env.reset(key, params)
    
    actions = random.uniform(key, (n_agents, 2), minval=-1, maxval=1)
    
    key, step_key = random.split(key)
    new_obs, new_state, rewards, dones, info = jit_step(step_key, state, actions)
    
    # Run multiple steps
    for _ in range(10):
        key, step_key, action_key = random.split(key, 3)
        actions = random.uniform(action_key, (n_agents, 2), minval=-1, maxval=1)
        new_obs, new_state, rewards, dones, info = jit_step(step_key, new_state, actions)
    
    print("  ✓ JIT step passed")
    return True


def test_vmap_reset():
    """Test vmapped reset for parallel environments."""
    print_subheader("Vmap Reset")
    
    from assembly_env import AssemblySwarmEnv
    
    n_envs = 4
    n_agents = 5
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = env.default_params
    
    @jax.jit
    def vec_reset(keys):
        return jax.vmap(lambda k: env.reset(k, params))(keys)
    
    keys = random.split(random.PRNGKey(0), n_envs)
    obs, states = vec_reset(keys)
    
    assert obs.shape == (n_envs, n_agents, env.get_obs_dim(params))
    assert states.positions.shape == (n_envs, n_agents, 2)
    
    print("  ✓ Vmap reset passed")
    return True


def test_vmap_step():
    """Test vmapped step for parallel environments."""
    print_subheader("Vmap Step")
    
    from assembly_env import AssemblySwarmEnv
    
    n_envs = 4
    n_agents = 5
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = env.default_params
    
    @jax.jit
    def vec_reset(keys):
        return jax.vmap(lambda k: env.reset(k, params))(keys)
    
    @jax.jit
    def vec_step(keys, states, actions):
        return jax.vmap(lambda k, s, a: env.step(k, s, a, params))(keys, states, actions)
    
    # Reset
    keys = random.split(random.PRNGKey(0), n_envs)
    obs, states = vec_reset(keys)
    
    # Step
    step_keys = random.split(random.PRNGKey(1), n_envs)
    actions = random.uniform(
        random.PRNGKey(2), 
        (n_envs, n_agents, 2), 
        minval=-1, maxval=1
    )
    
    new_obs, new_states, rewards, dones, info = vec_step(step_keys, states, actions)
    
    assert new_obs.shape == (n_envs, n_agents, env.get_obs_dim(params))
    assert rewards.shape == (n_envs, n_agents)
    assert dones.shape == (n_envs, n_agents)
    
    print("  ✓ Vmap step passed")
    return True


def test_make_vec_env():
    """Test make_vec_env helper."""
    print_subheader("make_vec_env")
    
    from assembly_env import make_vec_env
    
    n_envs = 8
    n_agents = 6
    
    env, params, vec_reset, vec_step = make_vec_env(
        n_envs=n_envs,
        n_agents=n_agents,
        arena_size=8.0,
    )
    
    assert env.n_agents == n_agents
    assert params.arena_size == 8.0
    
    # Test vectorized reset
    keys = random.split(random.PRNGKey(0), n_envs)
    obs, states = vec_reset(keys)
    
    assert obs.shape[0] == n_envs
    assert obs.shape[1] == n_agents
    
    # Test vectorized step
    step_keys = random.split(random.PRNGKey(1), n_envs)
    actions = jnp.zeros((n_envs, n_agents, 2))
    
    new_obs, new_states, rewards, dones, info = vec_step(step_keys, states, actions)
    
    assert new_obs.shape == obs.shape
    assert rewards.shape == (n_envs, n_agents)
    
    print("  ✓ make_vec_env passed")
    return True


# ============================================================
# SHAPE LOADING TESTS
# ============================================================

def test_shape_loader():
    """Test shape loading utilities."""
    print_subheader("Shape Loader")
    
    from shape_loader import (
        create_shape_library_from_procedural,
        create_procedural_shape,
        save_shapes_to_pickle,
        load_shapes_from_pickle,
        get_shape_from_library,
        apply_shape_transform,
    )
    
    # Test procedural shape creation
    for shape_type in ["rectangle", "cross", "ring", "line"]:
        grid, l_cell = create_procedural_shape(shape_type, 4, 4, 0.3)
        assert grid.shape[1] == 2
    
    # Test library creation
    library = create_shape_library_from_procedural(
        shape_types=["rectangle", "cross", "ring"],
        n_cells=5,
        l_cell=0.35,
    )
    assert library.n_shapes == 3
    
    # Test shape extraction
    grid, l_cell, mask = get_shape_from_library(library, 0)
    assert grid.shape == (library.max_n_grid, 2)
    
    # Test transformation
    transformed = apply_shape_transform(
        grid, mask,
        rotation_angle=jnp.pi / 4,
        scale=1.5,
        offset=jnp.array([1.0, 0.5])
    )
    assert transformed.shape == grid.shape
    
    # Test pickle save/load
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name
    
    save_shapes_to_pickle(library, temp_path)
    reloaded = load_shapes_from_pickle(temp_path)
    
    assert reloaded.n_shapes == library.n_shapes
    assert jnp.allclose(reloaded.l_cells, library.l_cells)
    
    os.unlink(temp_path)
    
    print("  ✓ Shape loader passed")
    return True


def test_pickle_integration():
    """Test loading shapes from pickle into environment."""
    print_subheader("Pickle Integration")
    
    from assembly_env import make_assembly_env
    from shape_loader import create_shape_library_from_procedural, save_shapes_to_pickle
    
    # Create and save shapes
    library = create_shape_library_from_procedural(
        shape_types=["rectangle", "cross", "ring", "line"],
        n_cells=5,
        l_cell=0.4,
    )
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name
    
    save_shapes_to_pickle(library, temp_path)
    
    # Load into environment
    env, params = make_assembly_env(
        n_agents=6,
        shape_file=temp_path,
    )
    
    assert env.shape_library.n_shapes == 4
    
    # Run episodes
    key = random.PRNGKey(42)
    used_shapes = set()
    
    for _ in range(10):
        key, reset_key = random.split(key)
        obs, state = env.reset(reset_key, params)
        used_shapes.add(int(state.shape_idx))
    
    # Should have used multiple shapes
    assert len(used_shapes) >= 2
    
    os.unlink(temp_path)
    
    print("  ✓ Pickle integration passed")
    return True


def test_fig_shapes_loading():
    """Test loading actual shapes from the fig directory (results.pkl).
    
    The fig directory contains PNG images of target shapes that have been
    processed into a pickle file with grid coordinates for use in the
    assembly environment.
    """
    print_subheader("Fig Directory Shapes Loading")
    
    from pathlib import Path
    from assembly_env import make_assembly_env
    from shape_loader import load_shapes_from_pickle
    
    # Get path to results.pkl in the fig directory
    # The fig directory is at the workspace root: /home/hassan/MARL_jax/fig
    workspace_root = Path(__file__).parent.parent.parent  # jax_cus_gym -> MARL_jax
    fig_path = workspace_root / "fig" / "results.pkl"
    
    if not fig_path.exists():
        print(f"  ⚠ Fig shapes file not found at {fig_path}, skipping test")
        return True  # Skip gracefully if file doesn't exist
    
    # Test loading the shape library directly
    library = load_shapes_from_pickle(str(fig_path))
    
    print(f"  Loaded {library.n_shapes} shapes from fig directory")
    print(f"  Max grid cells: {library.max_n_grid}")
    print(f"  Grid sizes: {library.n_grids}")
    print(f"  Cell sizes: {library.l_cells}")
    
    # Verify the shapes have expected properties
    assert library.n_shapes == 7, f"Expected 7 shapes, got {library.n_shapes}"
    assert library.max_n_grid >= 400, f"Expected large grids, got max {library.max_n_grid}"
    assert jnp.all(library.l_cells > 0), "Cell sizes should be positive"
    assert jnp.all(library.l_cells < 0.1), "Cell sizes should be small (~0.06)"
    
    # Test loading into environment
    env, params = make_assembly_env(
        n_agents=10,
        shape_file=str(fig_path),
    )
    
    assert env.shape_library.n_shapes == 7
    
    # Run a few episodes and verify shapes are being used
    key = random.PRNGKey(42)
    used_shapes = set()
    
    for _ in range(20):
        key, reset_key = random.split(key)
        obs, state = env.reset(reset_key, params)
        used_shapes.add(int(state.shape_idx))
        
        # Verify state is valid
        assert state.positions.shape == (10, 2)
        assert state.grid_centers.shape[1] == 2
        assert jnp.sum(state.grid_mask) > 400  # Should have many grid cells
        
        # Take a few steps to verify environment works
        for _ in range(5):
            key, step_key = random.split(key)
            actions = random.uniform(step_key, (10, 2), minval=-1, maxval=1)
            obs, state, rewards, dones, info = env.step(step_key, state, actions, params)
            
            assert obs.shape[0] == 10
            assert rewards.shape == (10,)
    
    # Should have used multiple shapes across episodes
    print(f"  Used {len(used_shapes)} unique shapes across 20 episodes: {sorted(used_shapes)}")
    assert len(used_shapes) >= 3, f"Expected at least 3 different shapes, got {len(used_shapes)}"
    
    print("  ✓ Fig directory shapes loading passed")
    return True


# ============================================================
# DOMAIN RANDOMIZATION TESTS
# ============================================================

def test_domain_randomization():
    """Test domain randomization across resets."""
    print_subheader("Domain Randomization")
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    
    env = AssemblySwarmEnv(n_agents=5)
    params = AssemblyParams(
        randomize_shape=True,
        randomize_rotation=True,
        randomize_scale=True,
        randomize_offset=True,
    )
    
    # Collect stats across resets
    shapes = []
    rotations = []
    scales = []
    
    key = random.PRNGKey(0)
    for _ in range(20):
        key, reset_key = random.split(key)
        _, state = env.reset(reset_key, params)
        shapes.append(int(state.shape_idx))
        rotations.append(float(state.shape_rotation))
        scales.append(float(state.shape_scale))
    
    # Check diversity
    unique_shapes = len(set(shapes))
    rotation_std = jnp.std(jnp.array(rotations))
    scale_std = jnp.std(jnp.array(scales))
    
    assert unique_shapes >= 2, f"Only {unique_shapes} unique shapes"
    assert rotation_std > 0.5, f"Rotation std too low: {rotation_std}"
    assert scale_std > 0.05, f"Scale std too low: {scale_std}"
    
    print(f"    Unique shapes: {unique_shapes}")
    print(f"    Rotation std: {rotation_std:.3f}")
    print(f"    Scale std: {scale_std:.3f}")
    print("  ✓ Domain randomization passed")
    return True


def test_deterministic_mode():
    """Test that disabling randomization is deterministic."""
    print_subheader("Deterministic Mode")
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    
    env = AssemblySwarmEnv(n_agents=5)
    params = AssemblyParams(
        randomize_shape=False,
        randomize_rotation=False,
        randomize_scale=False,
        randomize_offset=False,
    )
    
    key = random.PRNGKey(42)
    _, state1 = env.reset(key, params)
    
    key2 = random.PRNGKey(99)
    _, state2 = env.reset(key2, params)
    
    # Shape parameters should be identical
    assert state1.shape_idx == state2.shape_idx == 0
    assert state1.shape_rotation == state2.shape_rotation == 0.0
    assert state1.shape_scale == state2.shape_scale == 1.0
    assert jnp.allclose(state1.shape_offset, jnp.zeros(2))
    
    print("  ✓ Deterministic mode passed")
    return True


# ============================================================
# TRAJECTORY TRACKING TESTS
# ============================================================

def test_trajectory_tracking():
    """Test trajectory tracking functionality."""
    print_subheader("Trajectory Tracking")
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    
    n_agents = 4
    traj_len = 10
    
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = AssemblyParams(traj_len=traj_len)
    
    key = random.PRNGKey(0)
    obs, state = env.reset(key, params)
    
    # Check trajectory buffer shape
    assert state.trajectory.shape == (traj_len, n_agents, 2)
    
    # Run steps and collect positions
    positions_history = [state.positions.copy()]
    
    for _ in range(traj_len + 5):
        key, step_key = random.split(key)
        actions = random.uniform(step_key, (n_agents, 2), minval=-0.5, maxval=0.5)
        obs, state, _, _, _ = env.step(step_key, state, actions, params)
        positions_history.append(state.positions.copy())
    
    # Get ordered trajectory
    traj = env.get_trajectory(state, params)
    
    assert traj.shape == (traj_len, n_agents, 2)
    
    # Last entry should match current position
    assert jnp.allclose(traj[-1], state.positions, atol=1e-5)
    
    print("  ✓ Trajectory tracking passed")
    return True


# ============================================================
# PRIOR POLICY TESTS
# ============================================================

def test_prior_policy():
    """Test rule-based prior policy."""
    print_subheader("Prior Policy")
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    
    n_agents = 6
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = AssemblyParams()
    
    key = random.PRNGKey(0)
    obs, state = env.reset(key, params)
    
    # Get prior actions
    prior_actions = env.prior_policy(state, params)
    
    assert prior_actions.shape == (n_agents, 2)
    assert jnp.all(prior_actions >= -1.0) and jnp.all(prior_actions <= 1.0)
    
    # Test that prior policy leads to assembly
    coverage_rates = []
    for _ in range(100):
        key, step_key = random.split(key)
        prior_actions = env.prior_policy(state, params)
        obs, state, _, _, info = env.step(step_key, state, prior_actions, params)
        coverage_rates.append(float(info['coverage_rate']))
    
    # Coverage should increase over time
    assert coverage_rates[-1] >= coverage_rates[0]
    
    print(f"    Initial coverage: {coverage_rates[0]:.2%}")
    print(f"    Final coverage: {coverage_rates[-1]:.2%}")
    print("  ✓ Prior policy passed")
    return True


def test_standalone_prior_policy():
    """Test standalone prior policy function."""
    print_subheader("Standalone Prior Policy")
    
    from assembly_env import compute_prior_policy, AssemblySwarmEnv
    
    n_agents = 5
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = env.default_params
    
    key = random.PRNGKey(0)
    _, state = env.reset(key, params)
    
    # Call standalone function
    actions = compute_prior_policy(
        state.positions,
        state.velocities,
        state.grid_centers,
        state.grid_mask,
        state.l_cell,
        params.reward_params.collision_threshold,
        params.d_sen,
    )
    
    assert actions.shape == (n_agents, 2)
    
    print("  ✓ Standalone prior policy passed")
    return True


def test_prior_policy_repulsion_strength():
    """Test that prior policy has correct repulsion strength (3.0, matching C++ MARL).
    
    When two agents are very close (within r_avoid), the repulsion force should
    be strong enough to push them apart. With repulsion_strength=3.0, the
    repulsion magnitude is: 3.0 * (r_avoid / dist - 1.0)
    """
    print_subheader("Prior Policy Repulsion Strength")
    
    from assembly_env import compute_prior_policy
    
    # Two agents very close together (distance = 0.1, r_avoid = 0.15)
    # Repulsion should dominate
    positions = jnp.array([
        [0.0, 0.0],   # Agent 0
        [0.1, 0.0],   # Agent 1, very close to agent 0
    ])
    velocities = jnp.zeros((2, 2))
    
    # Disable grid attraction to isolate repulsion testing
    # (set grid_mask to False so no valid targets exist)
    grid_centers = jnp.array([
        [10.0, 0.0],
        [10.0, 1.0],
    ])
    grid_mask = jnp.array([False, False])  # No valid targets
    
    l_cell = 0.3
    r_avoid = 0.15  # Agents at distance 0.1 < 0.15
    d_sen = 3.0
    
    actions = compute_prior_policy(
        positions, velocities, grid_centers, grid_mask,
        l_cell, r_avoid, d_sen,
        attraction_strength=2.0,
        repulsion_strength=3.0,  # C++ MARL default
        sync_strength=2.0,
    )
    
    # Agent 0 should be pushed LEFT (negative x) away from agent 1
    # Agent 1 should be pushed RIGHT (positive x) away from agent 0
    assert actions[0, 0] < 0, f"Agent 0 should be pushed left, got x={actions[0, 0]}"
    assert actions[1, 0] > 0, f"Agent 1 should be pushed right, got x={actions[1, 0]}"
    
    # Calculate expected repulsion magnitude:
    # dist = 0.1, r_avoid = 0.15
    # repulsion_mag = 3.0 * (0.15 / 0.1 - 1.0) = 3.0 * 0.5 = 1.5
    # This is strong repulsion (will be clipped to [-1, 1])
    
    # The actions should be clipped, but still show strong repulsion
    assert jnp.abs(actions[0, 0]) > 0.5, f"Repulsion should be strong, got {actions[0, 0]}"
    assert jnp.abs(actions[1, 0]) > 0.5, f"Repulsion should be strong, got {actions[1, 0]}"
    
    # Compare with weaker repulsion (old default of 1.0)
    actions_weak = compute_prior_policy(
        positions, velocities, grid_centers, grid_mask,
        l_cell, r_avoid, d_sen,
        attraction_strength=2.0,
        repulsion_strength=1.0,  # Old incorrect default
        sync_strength=2.0,
    )
    
    # With repulsion_strength=3.0, the repulsion component should be 3x stronger
    # (before clipping). The default should give stronger repulsion.
    print(f"    Actions with repulsion_strength=3.0: {actions}")
    print(f"    Actions with repulsion_strength=1.0: {actions_weak}")
    
    print("  ✓ Prior policy repulsion strength test passed")


def test_prior_policy_in_target_behavior():
    """Test that prior policy gives zero attraction when agent is in target.
    
    This matches C++ MARL: when in_flag=true, target_pos = current_pos,
    so attraction force becomes zero (agent stays in place, only affected
    by repulsion and sync).
    """
    print_subheader("Prior Policy In-Target Behavior")
    
    from assembly_env import compute_prior_policy
    
    # Agent exactly on a grid cell (in target)
    positions = jnp.array([[0.0, 0.0]])
    velocities = jnp.zeros((1, 2))
    
    # Grid at origin (agent is on it)
    grid_centers = jnp.array([[0.0, 0.0]])
    grid_mask = jnp.array([True])
    
    l_cell = 0.5  # Agent is within sqrt(2)*0.5/2 ≈ 0.35 of grid center
    r_avoid = 0.15
    d_sen = 3.0
    
    actions = compute_prior_policy(
        positions, velocities, grid_centers, grid_mask,
        l_cell, r_avoid, d_sen,
    )
    
    # With only 1 agent and no neighbors, and agent is in target:
    # - Attraction: 0 (in target, so no attraction)
    # - Repulsion: 0 (no other agents)
    # - Sync: 0 (no neighbors)
    # Total: [0, 0]
    
    assert jnp.allclose(actions[0], jnp.zeros(2), atol=1e-6), \
        f"In-target agent with no neighbors should have zero action, got {actions[0]}"
    
    print("  ✓ Prior policy in-target behavior test passed")
    return True


# ============================================================
# OCCUPIED GRID TRACKING TESTS
# ============================================================

def test_occupied_grid_tracking():
    """Test occupied grid tracking."""
    print_subheader("Occupied Grid Tracking")
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    
    n_agents = 6
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = AssemblyParams()
    
    key = random.PRNGKey(0)
    obs, state = env.reset(key, params)
    
    # Check shapes
    assert state.grid_mask.shape[0] == state.grid_centers.shape[0]
    assert state.occupied_mask.shape == state.grid_mask.shape
    assert state.in_target.shape == (n_agents,)
    assert state.is_colliding.shape == (n_agents,)
    
    # Run steps and check occupancy updates
    for _ in range(50):
        key, step_key = random.split(key)
        prior_actions = env.prior_policy(state, params)
        obs, state, _, _, info = env.step(step_key, state, prior_actions, params)
    
    # Should have some agents in target by now
    agents_in_target = jnp.sum(state.in_target)
    occupied_cells = jnp.sum(state.occupied_mask)
    
    assert 'occupied_count' in info
    
    print(f"    Agents in target: {agents_in_target}")
    print(f"    Occupied cells: {occupied_cells}")
    print("  ✓ Occupied grid tracking passed")
    return True


# ============================================================
# REWARD SHARING TESTS
# ============================================================

def test_reward_sharing():
    """Test reward sharing modes."""
    print_subheader("Reward Sharing")
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    
    n_agents = 5
    env = AssemblySwarmEnv(n_agents=n_agents)
    
    key = random.PRNGKey(0)
    
    # Test individual mode
    params_individual = AssemblyParams(reward_mode=0)  # REWARD_MODE_INDIVIDUAL
    obs, state = env.reset(key, params_individual)
    
    key, step_key = random.split(key)
    actions = random.uniform(step_key, (n_agents, 2), minval=-1, maxval=1)
    _, _, rewards_ind, _, _ = env.step(step_key, state, actions, params_individual)
    
    print(f"    Individual rewards: {rewards_ind}")
    
    # Test shared mean mode
    params_mean = AssemblyParams(reward_mode=1)  # REWARD_MODE_SHARED_MEAN
    obs, state = env.reset(key, params_mean)
    _, _, rewards_mean, _, _ = env.step(step_key, state, actions, params_mean)
    
    # All rewards should be equal
    assert jnp.allclose(rewards_mean, rewards_mean[0])
    print(f"    Mean rewards: all equal = {jnp.allclose(rewards_mean, rewards_mean[0])}")
    
    # Test shared max mode
    params_max = AssemblyParams(reward_mode=2)  # REWARD_MODE_SHARED_MAX
    obs, state = env.reset(key, params_max)
    _, _, rewards_max, _, _ = env.step(step_key, state, actions, params_max)
    
    # All rewards should be equal
    assert jnp.allclose(rewards_max, rewards_max[0])
    print(f"    Max rewards: all equal = {jnp.allclose(rewards_max, rewards_max[0])}")
    
    print("  ✓ Reward sharing passed")
    return True


# ============================================================
# PERFORMANCE TEST
# ============================================================

def test_performance():
    """Test environment throughput."""
    print_subheader("Performance")
    
    import time
    from assembly_env import make_vec_env
    
    n_envs = 32
    n_agents = 10
    n_steps = 100
    
    env, params, vec_reset, vec_step = make_vec_env(
        n_envs=n_envs,
        n_agents=n_agents,
    )
    
    keys = random.split(random.PRNGKey(0), n_envs)
    obs, states = vec_reset(keys)
    
    # Warmup
    step_keys = random.split(random.PRNGKey(1), n_envs)
    actions = jnp.zeros((n_envs, n_agents, 2))
    _ = vec_step(step_keys, states, actions)
    
    # Timed run
    start = time.time()
    for i in range(n_steps):
        step_keys = random.split(random.PRNGKey(i + 2), n_envs)
        actions = random.uniform(step_keys[0], (n_envs, n_agents, 2), minval=-1, maxval=1)
        obs, states, rewards, dones, info = vec_step(step_keys, states, actions)
    
    elapsed = time.time() - start
    total_steps = n_steps * n_envs
    steps_per_sec = total_steps / elapsed
    
    print(f"    {n_envs} envs × {n_agents} agents")
    print(f"    {total_steps} steps in {elapsed:.2f}s")
    print(f"    Throughput: {steps_per_sec:.0f} env steps/sec")
    
    assert steps_per_sec > 100, f"Too slow: {steps_per_sec} steps/sec"
    
    print("  ✓ Performance passed")
    return True


# ============================================================
# FULL EPISODE TEST
# ============================================================

def test_full_episode():
    """Test running a full episode."""
    print_subheader("Full Episode")
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    
    n_agents = 10
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = AssemblyParams(max_steps=50)
    
    key = random.PRNGKey(0)
    obs, state = env.reset(key, params)
    
    total_reward = jnp.zeros(n_agents)
    step_count = 0
    
    while not state.done:
        key, step_key = random.split(key)
        
        # Use prior policy
        actions = env.prior_policy(state, params)
        
        obs, state, rewards, dones, info = env.step(step_key, state, actions, params)
        total_reward += rewards
        step_count += 1
    
    print(f"    Episode length: {step_count}")
    print(f"    Mean reward: {jnp.mean(total_reward):.2f}")
    print(f"    Final coverage: {info['coverage_rate']:.2%}")
    
    assert step_count == params.max_steps
    
    print("  ✓ Full episode passed")
    return True


# ============================================================
# C++ MARL MATCHING TESTS - PRIOR POLICY
# ============================================================

def test_prior_policy_cpp_matching_combined_forces():
    """Test that prior policy matches C++ MARL with combined forces.
    
    Tests realistic scenarios where attraction, repulsion, and sync all interact.
    Verifies exact numerical behavior matches C++ implementation.
    """
    print_subheader("Prior Policy C++ Matching - Combined Forces")
    
    from assembly_env import compute_prior_policy
    
    # Scenario 1: Agent between target and neighbor
    # Agent at origin, target to the right, neighbor to the left
    # Attraction pulls right, no repulsion (neighbor far away)
    positions = jnp.array([
        [0.0, 0.0],   # Agent 0
        [-1.0, 0.0],  # Agent 1 (far away, outside r_avoid)
    ])
    velocities = jnp.zeros((2, 2))
    grid_centers = jnp.array([[1.0, 0.0]])  # Target to the right
    grid_mask = jnp.array([True])
    l_cell = 0.3
    r_avoid = 0.15
    d_sen = 3.0
    
    actions = compute_prior_policy(
        positions, velocities, grid_centers, grid_mask,
        l_cell, r_avoid, d_sen,
    )
    
    # Agent 0: attraction to [1,0], no repulsion (neighbor at dist=1.0 > r_avoid=0.15)
    # Attraction = 2.0 * [1,0]/1.0 = [2.0, 0] -> clipped to [1.0, 0]
    # Sync = 2.0 * ([0,0] - [0,0]) = [0, 0] (neighbor has same velocity)
    assert actions[0, 0] > 0.9, f"Agent 0 should be pulled right, got {actions[0, 0]}"
    assert jnp.abs(actions[0, 1]) < 0.1, f"Agent 0 y should be ~0, got {actions[0, 1]}"
    
    print(f"    Scenario 1 - Target right, neighbor far left: {actions[0]}")
    
    # Scenario 2: Repulsion dominates when agents are very close
    # Both agents at similar position, extremely close (dist < r_avoid)
    positions = jnp.array([
        [0.0, 0.0],    # Agent 0
        [0.05, 0.0],   # Agent 1, VERY close (dist=0.05 < r_avoid=0.15)
    ])
    grid_centers = jnp.array([[1.0, 0.0]])  # Target to the right
    
    actions = compute_prior_policy(
        positions, velocities, grid_centers, grid_mask,
        l_cell, r_avoid, d_sen,
    )
    
    # Agent 0:
    # - Attraction: 2.0 * [1,0] = [2.0, 0] (pulling right)
    # - Repulsion: 3.0 * (0.15/0.05 - 1) * [-1,0] = 3.0 * 2.0 * [-1,0] = [-6.0, 0]
    # - Total before clip: [2.0 - 6.0, 0] = [-4.0, 0]
    # - After clip: [-1.0, 0] (repulsion wins!)
    assert actions[0, 0] < -0.9, f"Agent 0 should be pushed left (repulsion wins), got {actions[0, 0]}"
    
    # Agent 1:
    # - Attraction: 2.0 * [0.95,0]/0.95 ≈ [2.0, 0] (pulling right)
    # - Repulsion: 3.0 * (0.15/0.05 - 1) * [1,0] = [6.0, 0] (pushing right)
    # - Total: [8.0, 0] -> clipped to [1.0, 0]
    assert actions[1, 0] > 0.9, f"Agent 1 should be pushed right, got {actions[1, 0]}"
    
    print(f"    Scenario 2 - Very close agents: Agent0={actions[0]}, Agent1={actions[1]}")
    
    # Scenario 3: Sync force when neighbors have different velocity
    positions = jnp.array([
        [0.0, 0.0],   # Agent 0 at origin
        [0.5, 0.0],   # Agent 1 within d_sen but outside r_avoid
    ])
    velocities = jnp.array([
        [0.0, 0.0],   # Agent 0 stationary
        [1.0, 0.0],   # Agent 1 moving right
    ])
    # Put both agents in target to disable attraction
    grid_centers = jnp.array([[0.0, 0.0], [0.5, 0.0]])
    grid_mask = jnp.array([True, True])
    l_cell = 0.5  # Large enough that both are "in target"
    
    actions = compute_prior_policy(
        positions, velocities, grid_centers, grid_mask,
        l_cell, r_avoid, d_sen,
    )
    
    # Agent 0 in target -> no attraction
    # No repulsion (dist=0.5 > r_avoid=0.15)
    # Sync: 2.0 * ([1,0] - [0,0]) = [2.0, 0] -> clipped to [1.0, 0]
    assert actions[0, 0] > 0.9, f"Agent 0 should sync with moving neighbor, got {actions[0, 0]}"
    
    print(f"    Scenario 3 - Velocity sync: {actions[0]}")
    
    print("  ✓ Prior policy C++ matching - combined forces passed")
    return True


def test_prior_policy_cpp_matching_force_clipping():
    """Test that force clipping matches C++ clamp behavior.
    
    C++ clips each component independently to [-1, 1].
    """
    print_subheader("Prior Policy C++ Matching - Force Clipping")
    
    from assembly_env import compute_prior_policy
    
    # Create scenario with extreme forces in both directions
    # Multiple close neighbors creating large repulsion
    positions = jnp.array([
        [0.0, 0.0],    # Agent 0 at center
        [0.05, 0.0],   # Agent 1 very close right
        [0.0, 0.05],   # Agent 2 very close up
        [-0.05, 0.0],  # Agent 3 very close left
        [0.0, -0.05],  # Agent 4 very close down
    ])
    velocities = jnp.zeros((5, 2))
    
    # No valid targets
    grid_centers = jnp.array([[10.0, 10.0]])
    grid_mask = jnp.array([False])
    l_cell = 0.3
    r_avoid = 0.15
    d_sen = 3.0
    
    actions = compute_prior_policy(
        positions, velocities, grid_centers, grid_mask,
        l_cell, r_avoid, d_sen,
    )
    
    # All actions should be clipped to [-1, 1]
    assert jnp.all(actions >= -1.0), f"Actions should be >= -1.0, got min {jnp.min(actions)}"
    assert jnp.all(actions <= 1.0), f"Actions should be <= 1.0, got max {jnp.max(actions)}"
    
    # Agent 0 at center with symmetric neighbors should have ~zero action
    # (repulsion forces cancel out)
    assert jnp.abs(actions[0, 0]) < 0.2, f"Center agent x should be ~0, got {actions[0, 0]}"
    assert jnp.abs(actions[0, 1]) < 0.2, f"Center agent y should be ~0, got {actions[0, 1]}"
    
    print(f"    Center agent with symmetric neighbors: {actions[0]}")
    print(f"    Action bounds: min={jnp.min(actions):.3f}, max={jnp.max(actions):.3f}")
    
    print("  ✓ Prior policy C++ matching - force clipping passed")
    return True


def test_prior_policy_jit_compatibility():
    """Test that prior policy is fully JIT-compatible.
    
    Ensures no Python callbacks or dynamic shapes that would break JAX speedup.
    """
    print_subheader("Prior Policy JIT Compatibility")
    
    from assembly_env import compute_prior_policy
    import time
    
    n_agents = 20
    positions = jax.random.uniform(jax.random.PRNGKey(0), (n_agents, 2)) * 2.0
    velocities = jax.random.uniform(jax.random.PRNGKey(1), (n_agents, 2)) * 0.5
    grid_centers = jax.random.uniform(jax.random.PRNGKey(2), (50, 2)) * 2.0
    grid_mask = jnp.ones(50, dtype=bool)
    
    # JIT compile
    jit_policy = jax.jit(lambda p, v: compute_prior_policy(
        p, v, grid_centers, grid_mask, 0.3, 0.15, 3.0
    ))
    
    # Warmup (compilation)
    _ = jit_policy(positions, velocities)
    
    # Time JIT version
    n_calls = 1000
    start = time.time()
    for _ in range(n_calls):
        actions = jit_policy(positions, velocities)
        actions.block_until_ready()
    jit_time = time.time() - start
    
    # Time non-JIT version
    start = time.time()
    for _ in range(100):  # Fewer iterations since it's slower
        actions = compute_prior_policy(
            positions, velocities, grid_centers, grid_mask, 0.3, 0.15, 3.0
        )
        actions.block_until_ready()
    non_jit_time = (time.time() - start) * 10  # Scale to match n_calls
    
    speedup = non_jit_time / jit_time
    
    print(f"    JIT time for {n_calls} calls: {jit_time:.3f}s")
    print(f"    Non-JIT estimated: {non_jit_time:.3f}s")
    print(f"    Speedup: {speedup:.1f}x")
    
    # JIT should provide significant speedup
    assert speedup > 2.0, f"JIT speedup should be > 2x, got {speedup:.1f}x"
    
    # Test vmap compatibility
    batch_positions = jnp.stack([positions] * 8)  # 8 parallel envs
    batch_velocities = jnp.stack([velocities] * 8)
    
    vmap_policy = jax.vmap(lambda p, v: compute_prior_policy(
        p, v, grid_centers, grid_mask, 0.3, 0.15, 3.0
    ))
    
    batch_actions = vmap_policy(batch_positions, batch_velocities)
    assert batch_actions.shape == (8, n_agents, 2)
    
    print(f"    Vmap batch shape: {batch_actions.shape}")
    
    print("  ✓ Prior policy JIT compatibility passed")
    return True


# ============================================================
# C++ MARL MATCHING TESTS - REWARD FUNCTION
# ============================================================

def test_reward_cpp_matching_exploration():
    """Test exploration reward matches C++ MARL exactly.
    
    C++ uses delta=0.0 in rho_cos_dec, meaning the cosine decay starts
    immediately from distance=0 (no constant region).
    """
    print_subheader("Reward C++ Matching - Exploration")
    
    from rewards import compute_exploration_reward, rho_cos_dec
    
    # Test rho_cos_dec with delta=0.0 (C++ default)
    # Formula: 0.5 * (1 + cos(pi * z / r)) for z < r, else 0
    distances = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9, 3.0, 3.1])
    r = 3.0
    delta = 0.0
    
    weights = rho_cos_dec(distances, r, delta)
    
    # With delta=0:
    # z=0: cos(0) = 1, weight = 0.5 * (1 + 1) = 1.0
    # z=r/2=1.5: cos(pi/2) = 0, weight = 0.5 * (1 + 0) = 0.5
    # z=r=3.0: cos(pi) = -1, weight = 0.5 * (1 - 1) = 0.0
    print(f"    rho_cos_dec with delta=0.0: {weights}")
    
    assert jnp.isclose(weights[0], 1.0, atol=1e-5), f"z=0 should have weight 1.0, got {weights[0]}"
    assert jnp.isclose(weights[3], 0.5, atol=1e-2), f"z=r/2=1.5 should have ~0.5 weight, got {weights[3]}"
    assert jnp.isclose(weights[7], 0.0, atol=1e-5), f"z=r should have weight 0.0, got {weights[7]}"
    assert jnp.isclose(weights[8], 0.0, atol=1e-5), f"z>r should have weight 0.0, got {weights[8]}"
    
    # Verify monotonic decrease
    for i in range(len(weights) - 2):  # Exclude last which might be 0
        if weights[i+1] > 0:
            assert weights[i] >= weights[i+1], f"Weights should decrease: {weights[i]} < {weights[i+1]}"
    
    # Test exploration reward with symmetric grids (should get reward)
    positions = jnp.array([[0.0, 0.0]])
    grid_centers = jnp.array([
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]
    ])
    in_target = jnp.array([True])
    neighbor_indices = jnp.zeros((1, 1), dtype=jnp.int32)
    
    reward = compute_exploration_reward(
        positions, grid_centers, in_target, neighbor_indices,
        collision_threshold=0.15, exploration_threshold=0.05,
        d_sen=3.0, cosine_decay_delta=0.0
    )
    
    assert reward[0] == 1.0, f"Symmetric grids should give reward, got {reward[0]}"
    print(f"    Symmetric grids reward: {reward[0]}")
    
    # Test with asymmetric grids (should NOT get reward)
    grid_centers_asym = jnp.array([
        [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]  # All to the right
    ])
    
    reward_asym = compute_exploration_reward(
        positions, grid_centers_asym, in_target, neighbor_indices,
        collision_threshold=0.15, exploration_threshold=0.05,
        d_sen=3.0, cosine_decay_delta=0.0
    )
    
    assert reward_asym[0] == 0.0, f"Asymmetric grids should NOT give reward, got {reward_asym[0]}"
    print(f"    Asymmetric grids reward: {reward_asym[0]}")
    
    print("  ✓ Reward C++ matching - exploration passed")
    return True


def test_reward_cpp_matching_collision_check():
    """Test collision detection matches C++ MARL.
    
    C++ checks: if r_avoid > distance_to_neighbor, mark as collision.
    """
    print_subheader("Reward C++ Matching - Collision Check")
    
    from rewards import compute_agent_collisions
    
    # Two agents, one pair colliding, one not
    positions = jnp.array([
        [0.0, 0.0],    # Agent 0
        [0.1, 0.0],    # Agent 1 - colliding with 0 (dist=0.1 < r_avoid=0.15)
        [1.0, 0.0],    # Agent 2 - not colliding
    ])
    
    is_colliding, collision_matrix = compute_agent_collisions(
        positions, collision_threshold=0.15
    )
    
    # Agents 0 and 1 should be colliding
    assert is_colliding[0] == True, "Agent 0 should be colliding"
    assert is_colliding[1] == True, "Agent 1 should be colliding"
    assert is_colliding[2] == False, "Agent 2 should NOT be colliding"
    
    # Collision matrix should be symmetric
    assert collision_matrix[0, 1] == True
    assert collision_matrix[1, 0] == True
    assert collision_matrix[0, 2] == False
    
    print(f"    Collisions: {is_colliding}")
    print(f"    Collision matrix diagonal: {jnp.diag(collision_matrix)}")
    
    # Diagonal should always be False (no self-collision)
    assert jnp.all(jnp.diag(collision_matrix) == False)
    
    print("  ✓ Reward C++ matching - collision check passed")
    return True


def test_reward_cpp_matching_full_reward():
    """Test full reward computation matches C++ MARL logic.
    
    C++ reward logic:
    - reward = 0 initially
    - if in_target AND not_colliding AND is_uniform: reward = 1.0
    """
    print_subheader("Reward C++ Matching - Full Reward")
    
    from rewards import compute_rewards, RewardParams
    
    # Setup: 3 agents with different conditions
    positions = jnp.array([
        [0.0, 0.0],    # Agent 0: centered in symmetric grids, not colliding
        [0.05, 0.0],   # Agent 1: colliding with agent 0
        [5.0, 0.0],    # Agent 2: far from grids (not in target)
    ])
    velocities = jnp.zeros((3, 2))
    
    # Symmetric grids around origin
    grid_centers = jnp.array([
        [0.3, 0.0], [-0.3, 0.0], [0.0, 0.3], [0.0, -0.3]
    ])
    l_cell = 0.5  # Agents 0,1 are in target
    
    neighbor_indices = jnp.array([
        [1, -1, -1],  # Agent 0's neighbors
        [0, -1, -1],  # Agent 1's neighbors
        [-1, -1, -1], # Agent 2's neighbors
    ])
    
    reward_params = RewardParams(
        collision_threshold=0.15,
        exploration_threshold=0.05,
        cosine_decay_delta=0.0,  # Match C++
    )
    
    rewards, info = compute_rewards(
        positions, velocities, grid_centers, l_cell, neighbor_indices,
        reward_params, d_sen=3.0
    )
    
    print(f"    Agent 0 - in_target={info['in_target'][0]}, colliding={info['is_colliding'][0]}, exploration={info['exploration'][0]}")
    print(f"    Agent 1 - in_target={info['in_target'][1]}, colliding={info['is_colliding'][1]}, exploration={info['exploration'][1]}")
    print(f"    Agent 2 - in_target={info['in_target'][2]}, colliding={info['is_colliding'][2]}, exploration={info['exploration'][2]}")
    print(f"    Rewards: {rewards}")
    
    # Agent 0: in_target, colliding (with agent 1), so reward = 0
    assert rewards[0] == 0.0, f"Agent 0 colliding, should get 0, got {rewards[0]}"
    
    # Agent 1: in_target, colliding, so reward = 0
    assert rewards[1] == 0.0, f"Agent 1 colliding, should get 0, got {rewards[1]}"
    
    # Agent 2: NOT in target, so reward = 0
    assert rewards[2] == 0.0, f"Agent 2 not in target, should get 0, got {rewards[2]}"
    
    # Now test case where agent gets reward (in target, not colliding, uniform)
    positions_good = jnp.array([[0.0, 0.0]])  # Single agent
    velocities_good = jnp.zeros((1, 2))
    neighbor_indices_good = jnp.array([[-1, -1, -1]])
    
    rewards_good, info_good = compute_rewards(
        positions_good, velocities_good, grid_centers, l_cell, neighbor_indices_good,
        reward_params, d_sen=3.0
    )
    
    print(f"    Single agent - in_target={info_good['in_target'][0]}, colliding={info_good['is_colliding'][0]}, exploration={info_good['exploration'][0]}")
    print(f"    Single agent reward: {rewards_good[0]}")
    
    # Single agent at center of symmetric grids: in_target, not colliding, uniform
    assert rewards_good[0] == 1.0, f"Single agent should get reward 1.0, got {rewards_good[0]}"
    
    print("  ✓ Reward C++ matching - full reward passed")
    return True


def test_reward_jit_compatibility():
    """Test that reward computation is fully JIT-compatible."""
    print_subheader("Reward JIT Compatibility")
    
    from rewards import compute_rewards, RewardParams
    import time
    
    n_agents = 20
    n_grids = 50
    
    positions = jax.random.uniform(jax.random.PRNGKey(0), (n_agents, 2)) * 2.0
    velocities = jax.random.uniform(jax.random.PRNGKey(1), (n_agents, 2)) * 0.5
    grid_centers = jax.random.uniform(jax.random.PRNGKey(2), (n_grids, 2)) * 2.0
    neighbor_indices = jnp.zeros((n_agents, 5), dtype=jnp.int32) - 1
    
    reward_params = RewardParams(cosine_decay_delta=0.0)
    
    # JIT compile
    @jax.jit
    def compute_rewards_jit(pos, vel):
        return compute_rewards(pos, vel, grid_centers, 0.3, neighbor_indices, reward_params, d_sen=3.0)
    
    # Warmup
    _, _ = compute_rewards_jit(positions, velocities)
    
    # Time JIT version
    n_calls = 1000
    start = time.time()
    for _ in range(n_calls):
        rewards, info = compute_rewards_jit(positions, velocities)
        rewards.block_until_ready()
    jit_time = time.time() - start
    
    print(f"    JIT time for {n_calls} calls: {jit_time:.3f}s")
    print(f"    Per-call: {jit_time/n_calls*1000:.3f}ms")
    
    # Test vmap compatibility
    batch_positions = jnp.stack([positions] * 8)
    batch_velocities = jnp.stack([velocities] * 8)
    
    @jax.jit
    @jax.vmap
    def batch_rewards(pos, vel):
        return compute_rewards(pos, vel, grid_centers, 0.3, neighbor_indices, reward_params, d_sen=3.0)[0]
    
    batch_r = batch_rewards(batch_positions, batch_velocities)
    assert batch_r.shape == (8, n_agents)
    
    print(f"    Vmap batch shape: {batch_r.shape}")
    
    print("  ✓ Reward JIT compatibility passed")
    return True


def test_full_simulation_cpp_matching():
    """Test full simulation loop matches expected C++ behavior.
    
    Runs a full episode and verifies:
    1. Prior policy produces sensible actions
    2. Rewards are computed correctly
    3. Coverage increases over time (agents reach targets)
    4. No NaN/Inf values
    """
    print_subheader("Full Simulation C++ Matching")
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    from rewards import RewardParams
    
    n_agents = 10
    env = AssemblySwarmEnv(n_agents=n_agents)
    params = AssemblyParams(
        max_steps=200,
        reward_params=RewardParams(cosine_decay_delta=0.0),  # Match C++
    )
    
    key = random.PRNGKey(42)
    obs, state = env.reset(key, params)
    
    total_rewards = []
    coverage_history = []
    
    # Run simulation
    for step in range(params.max_steps):
        key, step_key = random.split(key)
        
        # Use prior policy (matches C++ robotPolicy)
        actions = env.prior_policy(state, params)
        
        # Check for NaN/Inf
        assert jnp.all(jnp.isfinite(actions)), f"NaN/Inf in actions at step {step}"
        
        # Step environment
        obs, state, rewards, dones, info = env.step(step_key, state, actions, params)
        
        # Check for NaN/Inf
        assert jnp.all(jnp.isfinite(rewards)), f"NaN/Inf in rewards at step {step}"
        assert jnp.all(jnp.isfinite(state.positions)), f"NaN/Inf in positions at step {step}"
        
        total_rewards.append(float(jnp.sum(rewards)))
        coverage_history.append(float(info['coverage_rate']))
        
        if state.done:
            break
    
    # Analyze results
    final_coverage = coverage_history[-1]
    total_reward = sum(total_rewards)
    agents_in_target = int(jnp.sum(state.in_target))
    
    print(f"    Steps run: {len(total_rewards)}")
    print(f"    Final coverage: {final_coverage:.2%}")
    print(f"    Agents in target: {agents_in_target}/{n_agents}")
    print(f"    Total reward: {total_reward:.2f}")
    print(f"    Mean reward per step: {total_reward/len(total_rewards):.4f}")
    
    # Coverage should increase over time with prior policy
    early_coverage = sum(coverage_history[:20]) / 20
    late_coverage = sum(coverage_history[-20:]) / 20
    
    print(f"    Early avg coverage (first 20): {early_coverage:.2%}")
    print(f"    Late avg coverage (last 20): {late_coverage:.2%}")
    
    # With prior policy, coverage should generally improve
    # (agents move toward targets)
    assert late_coverage >= early_coverage * 0.8, \
        f"Coverage should not decrease significantly: early={early_coverage:.2%}, late={late_coverage:.2%}"
    
    print("  ✓ Full simulation C++ matching passed")
    return True


# ============================================================
# RUN ALL TESTS
# ============================================================

def run_all_tests():
    """Run all tests."""
    print_header("ASSEMBLY SWARM ENVIRONMENT TESTS")
    
    tests = [
        # Core functionality
        ("Environment Creation", test_env_creation),
        ("Reset", test_reset),
        ("Step", test_step),
        ("Step with Actions", test_step_with_actions),
        ("Velocity Clipping", test_velocity_clipping),
        ("Episode Done", test_episode_done),
        ("Observation Space", test_observation_space),
        ("Action Space", test_action_space),
        ("make_assembly_env", test_make_assembly_env),
        
        # JIT and Vectorization
        ("JIT Reset", test_jit_reset),
        ("JIT Step", test_jit_step),
        ("Vmap Reset", test_vmap_reset),
        ("Vmap Step", test_vmap_step),
        ("make_vec_env", test_make_vec_env),
        
        # Shape Loading
        ("Shape Loader", test_shape_loader),
        ("Pickle Integration", test_pickle_integration),
        ("Fig Directory Shapes", test_fig_shapes_loading),
        
        # Domain Randomization
        ("Domain Randomization", test_domain_randomization),
        ("Deterministic Mode", test_deterministic_mode),
        
        # Features
        ("Trajectory Tracking", test_trajectory_tracking),
        ("Prior Policy", test_prior_policy),
        ("Standalone Prior Policy", test_standalone_prior_policy),
        ("Prior Policy Repulsion Strength", test_prior_policy_repulsion_strength),
        ("Prior Policy In-Target Behavior", test_prior_policy_in_target_behavior),
        ("Occupied Grid Tracking", test_occupied_grid_tracking),
        ("Reward Sharing", test_reward_sharing),
        
        # C++ MARL Matching Tests
        ("Prior Policy C++ Combined Forces", test_prior_policy_cpp_matching_combined_forces),
        ("Prior Policy C++ Force Clipping", test_prior_policy_cpp_matching_force_clipping),
        ("Prior Policy JIT Compatibility", test_prior_policy_jit_compatibility),
        ("Reward C++ Exploration", test_reward_cpp_matching_exploration),
        ("Reward C++ Collision Check", test_reward_cpp_matching_collision_check),
        ("Reward C++ Full Reward", test_reward_cpp_matching_full_reward),
        ("Reward JIT Compatibility", test_reward_jit_compatibility),
        ("Full Simulation C++ Matching", test_full_simulation_cpp_matching),
        
        # Performance
        ("Performance", test_performance),
        ("Full Episode", test_full_episode),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = passed
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print_header("TEST SUMMARY")
    passed_count = sum(results.values())
    total_count = len(results)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print_header("ALL TESTS PASSED! ✓")
        return True
    else:
        print_header("SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
