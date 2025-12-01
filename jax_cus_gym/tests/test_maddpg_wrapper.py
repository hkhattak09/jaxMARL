"""Tests for the MADDPG wrapper.

Run with: python tests/test_maddpg_wrapper.py

These tests use actual target shapes from the fig directory (results.pkl)
which contains processed shape data from PNG images.
"""

import jax
import jax.numpy as jnp
from jax import random
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_fig_shapes_path() -> str:
    """Get path to the actual shapes file in the fig directory.
    
    Returns:
        Path to fig/results.pkl containing processed target shapes.
        Returns None if the file doesn't exist.
    """
    workspace_root = Path(__file__).parent.parent.parent  # jax_cus_gym -> MARL_jax
    fig_path = workspace_root / "fig" / "results.pkl"
    if fig_path.exists():
        return str(fig_path)
    return None


# Get the shapes path once at module load
FIG_SHAPES_PATH = get_fig_shapes_path()


from maddpg_wrapper import (
    MADDPGWrapper,
    VectorizedMADDPGWrapper,
    MADDPGState,
    Transition,
    create_maddpg_env,
    create_vec_maddpg_env,
    rollout_episode,
    stack_transitions,
)


def test_wrapper_creation():
    """Test wrapper creation with actual shapes from fig directory."""
    print("Testing wrapper creation...")
    
    # Use actual shapes if available
    kwargs = {}
    if FIG_SHAPES_PATH:
        kwargs['shape_file'] = FIG_SHAPES_PATH
        print(f"  Using actual shapes from: {FIG_SHAPES_PATH}")
    
    wrapper = MADDPGWrapper(n_agents=10, **kwargs)
    
    assert wrapper.n_agents == 10
    assert wrapper.obs_dim > 0
    assert wrapper.action_dim == 2
    assert wrapper.global_state_dim > 0
    
    # With actual shapes, we expect 7 shapes with large grids
    if FIG_SHAPES_PATH:
        assert wrapper.env.shape_library.n_shapes == 7, f"Expected 7 shapes, got {wrapper.env.shape_library.n_shapes}"
        assert wrapper.env.shape_library.max_n_grid >= 400, f"Expected large grids, got {wrapper.env.shape_library.max_n_grid}"
    
    print(f"  n_shapes: {wrapper.env.shape_library.n_shapes}")
    print(f"  max_n_grid: {wrapper.env.shape_library.max_n_grid}")
    print(f"  obs_dim: {wrapper.obs_dim}")
    print(f"  action_dim: {wrapper.action_dim}")
    print(f"  global_state_dim: {wrapper.global_state_dim}")
    
    print("  ✓ Wrapper creation tests passed")


def test_reset():
    """Test wrapper reset with actual shapes."""
    print("Testing reset...")
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    wrapper = MADDPGWrapper(n_agents=8, **kwargs)
    key = random.PRNGKey(42)
    
    obs, state = wrapper.reset(key)
    
    assert obs.shape == (8, wrapper.obs_dim)
    assert isinstance(state, MADDPGState)
    assert state.episode_length == 0
    assert jnp.allclose(state.episode_returns, 0.0)
    
    print("  ✓ Reset tests passed")


def test_step():
    """Test wrapper step with actual shapes."""
    print("Testing step...")
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    wrapper = MADDPGWrapper(n_agents=6, **kwargs)
    key = random.PRNGKey(0)
    
    obs, state = wrapper.reset(key)
    
    key, step_key, action_key = random.split(key, 3)
    actions = random.uniform(action_key, (6, 2), minval=-1, maxval=1)
    
    next_obs, next_state, rewards, dones, info = wrapper.step(step_key, state, actions)
    
    assert next_obs.shape == obs.shape
    assert rewards.shape == (6,)
    assert dones.shape == (6,)
    assert next_state.episode_length == 1
    assert "episode_returns" in info
    assert "episode_length" in info
    
    print("  ✓ Step tests passed")


def test_global_state():
    """Test global state computation with actual shapes."""
    print("Testing global state...")
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    wrapper = MADDPGWrapper(n_agents=5, **kwargs)
    key = random.PRNGKey(0)
    
    obs, state = wrapper.reset(key)
    global_state = wrapper.get_global_state(state)
    
    expected_dim = wrapper.global_state_dim
    assert global_state.shape == (expected_dim,), f"Expected {expected_dim}, got {global_state.shape}"
    
    # Check components
    n_agents = 5
    n_grid = wrapper.env.shape_library.max_n_grid  # Use shape library grid size
    agent_dim = n_agents * 4  # positions + velocities
    grid_dim = n_grid * 2
    time_dim = 1
    
    assert expected_dim == agent_dim + grid_dim + time_dim
    
    print(f"  Global state dim: {expected_dim}")
    print("  ✓ Global state tests passed")


def test_transition_collection():
    """Test transition collection with actual shapes."""
    print("Testing transition collection...")
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    wrapper = MADDPGWrapper(n_agents=4, **kwargs)
    key = random.PRNGKey(0)
    
    obs, state = wrapper.reset(key)
    
    key, step_key, action_key = random.split(key, 3)
    actions = random.uniform(action_key, (4, 2), minval=-1, maxval=1)
    next_obs, next_state, rewards, dones, info = wrapper.step(step_key, state, actions)
    
    transition = wrapper.collect_transition(
        obs, actions, rewards, next_obs, dones, state, next_state
    )
    
    assert isinstance(transition, Transition)
    assert transition.obs.shape == (4, wrapper.obs_dim)
    assert transition.actions.shape == (4, 2)
    assert transition.rewards.shape == (4,)
    assert transition.next_obs.shape == (4, wrapper.obs_dim)
    assert transition.dones.shape == (4,)
    assert transition.global_state is not None
    assert transition.next_global_state is not None
    
    print("  ✓ Transition collection tests passed")


def test_vectorized_wrapper():
    """Test vectorized wrapper with actual shapes."""
    print("Testing vectorized wrapper...")
    
    n_envs = 4
    n_agents = 5
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    vec_wrapper = VectorizedMADDPGWrapper(n_envs=n_envs, n_agents=n_agents, **kwargs)
    
    assert vec_wrapper.n_envs == n_envs
    assert vec_wrapper.n_agents == n_agents
    
    keys = random.split(random.PRNGKey(0), n_envs)
    obs, states = vec_wrapper.reset(keys)
    
    assert obs.shape == (n_envs, n_agents, vec_wrapper.obs_dim)
    
    print("  ✓ Vectorized wrapper tests passed")


def test_vectorized_step():
    """Test vectorized step with actual shapes."""
    print("Testing vectorized step...")
    
    n_envs = 8
    n_agents = 6
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    vec_wrapper = VectorizedMADDPGWrapper(n_envs=n_envs, n_agents=n_agents, **kwargs)
    
    keys = random.split(random.PRNGKey(0), n_envs)
    obs, states = vec_wrapper.reset(keys)
    
    actions = random.uniform(
        random.PRNGKey(1), 
        (n_envs, n_agents, 2), 
        minval=-1, maxval=1
    )
    step_keys = random.split(random.PRNGKey(2), n_envs)
    
    next_obs, next_states, rewards, dones, info = vec_wrapper.step(
        step_keys, states, actions
    )
    
    assert next_obs.shape == (n_envs, n_agents, vec_wrapper.obs_dim)
    assert rewards.shape == (n_envs, n_agents)
    assert dones.shape == (n_envs, n_agents)
    
    print("  ✓ Vectorized step tests passed")


def test_vectorized_global_states():
    """Test global states for all environments with actual shapes."""
    print("Testing vectorized global states...")
    
    n_envs = 4
    n_agents = 5
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    vec_wrapper = VectorizedMADDPGWrapper(n_envs=n_envs, n_agents=n_agents, **kwargs)
    
    keys = random.split(random.PRNGKey(0), n_envs)
    obs, states = vec_wrapper.reset(keys)
    
    global_states = vec_wrapper.get_global_states(states)
    
    assert global_states.shape == (n_envs, vec_wrapper.global_state_dim)
    
    print("  ✓ Vectorized global states tests passed")


def test_convenience_functions():
    """Test convenience functions with actual shapes."""
    print("Testing convenience functions...")
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    
    # Test create_maddpg_env
    wrapper = create_maddpg_env(n_agents=7, arena_size=6.0, **kwargs)
    assert wrapper.n_agents == 7
    
    # Test create_vec_maddpg_env
    vec_wrapper = create_vec_maddpg_env(n_envs=3, n_agents=8, **kwargs)
    assert vec_wrapper.n_envs == 3
    assert vec_wrapper.n_agents == 8
    
    print("  ✓ Convenience functions tests passed")


def test_rollout_episode():
    """Test episode rollout with actual shapes."""
    print("Testing episode rollout...")
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    wrapper = MADDPGWrapper(n_agents=4, max_steps=20, **kwargs)
    
    # Simple random policy
    def random_policy(key, obs):
        return random.uniform(key, (4, 2), minval=-1, maxval=1)
    
    key = random.PRNGKey(42)
    transitions, episode_info = rollout_episode(wrapper, key, random_policy)
    
    assert len(transitions) > 0
    assert len(transitions) <= 20
    assert "episode_return" in episode_info
    assert "episode_length" in episode_info
    
    print(f"  Episode length: {episode_info['episode_length']}")
    print(f"  Episode return: {episode_info['episode_return']:.2f}")
    
    print("  ✓ Episode rollout tests passed")


def test_stack_transitions():
    """Test transition stacking with actual shapes."""
    print("Testing transition stacking...")
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    wrapper = MADDPGWrapper(n_agents=3, **kwargs)
    
    def random_policy(key, obs):
        return random.uniform(key, (3, 2), minval=-1, maxval=1)
    
    key = random.PRNGKey(0)
    transitions, _ = rollout_episode(wrapper, key, random_policy, max_steps=10)
    
    batched = stack_transitions(transitions)
    
    assert batched.obs.shape[0] == len(transitions)
    assert batched.actions.shape[0] == len(transitions)
    assert batched.rewards.shape[0] == len(transitions)
    
    print(f"  Batched obs shape: {batched.obs.shape}")
    
    print("  ✓ Transition stacking tests passed")


def test_jit_compatibility():
    """Test JIT compatibility with actual shapes."""
    print("Testing JIT compatibility...")
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    wrapper = MADDPGWrapper(n_agents=5, **kwargs)
    
    @jax.jit
    def jit_reset(key):
        return wrapper.reset(key)
    
    @jax.jit
    def jit_step(key, state, actions):
        return wrapper.step(key, state, actions)
    
    key = random.PRNGKey(0)
    obs, state = jit_reset(key)
    
    actions = random.uniform(key, (5, 2), minval=-1, maxval=1)
    key, step_key = random.split(key)
    next_obs, next_state, rewards, dones, info = jit_step(step_key, state, actions)
    
    # Run again to ensure compiled version works
    key, step_key = random.split(key)
    next_obs, next_state, rewards, dones, info = jit_step(step_key, next_state, actions)
    
    print("  ✓ JIT compatibility tests passed")


def test_custom_params():
    """Test wrapper with custom parameters and actual shapes."""
    print("Testing custom parameters...")
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    wrapper = MADDPGWrapper(
        n_agents=12,
        arena_size=8.0,
        max_velocity=2.0,
        max_acceleration=3.0,
        **kwargs
    )
    
    assert wrapper.n_agents == 12
    assert wrapper.params.arena_size == 8.0
    assert wrapper.params.max_velocity == 2.0
    
    key = random.PRNGKey(0)
    obs, state = wrapper.reset(key)
    
    # Grid size depends on shape library max_n_grid
    expected_grid = wrapper.env.shape_library.max_n_grid
    assert state.env_state.grid_centers.shape[0] == expected_grid
    
    print("  ✓ Custom parameters tests passed")


def test_agent_obs():
    """Test per-agent observation extraction with actual shapes."""
    print("Testing agent observation extraction...")
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    wrapper = MADDPGWrapper(n_agents=4, **kwargs)
    key = random.PRNGKey(0)
    obs, state = wrapper.reset(key)
    
    # Get individual agent observations
    for i in range(4):
        agent_obs = wrapper.get_agent_obs(obs, i)
        assert agent_obs.shape == (wrapper.obs_dim,)
        assert jnp.allclose(agent_obs, obs[i])
    
    print("  ✓ Agent observation extraction tests passed")


def test_episode_tracking():
    """Test episode return and length tracking with actual shapes."""
    print("Testing episode tracking...")
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    wrapper = MADDPGWrapper(n_agents=3, max_steps=10, **kwargs)
    key = random.PRNGKey(0)
    
    obs, state = wrapper.reset(key)
    
    total_steps = 0
    for i in range(15):
        key, step_key, action_key = random.split(key, 3)
        actions = random.uniform(action_key, (3, 2), minval=-1, maxval=1)
        obs, state, rewards, dones, info = wrapper.step(step_key, state, actions)
        total_steps += 1
        
        if state.env_state.done:
            break
    
    assert state.episode_length == total_steps
    assert state.episode_length <= 10
    
    print(f"  Episode length: {state.episode_length}")
    print("  ✓ Episode tracking tests passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running MADDPG wrapper tests")
    print("="*60 + "\n")
    
    if FIG_SHAPES_PATH:
        print(f"Using actual shapes from: {FIG_SHAPES_PATH}")
    else:
        print("Warning: fig/results.pkl not found, using procedural shapes")
    print()
    
    test_wrapper_creation()
    test_reset()
    test_step()
    test_global_state()
    test_transition_collection()
    test_vectorized_wrapper()
    test_vectorized_step()
    test_vectorized_global_states()
    test_convenience_functions()
    test_rollout_episode()
    test_stack_transitions()
    test_jit_compatibility()
    test_custom_params()
    test_agent_obs()
    test_episode_tracking()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    if FIG_SHAPES_PATH:
        print("Tests used actual target shapes from fig directory.")
    print("="*60 + "\n")
