#!/usr/bin/env python3
"""Comprehensive test script for JAX MARL vectorization.

Tests all core components to ensure they work correctly after vectorization changes.
Skips LLM-related files.

Run with: python -m pytest tests/test_vectorization.py -v
Or directly: python tests/test_vectorization.py
"""

import sys
from pathlib import Path

# Add paths for imports
jax_marl_dir = Path(__file__).parent.parent
sys.path.insert(0, str(jax_marl_dir))
sys.path.insert(0, str(jax_marl_dir / "algo"))  # For algo internal imports
sys.path.insert(0, str(jax_marl_dir.parent / "jax_cus_gym"))

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Tuple


# ============================================================================
# Test Configuration
# ============================================================================

TEST_N_AGENTS = 3
TEST_N_ENVS = 2
TEST_OBS_DIM = 20
TEST_ACTION_DIM = 2
TEST_BATCH_SIZE = 16
TEST_BUFFER_SIZE = 100
TEST_HIDDEN_DIM = 32
TEST_MAX_STEPS = 10


def print_test_header(name: str):
    """Print a formatted test header."""
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")


def print_success(msg: str):
    """Print success message."""
    print(f"  ✓ {msg}")


def print_fail(msg: str):
    """Print failure message."""
    print(f"  ✗ {msg}")


# ============================================================================
# 1. Test Networks (algo/networks.py)
# ============================================================================

def test_networks():
    """Test Actor and Critic networks."""
    print_test_header("Networks (algo/networks.py)")
    
    from algo.networks import Actor, Critic
    
    key = random.PRNGKey(0)
    
    # Test Actor
    actor = Actor(action_dim=TEST_ACTION_DIM, hidden_dims=(TEST_HIDDEN_DIM, TEST_HIDDEN_DIM))
    actor_params = actor.init(key, jnp.zeros((1, TEST_OBS_DIM)))
    
    # Single observation
    obs = jnp.ones((TEST_OBS_DIM,))
    action = actor.apply(actor_params, obs[None, :])
    assert action.shape == (1, TEST_ACTION_DIM), f"Expected (1, {TEST_ACTION_DIM}), got {action.shape}"
    print_success(f"Actor single obs: input {obs.shape} -> output {action.shape}")
    
    # Batch observation
    obs_batch = jnp.ones((TEST_BATCH_SIZE, TEST_OBS_DIM))
    actions = actor.apply(actor_params, obs_batch)
    assert actions.shape == (TEST_BATCH_SIZE, TEST_ACTION_DIM)
    print_success(f"Actor batch: input {obs_batch.shape} -> output {actions.shape}")
    
    # Test Critic (takes obs AND action as separate inputs)
    obs_dim_critic = TEST_OBS_DIM * TEST_N_AGENTS  # Global obs
    action_dim_critic = TEST_ACTION_DIM * TEST_N_AGENTS  # All actions
    
    critic = Critic(hidden_dims=(TEST_HIDDEN_DIM, TEST_HIDDEN_DIM))
    
    # Initialize with both obs and action
    dummy_obs = jnp.zeros((1, obs_dim_critic))
    dummy_action = jnp.zeros((1, action_dim_critic))
    critic_params = critic.init(key, dummy_obs, dummy_action)
    
    # Forward pass
    critic_obs = jnp.ones((TEST_BATCH_SIZE, obs_dim_critic))
    critic_action = jnp.ones((TEST_BATCH_SIZE, action_dim_critic))
    q_values = critic.apply(critic_params, critic_obs, critic_action)
    assert q_values.shape == (TEST_BATCH_SIZE, 1)
    print_success(f"Critic: obs {critic_obs.shape} + action {critic_action.shape} -> Q {q_values.shape}")
    
    print_success("All network tests passed!")
    return True


# ============================================================================
# 2. Test Noise (algo/noise.py)
# ============================================================================

def test_noise():
    """Test noise generation functions."""
    print_test_header("Noise (algo/noise.py)")
    
    from algo.noise import gaussian_noise, add_gaussian_noise, ou_noise_init, ou_noise_step, OUNoiseParams
    
    key = random.PRNGKey(42)
    
    # Test Gaussian noise
    noise = gaussian_noise(key, shape=(TEST_ACTION_DIM,), scale=0.1)
    assert noise.shape == (TEST_ACTION_DIM,)
    print_success(f"Gaussian noise shape: {noise.shape}")
    
    # Test add_gaussian_noise (returns tuple: noisy_action, noise)
    action = jnp.zeros((TEST_ACTION_DIM,))
    noisy_action, noise_added = add_gaussian_noise(key, action, scale=0.1)
    assert noisy_action.shape == action.shape
    assert not jnp.allclose(noisy_action, action)  # Should be different due to noise
    print_success(f"Add Gaussian noise: {action.shape} -> {noisy_action.shape}")
    
    # Test OU noise
    ou_state = ou_noise_init(TEST_ACTION_DIM)
    assert ou_state.state.shape == (TEST_ACTION_DIM,)
    print_success(f"OU noise init: state shape {ou_state.state.shape}")
    
    # Create OU params for the step
    ou_params = OUNoiseParams(action_dim=TEST_ACTION_DIM)
    new_ou_state, ou_noise = ou_noise_step(key, ou_state, ou_params)
    assert ou_noise.shape == (TEST_ACTION_DIM,)
    print_success(f"OU noise step: noise shape {ou_noise.shape}")
    
    print_success("All noise tests passed!")
    return True


# ============================================================================
# 3. Test Buffers (algo/buffers.py)
# ============================================================================

def test_buffers():
    """Test replay buffer including batch operations."""
    print_test_header("Buffers (algo/buffers.py)")
    
    from algo.buffers import ReplayBuffer, Transition
    
    buffer = ReplayBuffer(
        capacity=TEST_BUFFER_SIZE,
        n_agents=TEST_N_AGENTS,
        obs_dim=TEST_OBS_DIM,
        action_dim=TEST_ACTION_DIM,
        store_action_priors=True,
    )
    
    state = buffer.init()
    assert state.size == 0
    print_success(f"Buffer initialized with capacity {TEST_BUFFER_SIZE}")
    
    # Test single add
    transition = Transition(
        obs=jnp.ones((TEST_N_AGENTS, TEST_OBS_DIM)),
        actions=jnp.ones((TEST_N_AGENTS, TEST_ACTION_DIM)),
        rewards=jnp.ones((TEST_N_AGENTS,)),
        next_obs=jnp.ones((TEST_N_AGENTS, TEST_OBS_DIM)),
        dones=jnp.zeros((TEST_N_AGENTS,)),
        action_priors=jnp.ones((TEST_N_AGENTS, TEST_ACTION_DIM)),
    )
    
    state = buffer.add(state, transition)
    assert state.size == 1
    print_success(f"Single add: buffer size = {state.size}")
    
    # Test batch add (KEY VECTORIZATION TEST)
    batch_transitions = Transition(
        obs=jnp.ones((TEST_N_ENVS, TEST_N_AGENTS, TEST_OBS_DIM)),
        actions=jnp.ones((TEST_N_ENVS, TEST_N_AGENTS, TEST_ACTION_DIM)),
        rewards=jnp.ones((TEST_N_ENVS, TEST_N_AGENTS)),
        next_obs=jnp.ones((TEST_N_ENVS, TEST_N_AGENTS, TEST_OBS_DIM)),
        dones=jnp.zeros((TEST_N_ENVS, TEST_N_AGENTS)),
        action_priors=jnp.ones((TEST_N_ENVS, TEST_N_AGENTS, TEST_ACTION_DIM)),
    )
    
    state = buffer.add_batch(state, batch_transitions)
    assert state.size == 1 + TEST_N_ENVS
    print_success(f"Batch add ({TEST_N_ENVS} transitions): buffer size = {state.size}")
    
    # Fill buffer more for sampling test
    for _ in range(20):
        state = buffer.add_batch(state, batch_transitions)
    
    # Test sampling
    key = random.PRNGKey(0)
    can_sample = buffer.can_sample(state, TEST_BATCH_SIZE)
    assert can_sample, "Buffer should have enough samples"
    print_success(f"Can sample {TEST_BATCH_SIZE} from buffer with size {state.size}")
    
    batch = buffer.sample(state, key, TEST_BATCH_SIZE)
    assert batch.obs.shape == (TEST_BATCH_SIZE, TEST_N_AGENTS, TEST_OBS_DIM)
    assert batch.actions.shape == (TEST_BATCH_SIZE, TEST_N_AGENTS, TEST_ACTION_DIM)
    assert batch.rewards.shape == (TEST_BATCH_SIZE, TEST_N_AGENTS)
    print_success(f"Sample batch shapes correct: obs={batch.obs.shape}, actions={batch.actions.shape}")
    
    print_success("All buffer tests passed!")
    return True


# ============================================================================
# 4. Test Agents (algo/agents.py)
# ============================================================================

def test_agents():
    """Test DDPG agent functions."""
    print_test_header("Agents (algo/agents.py)")
    
    from algo.agents import (
        create_agent, AgentConfig, select_action, 
        select_action_with_noise, update_targets
    )
    
    key = random.PRNGKey(0)
    
    critic_input_dim = TEST_OBS_DIM * TEST_N_AGENTS + TEST_ACTION_DIM * TEST_N_AGENTS
    
    config = AgentConfig(
        obs_dim=TEST_OBS_DIM,
        action_dim=TEST_ACTION_DIM,
        critic_input_dim=critic_input_dim,
        hidden_dims=(TEST_HIDDEN_DIM, TEST_HIDDEN_DIM),
    )
    
    agent_state, actor, critic = create_agent(key, config)
    print_success(f"Agent created with obs_dim={TEST_OBS_DIM}, action_dim={TEST_ACTION_DIM}")
    
    # Test action selection
    obs = jnp.ones((TEST_OBS_DIM,))
    action = select_action(actor, agent_state.actor_params, obs)
    assert action.shape == (TEST_ACTION_DIM,)
    print_success(f"select_action: obs {obs.shape} -> action {action.shape}")
    
    # Test batched action selection
    obs_batch = jnp.ones((TEST_BATCH_SIZE, TEST_OBS_DIM))
    actions = select_action(actor, agent_state.actor_params, obs_batch)
    assert actions.shape == (TEST_BATCH_SIZE, TEST_ACTION_DIM)
    print_success(f"select_action batched: obs {obs_batch.shape} -> actions {actions.shape}")
    
    # Test action with noise
    action_noisy, log_prob, _ = select_action_with_noise(
        key, actor, agent_state.actor_params, obs,
        noise_scale=0.1, noise_type='gaussian'
    )
    assert action_noisy.shape == (TEST_ACTION_DIM,)
    print_success(f"select_action_with_noise: action {action_noisy.shape}")
    
    # Test target update
    new_agent_state = update_targets(agent_state, tau=0.01)
    assert new_agent_state is not None
    print_success("Target network update works")
    
    print_success("All agent tests passed!")
    return True


# ============================================================================
# 5. Test MADDPG (algo/maddpg.py)
# ============================================================================

def test_maddpg():
    """Test MADDPG including new batched methods."""
    print_test_header("MADDPG (algo/maddpg.py)")
    
    from algo.maddpg import MADDPG, MADDPGConfig
    
    config = MADDPGConfig(
        n_agents=TEST_N_AGENTS,
        obs_dims=tuple([TEST_OBS_DIM] * TEST_N_AGENTS),
        action_dims=tuple([TEST_ACTION_DIM] * TEST_N_AGENTS),
        hidden_dims=(TEST_HIDDEN_DIM, TEST_HIDDEN_DIM),
        buffer_size=TEST_BUFFER_SIZE,
        batch_size=TEST_BATCH_SIZE,
        warmup_steps=10,
    )
    
    maddpg = MADDPG(config)
    key = random.PRNGKey(0)
    state = maddpg.init(key)
    print_success(f"MADDPG initialized with {TEST_N_AGENTS} agents")
    
    # Test original select_actions (list-based)
    obs_list = [jnp.ones((TEST_OBS_DIM,)) for _ in range(TEST_N_AGENTS)]
    actions, log_probs, state = maddpg.select_actions(key, state, obs_list, explore=True)
    assert len(actions) == TEST_N_AGENTS
    assert actions[0].shape == (TEST_ACTION_DIM,)
    print_success(f"select_actions: {TEST_N_AGENTS} agents, action shape {actions[0].shape}")
    
    # Test NEW batched select_actions (KEY VECTORIZATION TEST)
    obs_batch = jnp.ones((TEST_N_ENVS, TEST_N_AGENTS, TEST_OBS_DIM))
    actions_batch, state = maddpg.select_actions_batched(key, state, obs_batch, explore=True)
    assert actions_batch.shape == (TEST_N_ENVS, TEST_N_AGENTS, TEST_ACTION_DIM)
    print_success(f"select_actions_batched: input {obs_batch.shape} -> output {actions_batch.shape}")
    
    # Test without exploration
    actions_batch_no_noise, _ = maddpg.select_actions_batched(key, state, obs_batch, explore=False)
    assert actions_batch_no_noise.shape == (TEST_N_ENVS, TEST_N_AGENTS, TEST_ACTION_DIM)
    print_success(f"select_actions_batched (no explore): output {actions_batch_no_noise.shape}")
    
    # Test original store_transition
    rewards_list = [jnp.array([0.5]) for _ in range(TEST_N_AGENTS)]
    dones_list = [jnp.array([False]) for _ in range(TEST_N_AGENTS)]
    state = maddpg.store_transition(state, obs_list, actions, rewards_list, obs_list, dones_list)
    print_success(f"store_transition: buffer size = {state.buffer_state.size}")
    
    # Test NEW batched store_transitions (KEY VECTORIZATION TEST)
    rewards_batch = jnp.ones((TEST_N_ENVS, TEST_N_AGENTS))
    dones_batch = jnp.zeros((TEST_N_ENVS, TEST_N_AGENTS))
    next_obs_batch = jnp.ones((TEST_N_ENVS, TEST_N_AGENTS, TEST_OBS_DIM))
    
    state = maddpg.store_transitions_batched(
        state,
        obs_batch,
        actions_batch,
        rewards_batch,
        next_obs_batch,
        dones_batch,
    )
    expected_size = 1 + TEST_N_ENVS
    assert state.buffer_state.size == expected_size, f"Expected {expected_size}, got {state.buffer_state.size}"
    print_success(f"store_transitions_batched: buffer size = {state.buffer_state.size}")
    
    # Fill buffer for update test
    for _ in range(20):
        state = maddpg.store_transitions_batched(
            state, obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch
        )
    
    # Test update
    key, update_key = random.split(key)
    state, info = maddpg.update(update_key, state)
    assert info.get('can_update', False), "Update should have succeeded"
    print_success(f"MADDPG update: info keys = {list(info.keys())[:5]}...")
    
    print_success("All MADDPG tests passed!")
    return True


# ============================================================================
# 6. Test Environment (jax_cus_gym/assembly_env.py)
# ============================================================================

def test_environment():
    """Test Assembly environment."""
    print_test_header("Environment (jax_cus_gym/assembly_env.py)")
    
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    
    env = AssemblySwarmEnv(n_agents=TEST_N_AGENTS)
    params = AssemblyParams()
    
    key = random.PRNGKey(0)
    obs, state = env.reset(key, params)
    
    assert obs.shape[0] == TEST_N_AGENTS
    print_success(f"Environment reset: obs shape = {obs.shape}")
    print_success(f"State positions shape: {state.positions.shape}")
    
    # Test step
    actions = jnp.zeros((TEST_N_AGENTS, 2))
    key, step_key = random.split(key)
    next_obs, next_state, rewards, dones, info = env.step(step_key, state, actions, params)
    
    assert next_obs.shape == obs.shape
    assert rewards.shape == (TEST_N_AGENTS,)
    assert dones.shape == (TEST_N_AGENTS,)
    print_success(f"Environment step: rewards shape = {rewards.shape}")
    
    print_success("All environment tests passed!")
    return True


# ============================================================================
# 7. Test Vectorized Environment (make_vec_env)
# ============================================================================

def test_vectorized_environment():
    """Test vectorized environment operations."""
    print_test_header("Vectorized Environment")
    
    from assembly_env import make_vec_env, AssemblyParams
    
    env, _, _, _ = make_vec_env(n_envs=TEST_N_ENVS, n_agents=TEST_N_AGENTS)
    params = AssemblyParams()
    
    # Create custom vectorized functions with params
    @jax.jit
    def test_vec_reset(keys):
        return jax.vmap(lambda k: env.reset(k, params))(keys)
    
    @jax.jit
    def test_vec_step(keys, states, actions):
        return jax.vmap(lambda k, s, a: env.step(k, s, a, params))(keys, states, actions)
    
    key = random.PRNGKey(0)
    keys = random.split(key, TEST_N_ENVS)
    
    # Test vectorized reset
    obs_batch, states = test_vec_reset(keys)
    assert obs_batch.shape[0] == TEST_N_ENVS
    assert obs_batch.shape[1] == TEST_N_AGENTS
    print_success(f"Vectorized reset: obs_batch shape = {obs_batch.shape}")
    
    # Test vectorized step
    actions_batch = jnp.zeros((TEST_N_ENVS, TEST_N_AGENTS, 2))
    step_keys = random.split(key, TEST_N_ENVS)
    
    next_obs, next_states, rewards, dones, info = test_vec_step(step_keys, states, actions_batch)
    
    assert next_obs.shape == obs_batch.shape
    assert rewards.shape == (TEST_N_ENVS, TEST_N_AGENTS)
    assert dones.shape == (TEST_N_ENVS, TEST_N_AGENTS)
    print_success(f"Vectorized step: rewards shape = {rewards.shape}")
    
    print_success("All vectorized environment tests passed!")
    return True


# ============================================================================
# 8. Test Prior Policy (jax_cus_gym/assembly_env.py)
# ============================================================================

def test_prior_policy():
    """Test prior policy computation including vectorized version."""
    print_test_header("Prior Policy")
    
    from assembly_env import compute_prior_policy, AssemblySwarmEnv, AssemblyParams
    
    env = AssemblySwarmEnv(n_agents=TEST_N_AGENTS)
    params = AssemblyParams()
    
    key = random.PRNGKey(0)
    _, state = env.reset(key, params)
    
    # Test single environment prior computation
    prior_actions = compute_prior_policy(
        state.positions,
        state.velocities,
        state.grid_centers,
        state.grid_mask,
        state.l_cell,
        params.reward_params.collision_threshold,
        params.d_sen,
    )
    
    assert prior_actions.shape == (TEST_N_AGENTS, 2)
    print_success(f"Prior policy single env: shape = {prior_actions.shape}")
    
    # Test vectorized prior computation (KEY VECTORIZATION TEST)
    from assembly_env import make_vec_env
    
    vec_env, _, _, _ = make_vec_env(n_envs=TEST_N_ENVS, n_agents=TEST_N_AGENTS)
    
    @jax.jit
    def vec_reset(keys):
        return jax.vmap(lambda k: vec_env.reset(k, params))(keys)
    
    keys = random.split(key, TEST_N_ENVS)
    _, states = vec_reset(keys)
    
    # vmap prior computation over environments
    def compute_priors_single(positions, velocities, grid_centers, grid_mask, l_cell):
        return compute_prior_policy(
            positions, velocities, grid_centers, grid_mask, l_cell,
            params.reward_params.collision_threshold,
            params.d_sen,
        )
    
    prior_actions_batch = jax.vmap(compute_priors_single)(
        states.positions,
        states.velocities,
        states.grid_centers,
        states.grid_mask,
        states.l_cell,
    )
    
    assert prior_actions_batch.shape == (TEST_N_ENVS, TEST_N_AGENTS, 2)
    print_success(f"Prior policy vectorized: shape = {prior_actions_batch.shape}")
    
    print_success("All prior policy tests passed!")
    return True


# ============================================================================
# 9. Test Observations (jax_cus_gym/observations.py)
# ============================================================================

def test_observations():
    """Test observation computation."""
    print_test_header("Observations (jax_cus_gym/observations.py)")
    
    from observations import compute_observation_dim, ObservationParams
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    
    # Use default ObservationParams
    obs_params = ObservationParams()
    
    obs_dim = compute_observation_dim(obs_params)
    print_success(f"Observation dimension: {obs_dim}")
    
    env = AssemblySwarmEnv(n_agents=TEST_N_AGENTS)
    params = AssemblyParams(obs_params=obs_params)
    
    key = random.PRNGKey(0)
    obs, state = env.reset(key, params)
    
    # obs shape is (n_agents, actual_obs_dim) where actual_obs_dim comes from environment
    assert obs.shape[0] == TEST_N_AGENTS
    assert obs.ndim == 2
    actual_obs_dim = obs.shape[1]
    print_success(f"Observations shape: {obs.shape} (obs_dim={actual_obs_dim})")
    
    print_success("All observation tests passed!")
    return True


# ============================================================================
# 10. Test Physics (jax_cus_gym/physics.py)
# ============================================================================

def test_physics():
    """Test physics simulation."""
    print_test_header("Physics (jax_cus_gym/physics.py)")
    
    from physics import (
        compute_pairwise_distances, compute_ball_to_ball_forces,
        integrate_dynamics, PhysicsParams
    )
    
    physics_params = PhysicsParams()
    
    # Test pairwise distances
    positions = jnp.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0]])
    rel_pos, distances, directions = compute_pairwise_distances(positions)
    
    assert distances.shape == (3, 3)
    assert rel_pos.shape == (3, 3, 2)
    print_success(f"compute_pairwise_distances: distances shape {distances.shape}")
    
    # Test ball-to-ball forces
    forces, is_colliding = compute_ball_to_ball_forces(
        positions,
        physics_params.agent_radius,
        physics_params.k_ball,
    )
    
    assert forces.shape == (3, 2)
    assert is_colliding.shape == (3, 3)  # Collision matrix: n_agents x n_agents
    print_success(f"compute_ball_to_ball_forces: forces shape {forces.shape}")
    
    # Test dynamics integration
    velocities = jnp.zeros((3, 2))
    control_forces = jnp.ones((3, 2)) * 0.1
    collision_forces_b2b = jnp.zeros((3, 2))
    collision_forces_b2w = jnp.zeros((3, 2))
    
    new_pos, new_vel, accel = integrate_dynamics(
        positions, velocities, control_forces,
        collision_forces_b2b, collision_forces_b2w,
        physics_params,
    )
    
    assert new_pos.shape == positions.shape
    assert new_vel.shape == velocities.shape
    print_success(f"integrate_dynamics: new_pos shape {new_pos.shape}")
    
    print_success("All physics tests passed!")
    return True


# ============================================================================
# 11. Test Rewards (jax_cus_gym/rewards.py)
# ============================================================================

def test_rewards():
    """Test reward computation."""
    print_test_header("Rewards (jax_cus_gym/rewards.py)")
    
    from rewards import compute_rewards, RewardParams
    from assembly_env import AssemblySwarmEnv, AssemblyParams
    from observations import get_k_nearest_neighbors_all_agents
    
    # Use same n_agents as environment will produce grid_mask
    env = AssemblySwarmEnv(n_agents=TEST_N_AGENTS)
    params = AssemblyParams()
    
    key = random.PRNGKey(0)
    _, state = env.reset(key, params)
    
    # Filter grid_centers to only valid entries
    valid_grid_centers = state.grid_centers[state.grid_mask]
    
    # Get neighbor indices (needed for compute_rewards)
    _, _, neighbor_indices, _ = get_k_nearest_neighbors_all_agents(
        state.positions,
        state.velocities,
        k=params.obs_params.topo_nei_max,
        d_sen=params.d_sen,
    )
    
    rewards, info = compute_rewards(
        state.positions,
        state.velocities,
        valid_grid_centers,
        state.l_cell,
        neighbor_indices,
        params.reward_params,
    )
    
    assert rewards.shape == (TEST_N_AGENTS,)
    assert "in_target" in info  # Check for actual info keys
    assert "is_colliding" in info
    print_success(f"Rewards shape: {rewards.shape}")
    print_success(f"Num in target: {info['num_in_target']}")
    
    print_success("All reward tests passed!")
    return True


# ============================================================================
# 12. Test Config (cfg/assembly_cfg.py)
# ============================================================================

def test_config():
    """Test configuration loading and conversion."""
    print_test_header("Config (cfg/assembly_cfg.py)")
    
    from cfg import (
        AssemblyTrainConfig, get_config,
        config_to_maddpg_config, config_to_assembly_params,
        get_checkpoint_dir, get_log_dir, get_eval_dir,
    )
    
    config = get_config()
    print_success(f"Loaded config: n_agents={config.n_agents}, n_parallel_envs={config.n_parallel_envs}")
    
    # Test MADDPG config conversion
    maddpg_config = config_to_maddpg_config(config, obs_dim=100, action_dim=2)
    assert maddpg_config.n_agents == config.n_agents
    print_success(f"MADDPG config: n_agents={maddpg_config.n_agents}")
    
    # Test assembly params conversion
    assembly_params = config_to_assembly_params(config)
    print_success(f"Assembly params created")
    
    # Test path functions
    checkpoint_dir = get_checkpoint_dir(config, "test_run")
    log_dir = get_log_dir(config, "test_run")
    eval_dir = get_eval_dir(config, "test_run")
    
    print_success(f"Checkpoint dir: ...{checkpoint_dir[-30:]}")
    print_success(f"Log dir: ...{log_dir[-30:]}")
    print_success(f"Eval dir: ...{eval_dir[-30:]}")
    
    print_success("All config tests passed!")
    return True


# ============================================================================
# 13. Test Shape Loader (jax_cus_gym/shape_loader.py)
# ============================================================================

def test_shape_loader():
    """Test shape loading functionality."""
    print_test_header("Shape Loader (jax_cus_gym/shape_loader.py)")
    
    from shape_loader import ShapeLibrary, create_procedural_shape
    
    # Test procedural shape creation with correct API
    grid_centers, l_cell = create_procedural_shape(
        shape_type="rectangle",
        n_cells_x=4,
        n_cells_y=4,
    )
    assert grid_centers.shape[0] == 16  # 4x4 grid
    assert grid_centers.shape[1] == 2
    print_success(f"Procedural rectangle shape: {grid_centers.shape}")
    
    # Test ring shape
    ring_centers, _ = create_procedural_shape(shape_type="ring", n_cells_x=4)
    print_success(f"Procedural ring shape: {ring_centers.shape}")
    
    # Test line shape
    line_centers, _ = create_procedural_shape(shape_type="line", n_cells_x=8)
    assert line_centers.shape[0] == 8
    print_success(f"Procedural line shape: {line_centers.shape}")
    
    print_success("All shape loader tests passed!")
    return True


# ============================================================================
# 14. Integration Test: Full Training Step
# ============================================================================

def test_training_integration():
    """Test a full training step to ensure everything works together."""
    print_test_header("Integration: Full Training Step")
    
    from algo.maddpg import MADDPG, MADDPGConfig
    from assembly_env import make_vec_env, AssemblyParams, compute_prior_policy
    from observations import compute_observation_dim
    
    # Setup
    n_agents = TEST_N_AGENTS
    n_envs = TEST_N_ENVS
    
    env, _, _, _ = make_vec_env(n_envs=n_envs, n_agents=n_agents)
    params = AssemblyParams()
    action_dim = 2
    
    # Create vectorized env functions
    @jax.jit
    def vec_reset(keys):
        return jax.vmap(lambda k: env.reset(k, params))(keys)
    
    @jax.jit
    def vec_step(keys, states, actions):
        return jax.vmap(lambda k, s, a: env.step(k, s, a, params))(keys, states, actions)
    
    # First reset to get actual obs_dim from environment
    key = random.PRNGKey(42)
    key, reset_key = random.split(key)
    reset_keys = random.split(reset_key, n_envs)
    obs_batch, env_states = vec_reset(reset_keys)
    obs_dim = obs_batch.shape[-1]  # Get actual obs_dim from environment output
    print_success(f"Environment reset: obs_batch shape = {obs_batch.shape}")
    
    # Create MADDPG with actual obs_dim
    maddpg_config = MADDPGConfig(
        n_agents=n_agents,
        obs_dims=tuple([obs_dim] * n_agents),
        action_dims=tuple([action_dim] * n_agents),
        hidden_dims=(32, 32),
        buffer_size=200,
        batch_size=16,
        warmup_steps=50,
    )
    
    maddpg = MADDPG(maddpg_config)
    maddpg_state = maddpg.init(key)
    print_success(f"MADDPG initialized: obs_dim={obs_dim}, action_dim={action_dim}")
    
    total_transitions = 0
    for step in range(TEST_MAX_STEPS):
        key, action_key, step_key = random.split(key, 3)
        
        # Select actions (batched)
        actions_batch, maddpg_state = maddpg.select_actions_batched(
            action_key, maddpg_state, obs_batch, explore=True
        )
        
        # Step environments
        step_keys = random.split(step_key, n_envs)
        next_obs_batch, env_states, rewards_batch, dones_batch, info = vec_step(
            step_keys, env_states, actions_batch
        )
        
        # Compute priors (batched)
        def compute_priors_single(positions, velocities, grid_centers, grid_mask, l_cell):
            return compute_prior_policy(
                positions, velocities, grid_centers, grid_mask, l_cell,
                params.reward_params.collision_threshold,
                params.d_sen,
            )
        
        prior_actions_batch = jax.vmap(compute_priors_single)(
            env_states.positions,
            env_states.velocities,
            env_states.grid_centers,
            env_states.grid_mask,
            env_states.l_cell,
        )
        
        # Store transitions (batched)
        maddpg_state = maddpg.store_transitions_batched(
            maddpg_state,
            obs_batch,
            actions_batch,
            rewards_batch,
            next_obs_batch,
            dones_batch,
            action_priors_batch=prior_actions_batch,
        )
        
        obs_batch = next_obs_batch
        total_transitions += n_envs
    
    print_success(f"Ran {TEST_MAX_STEPS} steps, stored {total_transitions} transitions")
    print_success(f"Buffer size: {maddpg_state.buffer_state.size}")
    
    # Try update
    key, update_key = random.split(key)
    maddpg_state, update_info = maddpg.update(update_key, maddpg_state)
    
    if update_info.get('can_update', True):
        print_success(f"Update performed successfully")
    else:
        print_success(f"Update skipped (buffer warmup)")
    
    print_success("Integration test passed!")
    return True


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("  JAX MARL VECTORIZATION TEST SUITE")
    print("=" * 60)
    print(f"  Test parameters:")
    print(f"    n_agents: {TEST_N_AGENTS}")
    print(f"    n_envs: {TEST_N_ENVS}")
    print(f"    obs_dim: {TEST_OBS_DIM}")
    print(f"    action_dim: {TEST_ACTION_DIM}")
    print("=" * 60)
    
    tests = [
        ("Networks", test_networks),
        ("Noise", test_noise),
        ("Buffers", test_buffers),
        ("Agents", test_agents),
        ("MADDPG", test_maddpg),
        ("Environment", test_environment),
        ("Vectorized Environment", test_vectorized_environment),
        ("Prior Policy", test_prior_policy),
        ("Observations", test_observations),
        ("Physics", test_physics),
        ("Rewards", test_rewards),
        ("Config", test_config),
        ("Shape Loader", test_shape_loader),
        ("Integration", test_training_integration),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            print_fail(f"FAILED: {e}")
            # Print short traceback for debugging
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error[:60]}...")
    
    print("=" * 60)
    print(f"  TOTAL: {passed}/{len(results)} tests passed")
    if failed > 0:
        print(f"  {failed} tests FAILED")
    else:
        print("  All tests PASSED! ✓")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
