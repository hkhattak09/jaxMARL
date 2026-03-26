---
name: jaxMARL Project Architecture
description: Complete architectural overview of jax_marl and jax_cus_gym — modules, classes, data flow, interaction patterns
type: project
---

## The System

JAX-based multi-agent RL system for swarm assembly task. Two packages:
- `jax_cus_gym/` — environment layer (physics, observations, rewards, shapes)
- `jax_marl/` — algorithm layer (MADDPG+TD3, training loop, config, LLM integration)

Entry point: `jax_marl/train/train_assembly.py` → `train(config)`
Config: `jax_marl/cfg/assembly_cfg.py` → `config = AssemblyTrainConfig(...)` (active config is the uncommented instance at module bottom)

**Why:** GPU-accelerated swarm robotics research with LLM-guided policy learning.
**How to apply:** Any new feature must be JAX-pure (no Python side effects, no in-place mutation, use `.replace()`).

---

## jax_cus_gym — Environment Layer

### environment.py
- `MultiAgentEnv` — abstract base, `step()` auto-resets on done, calls `step_env()` + `reset_env()`
- `EnvState(time: int)` / `EnvParams(max_steps_in_episode: int = 500)` — base dataclasses

### assembly_env.py — main env
- `AssemblyState(EnvState)` — positions(n,2), velocities(n,2), grid_centers(max_n_grid,2), grid_mask, l_cell, trajectory(traj_len,n,2), traj_idx, shape_idx/rotation/scale/offset, occupied_mask(max_n_grid,), in_target(n,), is_colliding(n,)
- `AssemblyParams` — arena_size, agent_radius, max_velocity, max_acceleration, k_neighbors, d_sen, physics/obs_params/reward_params sub-params, max_steps, dt, randomize_shape/rotation/scale/offset, traj_len, r_avoid, reward_mode
- `AssemblySwarmEnv(n_agents, shape_library)` — reset, step_env, get_obs_dim, get_action_dim
- `compute_prior_policy(state, params) -> actions` — rule-based repulsion/attraction baseline
- `make_assembly_env(n_agents, ...) -> (env, params)` — factory
- `make_vec_env(n_envs, n_agents, ...) -> (env, _, _, _)` — vmap wrapper

### physics.py
- `PhysicsParams` — k_ball, k_wall, c_wall, c_aero, agent_radius, agent_mass, dt, vel_max
- `compute_pairwise_distances(positions) -> (rel_pos, distances, directions)` — shapes (n,n,2), (n,n), (n,n,2)
- `compute_pairwise_distances_periodic(...)` — with arena wrapping
- `compute_collision_forces(positions, velocities, params) -> forces` — spring-damper
- `physics_step(state, actions, params) -> (new_positions, new_velocities)` — Euler integration with wall clamp

### observations.py
- `ObservationParams` — topo_nei_max=6, num_obs_grid_max=80, d_sen, include_self_state, normalize_obs, l_max, vel_max
- Obs layout: self(4) + k_neighbors*4(24) + target(4) + grid*2*80(160) = 192 dims default
- `compute_observation_dim(obs_params) -> int`
- `get_k_nearest_neighbors(agent_idx, positions, velocities, k, d_sen) -> (rel_pos, rel_vel, mask, indices)`
- `get_k_nearest_neighbors_all_agents(...)` — vmapped
- `compute_observations(state, params) -> (n_agents, obs_dim)`

### rewards.py
- `RewardParams` — reward_entering=1.0, penalty_collision=0.0, reward_exploration=0.1, collision_threshold=0.15 (=2*agent_radius), exploration_threshold=0.05, cosine_decay_delta=0.0, reward_mode=0
- Constants: REWARD_MODE_INDIVIDUAL=0, REWARD_MODE_SHARED_MEAN=1, REWARD_MODE_SHARED_MAX=2
- `rho_cos_dec(z, r, delta) -> weights` — cosine decay 1→0 over [delta*r, r)
- `compute_in_target(positions, grid_centers, l_cell) -> mask`
- `compute_rewards(state, params) -> (n_agents,)` — entering + collision + exploration, then mode aggregation

### shape_loader.py
- `ShapeLibrary` — grid_centers(n_shapes, max_n_grid, 2), l_cells(n_shapes,), n_grids(n_shapes,), shape_masks(n_shapes, max_n_grid), n_shapes, max_n_grid
- `load_shapes_from_pickle(filepath) -> ShapeLibrary` — from `fig/results.pkl`
- `create_shape_library_from_procedural(shape_types, n_cells, l_cell) -> ShapeLibrary`
- `get_shape_from_library(library, shape_idx) -> (grid_centers, l_cell, mask)`
- `apply_shape_transform(grid_centers, rotation, scale, offset) -> transformed`

### spaces.py — Box, Discrete, MultiAgentActionSpace, MultiAgentObservationSpace

### maddpg_wrapper.py — MADDPGWrapper
- Converts env to MADDPG-compatible interface
- `get_global_state(state) -> global_state` — concatenated obs+shape info for centralized critic
- `global_state_dim = n_agents*4 + max_n_grid*2 + 1`

### robot_api.py — NeighborInfo, TargetInfo, GridInfo; JAX-compatible perception functions for LLM policies

### visualize/renderer.py — Renderer: render(), render_trajectory(), render_to_gif()

---

## jax_marl — Algorithm Layer

### cfg/assembly_cfg.py
- `AssemblyTrainConfig(NamedTuple)` — single master config, active instance is `config = AssemblyTrainConfig(...)` at bottom
- Active config: n_agents=20, n_parallel_envs=8, arena_size=5.0, agent_radius=0.035, d_sen=0.4, max_steps=200, hidden_dim=256, buffer_size=240000, batch_size=2048, warmup_steps=50000, n_episodes=3000, prior_weight=0.5, use_td3=True
- `get_config()` — returns active `config` or default
- `config_to_maddpg_config(config, obs_dim, action_dim) -> MADDPGConfig`
- `config_to_assembly_params(config) -> AssemblyParams` — creates sub-params hierarchy
- Path helpers: `get_shape_file_path`, `get_checkpoint_dir`, `get_log_dir`, `get_eval_dir`

### algo/networks.py
- `Actor(action_dim, hidden_dims=(64,64,64), activation=leaky_relu, use_layer_norm=False, dropout_rate=0.0)` — obs→tanh→[-1,1]
- `ActorDiscrete(n_actions, ...)` — outputs logits
- `Critic(...)` — concat(obs, action)→scalar Q
- `CriticTwin(...)` — two critics, q_min() for TD3
- Factories: `create_actor`, `create_critic`, `create_maddpg_networks`, `create_all_agents_networks`

### algo/agents.py
- `DDPGAgentState` — actor/critic/target params + opt states + noise_state + step
- `AgentConfig` — obs_dim, action_dim, critic_input_dim, hidden_dims, lr_actor/critic, gamma, tau, discrete_action, noise_type
- `create_agent(key, config)`, `select_action`, `select_action_with_noise`, `select_target_action_with_smoothing` (TD3), `update_critic`, `update_actor`, `update_targets(tau)`

### algo/maddpg.py
- `MADDPGConfig` — n_agents, obs_dims(tuple), action_dims(tuple), global_state_dim, hidden_dims, lr_actor/critic, gamma=0.95, tau=0.01, buffer_size, batch_size, warmup_steps, noise_schedule, use_td3, policy_delay, target_noise, target_noise_clip, prior_weight
- `MADDPGState` — agent_states[], buffer_state, step, episode, noise_scale
- `MADDPG(config)`:
  - `init(key) -> MADDPGState`
  - `select_actions(key, state, observations, explore=True) -> (actions, log_probs, state)`
  - `select_target_actions(state, next_observations) -> actions`
  - `store_transition(state, obs, actions, rewards, next_obs, dones) -> state`
  - `update(key, state) -> (state, info_dict)` — critic update every step, actor every policy_delay steps

### algo/buffers.py
- `Transition` — obs, actions, rewards, next_obs, dones, global_state, next_global_state, log_probs, action_priors
- `ReplayBufferState` — preallocated JAX arrays + position + size
- `ReplayBuffer(capacity, n_agents, obs_dim, action_dim)`:
  - `init() -> state`, `add(state, transition) -> state` (circular), `sample(key, batch_size) -> BatchTransition`

### algo/noise.py
- `gaussian_noise`, `add_gaussian_noise`
- `OUNoiseState`, `OUNoiseParams(mu, theta, sigma, dt)`, `ou_noise_step`, `ou_noise_reset`
- `NoiseScheduler` with linear/exponential/cosine/warmup_linear schedules

### algo/utils.py
- `soft_update(target, online, tau)`, `hard_update`
- `gumbel_softmax`, `onehot_from_logits` (for discrete)
- `td_target(rewards, next_q, gamma, done) -> target`
- `huber_loss`, `mse_loss`, `get_gradient_norm`

### train/train_assembly.py
- `TrainingMetrics` — reward_mean/std, coverage_rate, distribution_uniformity, voronoi_uniformity, collision_rate, avg_dist_to_target, step_time, train_time, noise_scale, buffer_size
- `TrainingState` — maddpg_state, env_states(batched n_parallel_envs), key, episode, total_steps, best_reward
- `create_training_state(config, key) -> (TrainingState, env, maddpg, params, vec_reset, vec_step)` — loads shapes, vmaps env, gets real obs_dim from test reset
- `vec_reset = jax.jit(vmap(env.reset))`, `vec_step = jax.jit(vmap(env.step))`
- `train(config)` — main loop: rollout → store → update → log → checkpoint

### llm/ — LLM policy generation
- `gpt_client.py` — GPTClient, `call_gpt(prompt, api_key, model)`
- `prompts.py` — `build_generation_prompt`, `GENERATION_PROMPT_TEMPLATE`
- `parser.py` — `parse_code_blocks`, `extract_functions`, `validate_generated_code`
- `generator.py` — `generate_reward_and_policy`, `load_functions_from_file(filepath) -> (reward_fn, policy_fn)`
- Prior policy used as regularizer: `loss += prior_weight * ||actor(obs) - action_prior||²`

---

## Critical Data Flow

```
AssemblyTrainConfig
  ↓ config_to_maddpg_config / config_to_assembly_params
MADDPGConfig + AssemblyParams
  ↓
MADDPG.init() + AssemblySwarmEnv + ShapeLibrary
  ↓
vec_reset → obs(n_envs, n_agents, obs_dim), env_states
  ↓ [per episode]
MADDPG.select_actions(obs) → actions(n_envs, n_agents, 2)
vec_step(actions) → obs', rewards(n_envs, n_agents), done, state'
  obs' = compute_observations(state', params)  [in env]
  rewards = compute_rewards(state', params)    [in env]
store_transition → ReplayBuffer circular write
  ↓ [every update_every steps, updates_per_step times]
buffer.sample() → BatchTransition(batch, n_agents, ...)
critic_update: td_target = reward + gamma * min_Q_target(next_obs, target_actions)
actor_update: loss = -Q(obs, actor(obs)) + prior_weight*||actor-prior||²  [every policy_delay]
soft_update(tau=0.01)
```

## Observation Breakdown (192 dims, default params)

```
[0:4]    self pos(2) + vel(2)
[4:28]   6 neighbors × (rel_pos(2) + rel_vel(2))
[28:32]  target cell rel_pos(2) + zeros(2)
[32:192] 80 grid cells × rel_pos(2)   (zero-padded if fewer)
```

## Key Shapes

| Tensor | Shape |
|--------|-------|
| observations | (n_envs, n_agents, obs_dim) |
| actions | (n_envs, n_agents, 2) |
| rewards | (n_envs, n_agents) |
| positions | (n_agents, 2) |
| grid_centers | (max_n_grid, 2) |
| global_state | (n_agents*4 + max_n_grid*2 + 1,) |
| trajectory | (traj_len=15, n_agents, 2) |
