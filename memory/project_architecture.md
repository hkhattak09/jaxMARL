---
name: jaxMARL Project Architecture
description: Complete architectural overview of jax_marl and jax_cus_gym — modules, classes, data flow, interaction patterns
type: project
---

## The System

JAX-based multi-agent RL system for swarm assembly task. Two packages:
- `jax_cus_gym/` — environment layer (physics, observations, rewards, shapes)
- `jax_marl/` — algorithm layer (MADDPG+TD3, training loop, config, LLM integration)

Entry point: `jax_marl/train/training.py` → calls `train(config)` from `train_assembly.py`
Config: `jax_marl/cfg/assembly_cfg.py` → active instance is the uncommented `config = AssemblyTrainConfig(...)` at module bottom (currently FULL SCALE)

**Why:** GPU-accelerated swarm robotics research with LLM-guided policy learning.
**How to apply:** Any new feature must be JAX-pure (no Python side effects, no in-place mutation, use `.replace()`). Sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` at import.

---

## Active Config (FULL SCALE — as of 2026-03-28)

| Parameter | Value | Notes |
|-----------|-------|-------|
| n_agents | 30 | Swarm size |
| n_parallel_envs | 8 | vmap'd envs |
| arena_size | 5.0 | Square arena |
| agent_radius | 0.035 | Matches MARL-LLM |
| d_sen | 0.4 | Sensing distance |
| max_steps | 200 | Steps/episode |
| hidden_dim | 256 | Network width |
| buffer_size | 80,000 | ~50 episodes × 1,600 transitions |
| batch_size | 2,048 | |
| warmup_steps | 50,000 | Steps before training |
| updates_per_step | 20 | Gradient steps per rollout step |
| noise_scale_initial | 0.9 | Decays to 0.5 |
| noise_scale_final | 0.5 | Noise floor (clamped at ep 1330) |
| noise_decay_steps | 2,112,000 | 44% × 3000 eps × 1600 steps |
| prior_weight | 0.5 | LLM regularization strength |
| use_td3 | True | Twin critics + delayed actor |
| policy_delay | 2 | Actor update every 2 critic updates |
| n_episodes | 3,000 | |
| eval_interval | 500 | |
| gamma | 0.95 | Discount |
| tau | 0.01 | Soft target update |

**Buffer sizing formula:** `0.05 × n_episodes × max_steps × n_parallel_envs`
**noise_decay_steps formula:** `0.44 × n_episodes × max_steps × n_parallel_envs`

---

## jax_cus_gym — Environment Layer

### environment.py
- `MultiAgentEnv` — abstract base, `step()` auto-resets on done, calls `step_env()` + `reset_env()`
- `EnvState(time: int)` / `EnvParams(max_steps_in_episode: int = 500)` — base dataclasses

### assembly_env.py — main env
- `AssemblyState(EnvState)` — positions(n,2), velocities(n,2), grid_centers(max_n_grid,2), grid_mask, l_cell, trajectory(traj_len,n,2), traj_idx, shape_idx/rotation/scale/offset, occupied_mask(max_n_grid,), in_target(n,), is_colliding(n,)
- `AssemblyParams` — arena_size, agent_radius, max_velocity, max_acceleration, k_neighbors, d_sen, physics/obs_params/reward_params sub-params, max_steps, dt, randomize_shape/rotation/scale/offset, traj_len, r_avoid (None=auto-compute), reward_mode
- `AssemblySwarmEnv(n_agents, shape_library)` — reset, step_env, get_obs_dim, get_action_dim
- `compute_prior_policy(state, params) -> actions` — rule-based repulsion/attraction/sync baseline (3 forces: attraction to target, neighbor repulsion, velocity sync)
- `compute_r_avoid(grid_mask, n_agents, l_cell, r_avoid) -> float` — auto-compute: `sqrt(4*n_grid/(n_agents*pi)) * l_cell`
- `make_assembly_env(n_agents, ...) -> (env, params)` — factory
- `make_vec_env(n_envs, n_agents, ...) -> (env, _, _, _)` — vmap wrapper

### physics.py
- `PhysicsParams` — k_ball=30.0, k_wall=100.0, c_wall=5.0, c_aero=1.2, agent_radius, agent_mass=1.0, dt, vel_max
- `compute_pairwise_distances(positions) -> (rel_pos, distances, directions)` — shapes (n,n,2), (n,n), (n,n,2)
- `compute_pairwise_distances_periodic(...)` — with arena wrapping
- `compute_collision_forces(positions, velocities, params) -> forces` — spring-damper
- `physics_step(positions, velocities, accelerations, params) -> (new_positions, new_velocities)` — Euler integration + wall clamp

### observations.py
- `ObservationParams` — topo_nei_max=6, num_obs_grid_max=80, d_sen, include_self_state=True, normalize_obs=True, l_max, vel_max
- Obs layout: self(4) + k_neighbors×4(24) + target(4) + grid×2×80(160) = **192 dims** default
- `compute_observation_dim(obs_params) -> int`
- `get_k_nearest_neighbors(agent_idx, positions, velocities, k, d_sen, ...) -> (rel_pos, rel_vel, mask, indices)`
- `get_k_nearest_neighbors_all_agents(...)` — vmapped
- `compute_observations(positions, velocities, grid_centers, grid_mask, l_cell, obs_params) -> (n_agents, obs_dim)`

### rewards.py
- `RewardParams` — reward_entering=1.0, penalty_collision=0.0 (no penalty), reward_exploration=0.1, collision_threshold=0.15 (=2×agent_radius), exploration_threshold=0.05, cosine_decay_delta=0.0, reward_mode=0
- Constants: `REWARD_MODE_INDIVIDUAL=0`, `REWARD_MODE_SHARED_MEAN=1`, `REWARD_MODE_SHARED_MAX=2`
- `rho_cos_dec(z, r, delta) -> weights` — cosine decay 1→0 over [delta×r, r)
- `compute_in_target(positions, grid_centers, l_cell) -> mask`
- `compute_agent_collisions(positions, agent_radius) -> bool mask`
- `compute_rewards(positions, grid_centers, grid_mask, l_cell, dones, is_colliding, in_target, reward_params) -> (n_agents,)`
- Reward logic: entering_reward if in_target and not colliding + exploration bonus; then mode aggregation

### shape_loader.py
- `ShapeLibrary` — grid_centers(n_shapes, max_n_grid, 2), l_cells(n_shapes,), n_grids(n_shapes,), shape_masks(n_shapes, max_n_grid), n_shapes, max_n_grid
- `load_shapes_from_pickle(filepath) -> ShapeLibrary` — from `fig/results.pkl` (7 shapes in actual file)
- `create_shape_library_from_procedural(shape_types, n_cells, l_cell) -> ShapeLibrary`
- `get_shape_from_library(library, shape_idx) -> (grid_centers, l_cell, mask)`
- `apply_shape_transform(grid_centers, l_cell, rotation, scale, offset) -> transformed`

### spaces.py — Box, Discrete, MultiAgentActionSpace, MultiAgentObservationSpace

### maddpg_wrapper.py — MADDPGWrapper
- Converts env to MADDPG-compatible interface
- `get_global_state(state) -> global_state` — concatenated obs+shape info for centralized critic
- `global_state_dim = n_agents×4 + max_n_grid×2 + 1`

### robot_api.py — NeighborInfo, TargetInfo, GridInfo; JAX-compatible perception functions for LLM policies

### visualize/renderer.py — Renderer: render(), render_trajectory(), render_to_gif()

---

## jax_marl — Algorithm Layer

### cfg/assembly_cfg.py
- `AssemblyTrainConfig(NamedTuple)` — single master config, active instance is `config = AssemblyTrainConfig(...)` at bottom (FULL SCALE preset)
- Preset slots: `config = None` (uses class defaults), FAST DEBUG, SMALL SCALE, FULL SCALE
- `get_config()` — returns active `config` or default
- `config_to_maddpg_config(config, obs_dim, action_dim) -> MADDPGConfig`
- `config_to_assembly_params(config) -> AssemblyParams`
- Path helpers: `get_shape_file_path`, `get_checkpoint_dir`, `get_log_dir`, `get_eval_dir`
  - Checkpoint: `jax_marl/checkpoints/assembly/<run_name>/`
  - Logs: `jax_marl/logs/assembly/<run_name>/`
  - Eval videos: `jax_marl/eval_videos/assembly/<run_name>/`

### algo/networks.py
- `Actor(action_dim, hidden_dims=(64,64), activation=leaky_relu, use_layer_norm=False, dropout_rate=0.0)` — obs→tanh→[-1,1]; orthogonal init
- `ActorDiscrete(n_actions, ...)` — outputs logits
- `Critic(...)` — concat(global_state, all_actions)→scalar Q
- `CriticTwin(...)` — two critics, q_min() for TD3
- Factories: `create_actor`, `create_critic`, `create_maddpg_networks`, `create_all_agents_networks`

### algo/agents.py
- `DDPGAgentState` (@struct.dataclass) — actor/critic/target params + opt states + noise_state + step
- `AgentConfig(NamedTuple)` — obs_dim, action_dim, critic_input_dim, hidden_dims, lr_actor/critic, gamma, tau, discrete_action, noise_type, use_layer_norm, max_grad_norm
- `create_agent(key, config) -> DDPGAgentState`
- `select_action(actor_params, obs)`, `select_action_with_noise(key, actor_params, obs, noise_scale)`
- `select_target_action_with_smoothing(key, actor, target_params, obs, target_noise, target_noise_clip)` — TD3
- `update_critic(...)`, `update_actor(...)`, `update_critic_td3(...)`, `update_actor_td3(...)`
- `update_targets(agent_state, tau)` — soft update

### algo/maddpg.py
- `MADDPGConfig(NamedTuple)` — n_agents, obs_dims(tuple), action_dims(tuple), hidden_dims, lr_actor/critic, gamma=0.95, tau=0.01, buffer_size, batch_size, warmup_steps, noise_scale_initial/final/decay_steps, use_td3=True, policy_delay=2, target_noise=0.2, target_noise_clip=0.5, prior_weight, updates_per_step, use_layer_norm=True, max_grad_norm=None, shared_critic=False
- `MADDPGState` (@struct.dataclass) — agent_states (stacked DDPGAgentState), buffer_state, step, episode, noise_scale
- `MADDPG(config)`:
  - `init(key) -> MADDPGState` — stacked agent states via vmap
  - `select_actions(key, state, observations, explore=True) -> (actions, log_probs, state)` — stateless Gaussian noise
  - `select_actions_batched(key, state, obs_batch, explore=True)` — over n_parallel_envs
  - `store_transitions_batched(state, obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch, action_priors_batch) -> state`
  - `create_jit_update() -> jit_update_fn` — returns JIT-compiled update function
  - `increment_episode(state)`, `reset_noise(state)`
  - `get_params(state)`, `load_params(state, params)` — checkpointing

**Note:** Uses Gaussian noise (stateless), not OU noise. Critic sees (global_state, all_actions) — centralized training.

### algo/buffers.py
- `Transition(NamedTuple)` — obs, actions, rewards, next_obs, dones, global_state, next_global_state, log_probs, action_priors (all per-agent)
- `ReplayBufferState` (@struct.dataclass) — pre-allocated JAX arrays + position(int32) + size(int32), optional global_states/log_probs/action_priors
- `ReplayBuffer(capacity, n_agents, obs_dim, action_dim, global_state_dim, ...)`:
  - `init() -> state`, `add(state, transition) -> state` (circular), `add_batch(state, transitions) -> state`
  - `sample(state, key, batch_size) -> BatchTransition`, `can_sample(state, batch_size) -> bool`
  - All JIT-compatible (pure JAX ops)
- `PerAgentReplayBuffer` — single-agent variant

### algo/noise.py
- `gaussian_noise`, `add_gaussian_noise` — stateless, no OU
- `OUNoiseState`, `OUNoiseParams`, `ou_noise_step`, `ou_noise_reset` — available but not used in active config
- `NoiseScheduler` — linear/exponential/cosine/warmup_linear schedules

### algo/utils.py
- `soft_update(target, online, tau)`, `hard_update`
- `gumbel_softmax`, `onehot_from_logits` (discrete)
- `td_target(rewards, next_q, gamma, done) -> target`
- `huber_loss`, `mse_loss`, `get_gradient_norm`

### train/train_assembly.py
- `TrainingMetrics(NamedTuple)` — episode_reward_mean/std, coverage_rate, distribution_uniformity, voronoi_uniformity, collision_rate, avg_dist_to_target, step_time, train_time, noise_scale, buffer_size
- `TrainingState` (@struct.dataclass) — maddpg_state, env_states (batched n_parallel_envs), key, episode, total_steps, best_reward
- `RolloutCarry` (@struct.dataclass) — key, obs_batch, env_states, maddpg_state, episode_rewards, total_collision, final_coverage, final_distribution_uniformity, final_voronoi_uniformity, done_flag
- `RolloutMetrics` (@struct.dataclass) — reward, coverage, collision, distribution_uniformity, voronoi_uniformity
- `create_training_state(config, key) -> (TrainingState, env, maddpg, params, vec_reset, vec_step)` — loads 7 shapes from pickle, vmaps env, gets real obs_dim=192 from test reset
- `create_jit_rollout_fn(maddpg, params, config, vec_step)` — uses `jax.lax.scan` over steps; computes prior actions if prior_weight>0; stores transitions; tracks metrics
- `run_episode(...)` — single episode driver
- `train(config, checkpoint_path=None)` — main loop: rollout → store → update (JIT) → log JSON → checkpoint
  - Memory report at init and post-compile
  - Incremental checkpoints every `save_interval` episodes
  - Eval runs at `eval_interval`, saves GIF

### train/training.py — Entry point script (calls `train(config)`)

### train/memory_utils.py
- `memory_report(maddpg_state, training_state, config, label="") -> dict` — prints breakdown:
  - Device memory: in use, peak, limit, free
  - Components: replay buffer, actor/critic networks (params+targets+opt state), env states, XLA/other
  - Scaling headroom: max buffer_size at 85% free memory, extra agents that fit
  - Returns dict with all values in bytes

### eval/evaluate.py
- `EvalConfig` (@dataclass) — checkpoint_path (required), shape_file, output_dir, n_episodes_per_shape=5, max_steps, seed, save_videos, video_fps, verbose
- `ShapeResult` (@dataclass) — shape_idx, shape_name, n_episodes, mean_reward/std, mean_coverage, mean_collision_rate
- `evaluate_on_all_shapes(config)` — no domain randomization, fixed shape/rotation/scale
- `evaluate_single_shape(config, shape_idx)`

### eval/run_eval.py — CLI entry point for evaluation

### llm/ — LLM policy generation
- `gpt_client.py` — GPTClient, `call_gpt(prompt, api_key, model)`
- `prompts.py` — `build_generation_prompt`, `GENERATION_PROMPT_TEMPLATE`
- `parser.py` — `parse_code_blocks`, `extract_functions`, `validate_generated_code`
- `generator.py` — `generate_reward_and_policy`, `load_functions_from_file(filepath) -> (reward_fn, policy_fn)`
- Prior policy used as regularizer: `actor_loss += prior_weight × ||actor(obs) - action_prior||²`

---

## Critical Data Flow

```
AssemblyTrainConfig (assembly_cfg.py, active = FULL SCALE)
  ↓ config_to_maddpg_config / config_to_assembly_params
MADDPGConfig + AssemblyParams
  ↓
MADDPG.init() + AssemblySwarmEnv + ShapeLibrary (7 shapes)
  ↓
vec_reset → obs(n_envs=8, n_agents=30, obs_dim=192), env_states
  ↓ [per episode, via lax.scan over 200 steps]
MADDPG.select_actions_batched(obs) → actions(8, 30, 2)
  + compute_prior_policy → action_priors(8, 30, 2)  [if prior_weight > 0]
vec_step(actions) → obs'(8,30,192), rewards(8,30), done, state'
store_transitions_batched → ReplayBuffer circular write
  ↓ [every step, updates_per_step=20 times per step]
buffer.sample(batch_size=2048) → BatchTransition
critic_update: td_target = reward + γ × min(Q1_target, Q2_target)(next_obs, target_actions+noise)
actor_update (every policy_delay=2): loss = -Q(obs, actor(obs)) + prior_weight×||actor-prior||²
soft_update(τ=0.01)
```

## Observation Breakdown (192 dims, active config)

```
[0:4]    self pos(2) + vel(2)
[4:28]   6 neighbors × (rel_pos(2) + rel_vel(2))
[28:32]  target cell rel_pos(2) + zeros(2)
[32:192] 80 grid cells × rel_pos(2)   (zero-padded if fewer)
```

## Key Shapes

| Tensor | Shape |
|--------|-------|
| observations | (n_envs=8, n_agents=30, obs_dim=192) |
| actions | (n_envs=8, n_agents=30, 2) |
| rewards | (n_envs=8, n_agents=30) |
| action_priors | (n_envs=8, n_agents=30, 2) |
| positions | (n_agents=30, 2) |
| grid_centers | (max_n_grid, 2) |
| global_state | (n_agents×4 + max_n_grid×2 + 1,) |
| trajectory | (traj_len=15, n_agents=30, 2) |

## Training Results (run 2026-03-27)

30 agents, 8 parallel envs, 3000 episodes on A100 (40GB):
- Memory: 4.94 GB in-use, 10.38 GB peak (post-compile)
- Replay buffer: 3.50 GB (80,000 slots × 30 agents)
- Convergence: coverage ~0.58, DistUnif ~0.11 by ep ~500-580
- Noise floor 0.5 reached at ep 1330 (noise_decay_steps=2,112,000 / 1600 steps/ep ≈ 1320 eps)
- Eval rewards 0.89-0.92 (noise-free greedy); training rewards ~0.75-0.81
- Collisions → near-zero by ep 500 (from ~0.02 at start)
- Speed: ~3 seconds/episode, ~100 min total
- Checkpoints every 400 eps (incremental); evals at ep 499/998/1497/1996/2495
