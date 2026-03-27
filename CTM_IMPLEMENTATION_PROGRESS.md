# CTM Temporal Critic — Implementation Progress

## Status: Complete (Phases 1–4). Ready to enable and run.

---

## What Was Built

The CTM (Continuous Thought Machine) temporal critic replaces the MLP critic in MADDPG with a recurrent model that cross-attends over a trajectory history window at each internal tick, producing a Q-value and a certainty score per tick. The most-certain tick's Q-value is used for both the Bellman target and the actor update.

---

## Files Modified

### `jax_marl/algo/buffers.py`
- Added `trajectory` and `next_trajectory` optional fields to `Transition`, `BatchTransition`, `ReplayBufferState`
- `ReplayBuffer.__init__` accepts `traj_len`, `store_trajectories`, `trajectory_dtype=jnp.float16`
- When enabled, allocates `(capacity, traj_len, n_agents, 2)` arrays in float16 (D6)
- `add()`, `add_batch()`, `sample()` all handle trajectory fields transparently; disabled path is zero-overhead

### `jax_marl/algo/networks.py`
- Added `TrajectoryEncoder(nn.Module)` — encodes a position history window into hybrid kv tokens:
  - Temporal branch: finite-diff velocity + position → `Dense(d_input)` → `traj_len` tokens
  - Per-agent branch: concat(obs, action) → `Dense(d_input)` → `n_agents` tokens
  - Output: `LayerNorm(concat([traj_kv, agent_kv]))` shape `(B, traj_len+n_agents, d_input)`
  - Float16 trajectory cast to float32 at entry (D6)
- Added `CTMCritic(nn.Module)` — full CTM architecture replacing the CNN backbone:
  - Pairwise outer-product EMA synchronization (action synch → attention query, output synch → Q head)
  - Multi-head cross-attention over kv tokens
  - GLU synapse with LayerNorm
  - Sliding trace window + SuperLinear (NLM) trace processor (from jaxCTM)
  - Scalar Q head: `Dense(1)`
  - Hybrid certainty head (D2): `cert_t = cert_raw*(1-α) + (1-tick_var)*α`
    - `cert_raw` = learned sigmoid head (trained via auxiliary loss — see below)
    - `tick_var` = variance of Q across ticks so far, normalized to [0,1) — structural, parameter-free
  - Decay params clamped to [0, 15] at forward entry (D8)
  - Outputs: `q_values (B, 1, T)`, `certainties (B, 2, T)`
- Added `create_ctm_critic(key, n_agents, obs_dim, action_dim, traj_len, ...)` factory

### `jax_marl/algo/agents.py`
- Added `compute_ctm_critic_loss(...)`:
  - Target network selects most-certain tick of next state → certainty-discounted Bellman target (D3):
    `bellman_target = r + γ * Q_next[tick_certain] * cert_score * (1 - done)`
  - Dual-tick MSE loss (D2): `loss = 0.5 * (MSE(Q[tick_best], target) + MSE(Q[tick_certain], target))`
  - Auxiliary certainty loss: `log_softmax` cross-entropy over ticks with `tick_best` as label — trains `cert_projector` weights (without this, `argmax` blocks all gradients to the learned certainty head)
  - Returns `(loss, info)` with `ctm_critic_loss`, `cert_aux_loss`, `q_mean`, `bellman_target_mean`, `cert_score_mean`, `td_error_mean`
- Added `compute_actor_loss_ctm(...)`:
  - Substitutes agent i's action via `jax.lax.dynamic_update_slice` (same interface as TD3 actor)
  - Queries CTM at `tick_certain`, loss = `-mean(Q[tick_certain])` + optional prior regularization
- Added `update_critic_ctm(...)`:
  - Standard gradient update via optax
  - Post-update D8 clamp: `decay_params_action` and `decay_params_out` clipped to [0, 15] via `tree_flatten_with_path` + `DictKey` matching
  - Does **not** increment `step` — `update_targets` owns that counter for all paths
- Added `update_actor_ctm(...)`: standard gradient update

### `jax_marl/algo/maddpg.py`
- Added `_reorder_trajectory(raw_traj, traj_idx, traj_len)` — unwraps circular buffer to chronological order. Called at store time (D7), not sample time, so the O(traj_len) gather is paid once per store rather than once per sample (~30× savings)
- Added 12 CTM fields to `MADDPGConfig`:
  ```
  use_ctm_critic, ctm_traj_len, ctm_iterations, ctm_d_model, ctm_d_input,
  ctm_memory_length, ctm_heads, ctm_n_synch_out, ctm_n_synch_action,
  ctm_memory_hidden_dims, ctm_alpha_initial, ctm_alpha_final, ctm_alpha_anneal_steps
  ```
- `MADDPG.__init__`: constructs `CTMCritic` instance when `use_ctm_critic=True`; configures buffer with trajectory storage; warns if `use_td3=True` alongside `use_ctm_critic=True` (target smoothing conflicts with certainty-discounted Bellman)
- `MADDPG.init()`: uses CTM dummy inputs `(1, traj_len, n_agents, 2)` for param init when CTM enabled
- `store_transitions_batched`: vmaps `_reorder_trajectory` over env batch before writing to buffer
- `create_jit_update`: Python-level branch at JIT trace time (not a traced conditional):
  - Alpha annealing: `α = α_initial - (α_initial - α_final) * min(step / anneal_steps, 1.0)`
  - CTM critic update path: vmaps `update_critic_ctm` over agents
  - CTM actor update path: vmaps `update_actor_ctm` over agents
  - Policy delay applies to CTM path (hardcoded via `or use_ctm_critic` condition)
  - `lax.cond` branches (`do_update` / `skip_update`) carry `extra_info` dict with CTM monitoring metrics — both branches return matching pytree structure

### `jax_marl/cfg/assembly_cfg.py`
- Added all 12 CTM fields to `AssemblyTrainConfig` (NamedTuple) with safe defaults (`use_ctm_critic=False`)
- `config_to_maddpg_config()` passes all CTM fields through to `MADDPGConfig`

### `jax_marl/train/train_assembly.py`
- `rollout_step` passes `trajectory_batch`, `trajectory_idx_batch`, `next_trajectory_batch`, `next_trajectory_idx_batch` from `env_states` to `store_transitions_batched`

---

## Test Files Written

Individual test files — run each independently or via the runner:

| File | What it covers |
|---|---|
| `jax_marl/algo/tests/test_ctm_reorder.py` | `_reorder_trajectory`: identity, mid-buffer, full-wrap, all offsets, vmapped batch |
| `jax_marl/algo/tests/test_ctm_buffer.py` | Buffer trajectory allocation (float16), add/sample shapes, circular overwrite |
| `jax_marl/algo/tests/test_ctm_encoder.py` | `TrajectoryEncoder` output shapes, float16 input, zero trajectory, finiteness |
| `jax_marl/algo/tests/test_ctm_critic.py` | `CTMCritic` output shapes, certainty ∈ [0,1], sum=1, alpha blends, determinism, target network |
| `jax_marl/algo/tests/test_ctm_loss.py` | Critic/actor loss scalars, Bellman target (done kills bootstrap), gradient flow, prior regularization |
| `jax_marl/algo/tests/test_ctm_update.py` | Param updates, step counter ownership, D8 decay clamping above/below |

Run all 57 tests:
```bash
python tests/run_ctm_tests.py
# or with filter
python tests/run_ctm_tests.py -k encoder
```

Last run: **57/57 passed** (137s on CPU).

---

## How to Enable CTM

In `jax_marl/cfg/assembly_cfg.py`, update the active config instance at the bottom:

```python
config = AssemblyTrainConfig(
    n_agents=30,
    use_ctm_critic=True,
    use_td3=False,          # TD3's twin-critic and actor update are bypassed by CTM
                            # target smoothing conflicts with certainty-discounted Bellman (D3)
    # CTM hyperparameters (defaults shown):
    ctm_traj_len=15,        # must match AssemblyParams.traj_len
    ctm_iterations=8,
    ctm_d_model=64,
    ctm_d_input=128,
    ctm_memory_length=5,
    ctm_heads=2,
    ctm_n_synch_out=8,
    ctm_n_synch_action=8,
    ctm_memory_hidden_dims=8,
    ctm_alpha_initial=1.0,  # start with structural certainty (tick variance)
    ctm_alpha_final=0.0,    # anneal to learned certainty
    ctm_alpha_anneal_steps=100000,
    ...
)
```

To switch back to standard MLP+TD3:
```python
config = AssemblyTrainConfig(
    n_agents=30,
    use_ctm_critic=False,
    use_td3=True,
    ...
)
```

No other code changes required — the hot path branches at JIT trace time.

---

## Key Design Decisions Made

| Decision | Choice | Reason |
|---|---|---|
| D2 Certainty | Hybrid tick-variance + learned sigmoid | Tick-variance works from step 0; learned head trained via auxiliary cert loss as alpha anneals |
| D3 Bellman target | Certainty-discounted: `r + γ*Q*cert*(1-done)` | Conservative when model is uncertain; no separate twin critics needed |
| D4 kv tokens | 15 temporal + n_agents per-agent = 20 total | Trajectory gives temporal context; per-agent tokens give current action context |
| D6 Float16 storage | Trajectories stored as float16, cast to float32 at forward entry | ~50% buffer memory saving; negligible precision loss for positions |
| D7 Reorder at store | `_reorder_trajectory` called in `store_transitions_batched` | Pays gather cost once per store, not once per sample (~30× saving) |
| D8 Decay clamping | Clip `decay_params` to [0, 15] post-update and at forward entry | Prevents `exp(-decay) > 1` EMA divergence |
| use_td3 with CTM | Set `use_td3=False` | With CTM, `use_td3=True` only adds target smoothing noise — this conflicts with certainty-discounted Bellman which already provides conservative estimates |

---

## Known Non-Issues

- **`per_agent_feat` in `TrajectoryEncoder`** — computed but discarded (`kv, _ = traj_encoder(...)`). Intentional placeholder for a future Option-3 credit-assignment upgrade. The `agent_feat` Dense layer and its optimizer state are allocated but XLA eliminates forward computation. Can be removed if Option-3 is not pursued.

- **`agent_state.step` ownership** — incremented once per full update by `update_targets` for all paths (CTM and MLP). Neither `update_critic_ctm` nor `update_actor_ctm` touches it. Consistent with `update_critic_td3` / `update_actor_td3`. The policy delay check uses `MADDPGState.step` (global), not `agent_state.step`.

---

## Monitoring Metrics (logged per update when CTM enabled)

| Key | What it tracks |
|---|---|
| `ctm_q_mean` | Mean Q at most-certain tick — watch for divergence |
| `ctm_bellman_target` | Mean Bellman target — should be stable |
| `ctm_cert_score` | Mean certainty score used in target discount — if persistently near 0, targets are near-zero (conservative but may slow learning) |
| `ctm_td_error` | Mean `|Q[tick_certain] - target|` — primary convergence signal |
| `cert_aux_loss` | Certainty head cross-entropy — should decrease as the head learns which tick to be confident at |
