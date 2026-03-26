# CTM Temporal Critic — Design Decisions

Decisions made during planning. Each entry states what was chosen, the alternatives
that were considered, and the reasoning.

---

## D1 — Module Structure

**Decision:** Create standalone `CTMCritic` in `jax_marl/algo/networks.py`, importing
only `SuperLinear` from `jaxCTM.layers`. Do not modify `ContinuousThoughtMachine`.

**Alternatives considered:**
- Modify `ContinuousThoughtMachine` to accept optional pre-computed `kv`, skipping CNN.

**Why:**
The existing CTM allocates CNN + BatchNorm weights that would be unused dead parameters.
BatchNorm requires `batch_stats` in the optimizer state — that machinery would bleed
into the RL training loop. The recurrent loop for regression differs enough from
classification (no entropy certainty, different output head) that sharing code means
constant conditional branches throughout. `jaxCTM/` stays untouched as a standalone
research artifact. The only true reuse is `SuperLinear`, which is a clean import.

---

## D2 — Certainty Mechanism for Scalar Q Output

**Decision:** Hybrid — tick-variance structural certainty blended with a learned
certainty head, with blend parameter `α` annealed from 1 → 0 over training.

```
q_t        = Dense(1)(synch_out_t)               # Q head
cert_raw   = Dense(1)(synch_out_t) → sigmoid     # Learned certainty head
tick_var   = Var(q_0..q_t) / (1 + Var(q_0..q_t)) # Structural (parameter-free)
cert_t     = cert_raw * (1 - α) + (1 - tick_var) * α
```

Certainty output format preserved: `(B, 2, T)` where `[:, 1, :]` = certainty score.

**Alternatives considered:**
- A: `out_dims=2`, treat output as `[Q, dummy]`, use entropy of 2-class softmax.
  Rejected: the dummy dimension has no meaning; certainty becomes a hack.
- B: Separate learned `Dense(1) → sigmoid` head only.
  Rejected alone: uncalibrated at step 0, could slow early training significantly.
- C: Tick-variance only (parameter-free).
  Rejected alone: can't be trained to be confident earlier; purely reactive.

**Why hybrid:**
Tick-variance is calibrated from step 0 — variance is always meaningful regardless
of training progress. The learned head takes over as `α` anneals to 0, providing
a richer signal once calibrated. The dual-tick loss trains the learned head to be
high exactly where `q_t ≈ bellman_target`, so both mechanisms reinforce each other.

**Initialization:** `certainty_projector` bias initialized to `+2.0` so
`sigmoid(2.0) ≈ 0.88`. Model starts fairly confident and learns to be uncertain
only where needed — not the reverse, which would slow learning.

**`α` annealing:** Lives in `MADDPGConfig` / `AssemblyTrainConfig`, passed to
`compute_ctm_critic_loss()` at each call. Not a model parameter.

---

## D3 — Overestimation Protection

**Decision:** Single `CTMCritic` (no twin critics). Use certainty-discounted Bellman
target instead of `min(Q1, Q2)`:

```
next_q, next_cert  = target_ctm(next_traj, next_obs, next_actions)
tick_certain       = argmax(next_cert[:, 1, :], axis=-1)          # (B,)
q_at_certain       = take_along_axis(next_q[:, 0, :], tick_certain)  # (B,)
cert_score         = take_along_axis(next_cert[:, 1, :], tick_certain)  # (B,)
bellman_target     = r + γ * q_at_certain * cert_score * (1 - done)
```

When `cert_score = 1.0`: standard DDPG target.
When `cert_score → 0`: target collapses to `r` alone — maximally conservative.

**Alternatives considered:**
- `CTMCriticTwin`: shared backbone, two Q-heads, `min(Q1, Q2)` for target.

**Why certainty-discount:**
TD3's twin critics are an implicit uncertainty estimator: two networks diverge most
in high-uncertainty regions, and the minimum exploits that divergence. CTM certainty
is explicit uncertainty modeling via the same insight, more principled, and half the
structural overhead. The certainty-discounted target is adaptive — conservative
exactly when the model says it should be, not always.

During the `α=1.0` warmup phase the Bellman target uses tick-variance pessimism
(structural, calibrated from step 0), providing TD3-level protection without relying
on an uncalibrated learned head.

**Monitoring for overestimation post-warmup:**
Log the following metrics during training:
- `q_mean`: mean Q-value over sampled batch
- `bellman_target_mean`: mean Bellman target over sampled batch
- `cert_score_mean`: mean certainty score used in target
- `td_error_mean`: mean |Q - target|

If `q_mean` diverges significantly above `bellman_target_mean` as `α → 0` (learned
certainty takes over), that signals the certainty head is not providing enough
pessimism. Response: add a second Q-head to share the backbone (one-function addition,
no architecture rework needed since backbone is already shared-backbone-ready).

---

## D4 — TrajectoryEncoder Token Structure

**Decision:** Hybrid — joint temporal tokens (15) + per-agent current tokens (20) = 35 kv positions.

```
Temporal branch (joint per step):
  pos[t], vel[t] → concat → reshape(B, n_agents*4) → Dense(d_input)
  traj_kv: (B, 15, d_input)

Per-agent branch (current state):
  concat(obs_i, actions_i) → Dense(d_input)   [Flax broadcasts over agent dim]
  agent_kv: (B, n_agents, d_input)

Combined:
  kv = LayerNorm(concat([traj_kv, agent_kv], axis=1))   # (B, 35, d_input)

Per-agent features for Option 3 (auxiliary output, ignored in Option 2):
  per_agent_feat = Dense(d_model)(concat(obs_i, actions_i))  # (B, n_agents, d_model)
```

**Alternatives considered:**
- A: Joint temporal only (16 tokens). Current state compressed as one flat vector.
- B: Per-agent per step (320 tokens). Attention quadratic in S; CTM must re-learn
  timestep grouping. Rejected.

**Why hybrid over A:**
Option A's current-state token compresses 20 agents × (obs_dim + 2) = 3880 dims into
128 dims in a single linear projection — asking one Dense layer to disentangle 20
agents. The CTM has no way to attend to individual agents' current state.

With 20 per-agent current tokens, the CTM can attend directly to agent i's state
(token at index 15+i). This is the natural structure of the problem.

**Option 3 payoff:**
`per_agent_feat: (B, n_agents, d_model)` is already computed during Option 2 (just
not wired). In Option 3, changing `activated_state` initialization from the broadcast
learned scalar to `per_agent_feat` is one line. The pathway is built; it just isn't
connected yet. Without this, Option 3 would require a new feature computation module.

**35 tokens is small.** Attention cost is negligible. The only cost is ~10 extra lines
in `TrajectoryEncoder`.

---

## D5 — Velocity Normalization

**Decision:** Normalize finite-difference velocities by `vel_max = 0.8`.

```python
vel[0] = zeros                                  # no predecessor at t=0
vel[t] = (pos[t] - pos[t-1]) / vel_max          # ≈ [-1, 1]
pos_normalized[t] = pos[t] / (arena_size / 2)   # ≈ [-1, 1]
```

**Alternatives considered:**
- A: Raw finite difference (positions and velocities both in arena units).
- B: Divide by `dt` (gives physical velocity, but multiplies raw diff by 20 — worse).

**Why `vel_max`:**
`observations.py` normalizes velocities by `vel_max=0.8`. The trajectory encoder
should be consistent with the observation normalization scale that the actor networks
have been trained on. Both features then live in approximately `[-1, 1]`, matching
the LayerNorm applied to `kv` afterward.

---

## D6 — Buffer Memory Cost

**Decision:** Store trajectories as `float16`, cast to `float32` in the CTMCritic
forward pass. Add separate `trajectory_dtype=jnp.float16` parameter to `ReplayBuffer`.

**Cost:** `240,000 × 15 × 20 × 2 × 2 bytes = 288 MB` (vs 576 MB at float32).

**Why float16 is safe:**
Position values in `[-2.5, 2.5]`. float16 precision at this range is ~0.001 —
more than sufficient for position history. The cast to float32 before entering
the network is a single `jnp.astype(jnp.float32)` call, zero compute overhead on GPU.

The existing `dtype` parameter on `ReplayBuffer` controls all other fields (float32).
`trajectory_dtype` is a separate parameter affecting only the two trajectory fields.

---

## D7 — Trajectory Reordering

**Decision:** Reorder at store time, inside `collect_transition` in `maddpg_wrapper.py`.

```python
def _reorder_trajectory(raw_traj, traj_idx, traj_len):
    indices = (jnp.arange(traj_len) + traj_idx + 1) % traj_len
    return raw_traj[indices]   # chronological: oldest first, newest last
```

**Alternatives considered:**
- B: Store raw circular buffer + `traj_idx`, reorder inside CTMCritic forward pass.

**Why store time:**
Each transition is stored once and sampled `updates_per_step=30` times on average.
Reordering at store time pays the gather cost once; reordering at sample time pays
it 30× per transition. CTMCritic stays clean — receives chronological input with no
circular buffer awareness.

**Unit test required:** Insert 15 known positions into the circular buffer at varying
`traj_idx` offsets, reorder, verify `result[-1] == last_written_position`. Off-by-one
here silently corrupts the entire temporal signal.

---

## D8 — Decay Parameter Clamping

**Decision:** Clamp `decay_params_action` and `decay_params_out` to `[0, 15]` in
two places:

1. **At the start of `CTMCritic.__call__`** (defensive, mirrors existing CTM):
   ```python
   decay_params_action = jnp.clip(self.decay_params_action, 0.0, 15.0)
   decay_params_out    = jnp.clip(self.decay_params_out,    0.0, 15.0)
   ```

2. **After `optax.apply_updates` in `update_critic_ctm`**:
   ```python
   new_params = new_params.copy({
       'decay_params_action': jnp.clip(new_params['decay_params_action'], 0.0, 15.0),
       'decay_params_out':    jnp.clip(new_params['decay_params_out'],    0.0, 15.0),
   })
   ```

**Why two places:**
The forward-pass clamp catches any parameter that escapes (e.g. during target network
initialization or hard update). The post-update clamp is the primary enforcement.
`r = exp(-decay_param)` — if `decay_param` goes negative, `r > 1` and the EMA grows
instead of decaying, causing immediate training instability.

---

## D9 — Actor Loss Signature

**Decision:** New `compute_actor_loss_ctm` function. Replace `global_obs` (flat vector)
with `(trajectory, all_obs)`:

```python
def compute_actor_loss_ctm(
    actor, actor_params, ctm_critic, critic_params,
    trajectory,                  # (batch, traj_len, n_agents, 2)
    all_obs,                     # (batch, n_agents, obs_dim)   ← was global_obs
    agent_obs,                   # (batch, obs_dim)
    all_actions_except_agent, agent_action_idx, action_dim,
    action_prior=None, prior_weight=0.0,
):
    policy_action  = actor(agent_obs)
    all_actions    = insert(policy_action, all_actions_except_agent, agent_action_idx)
    all_actions_   = all_actions.reshape(batch, n_agents, action_dim)

    q_vals, certs  = ctm_critic(trajectory, all_obs, all_actions_)
    tick_certain   = argmax(certs[:, 1, :], axis=-1)
    q_at_cert      = take_along_axis(q_vals[:, 0, :], tick_certain[:, None]).squeeze(-1)
    loss           = -mean(q_at_cert)
```

`all_obs` is `batch.obs` — shape `(batch, n_agents, obs_dim)`, already in the sampled
batch. No new data required. The reshape of `all_actions` from flat to per-agent is a
view operation (zero cost), required because `CTMCritic` takes per-agent structured input.

---

## D10 — CTM Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `iterations` | 8 | 15 ticks is slow to JIT-unroll. 8 gives enough ticks for `tick_best` / `tick_certain` to diverge meaningfully. Sweep later. |
| `d_model` | 64 | Arbitrary in Option 2 (not tied to n_agents). Smaller = faster, more stable early. Option 3 fixes this to 20. |
| `d_input` | 128 | Trajectory token and attention embedding dimension. Matches MNIST baseline; 35 tokens at this size is fine. |
| `memory_length` | 5 | Per-neuron activation history window. 5 steps of internal history is sufficient for Q regression. |
| `n_synch_out` | 8 | `synch_repr_size = 8*9//2 = 36`. Output projection `Dense(1)(36-dim)` is appropriately sized for scalar Q. |
| `n_synch_action` | 8 | Same reasoning. Attention query `Dense(d_input)(36-dim)` — reasonable. |
| `heads` | 2 | `d_input=128`, `head_dim=64`. Standard. |
| `memory_hidden_dims` | 8 | Unchanged from MNIST baseline. |

**Option 3 compatibility check:** `d_model` will change to 20. `n_synch_out=8` and
`n_synch_action=8` must be ≤ `d_model=20` — satisfied. `synch_repr_size=36` unchanged.
No hyperparameter changes needed when moving to Option 3.

---

## Implementation Sequence

```
Phase 1 — Data pipeline
  buffers.py         add trajectory/next_trajectory fields + trajectory_dtype param
  maddpg_wrapper.py  _reorder_trajectory() + extract trajectory in collect_transition
  assembly_cfg.py    pass traj_len to wrapper

Phase 2 — Model
  networks.py        TrajectoryEncoder (hybrid 35-token) + CTMCritic

Phase 3 — Loss functions
  agents.py          compute_ctm_critic_loss() + compute_actor_loss_ctm()

Phase 4 — Wiring
  maddpg.py          use_ctm_critic flag + CTM config fields + conditional update path
  assembly_cfg.py    CTM hyperparameter fields

Phase 5 — Validate
  - Q-values finite and bounded
  - tick_best ≠ tick_certain on some samples (mechanism is active)
  - cert_score varies across batch (not saturated)
  - No NaN in synch EMA
  - Overestimation monitoring metrics logging from step 1
```
