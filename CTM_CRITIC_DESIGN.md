# CTM Critic Design: Options 2 & 3

---

## Option 2: CTM as Temporal Critic

### The Problem

The current MLP critic sees a single transition `(global_state, all_actions) → Q`. It has
no memory of what happened before this timestep. Assembly is path-dependent in ways that
matter for Q estimation:

- An agent inside a target cell but drifting away ≠ an agent approaching the same cell
- A stable cluster that has been assembled for 10 steps ≠ agents that just arrived
- A collision about to happen ≠ one that just resolved

The Bellman target `r + γ * Q(s')` handles the future. Nothing handles the past. Yet
`AssemblyState.trajectory: (traj_len=15, n_agents, 2)` already stores 15 steps of position
history for every agent — and the MLP critic ignores it entirely.

### The Idea

Replace the MLP critic with a CTM that cross-attends over the trajectory. Each trajectory
step becomes a token in the CTM's external memory (`kv`). The CTM's internal ticks then
correspond to reasoning about different moments in the recent history before converging on
a Q-value.

The mapping from the original CTM is nearly 1:1:
- Original: CNN over image patches → `kv: (B, num_patches, d_input)`
- Temporal critic: MLP over trajectory steps → `kv: (B, 15, d_input)`

Image patches and trajectory frames are both sequences the CTM cross-attends over. The
swap is architectural, not conceptual.

### Architecture

```
Inputs:
  trajectory:      (B, traj_len=15, n_agents, 2)   position history
  current_obs:     (B, n_agents, obs_dim)           current observations
  current_actions: (B, n_agents, 2)                 actions being evaluated

Step 1 — Trajectory Backbone (replaces CNN):
  For each step t in [0, traj_len):
    approx_vel[t] = (pos[t] - pos[t-1]) / dt           finite-difference velocity
    joint_state[t] = flatten(pos[t], vel[t])            (B, n_agents * 4)
    token[t]       = Dense(d_input)(joint_state[t])     (B, d_input)
  current_token    = Dense(d_input)(flatten(current_obs, current_actions))
  kv = stack([token_0, ..., token_14, current_token])   (B, 16, d_input)

Step 2 — CTM (unchanged internal mechanics):
  Query derived from neuron synchronization as normal.
  CTM cross-attends over kv at each of its internal ticks.
  tick 1–5:  builds global picture from full trajectory
  tick 6–10: focuses on the most relevant moments (entering target, collisions)
  tick 11–15: converges on stable Q estimate
  Output: q_t for each tick t,  certainty_t for each tick t

Step 3 — Output:
  q:          (B, 1, num_ticks)   Q-value at each tick
  certainties:(B, 2, num_ticks)   [entropy, 1-entropy] at each tick
  Final Q:    q at most-certain tick
```

### Loss Function

The CTM classification loss (cross-entropy) is replaced with regression (MSE against
Bellman target), but the dual-tick structure transfers directly:

```
bellman_target y = r + γ * min(Q1_target(s', a'_target), Q2_target(s', a'_target))

tick_best     = argmin_t  |q_t - y|          tick where Q estimate is closest to target
tick_certain  = argmax_t  certainty[:, 1, t] tick where model is most confident

loss = ( MSE(q[tick_best], y) + MSE(q[tick_certain], y) ) / 2
```

`tick_best` identifies where the gradient information is cleanest — that tick gets a strong
update signal. `tick_certain` identifies where the model trusts itself — that tick's output
is used for actor updates. Both terms together teach the model to be confident exactly when
its Q estimate is accurate.

### Data Pipeline Changes

The trajectory needs to reach the replay buffer. Currently the buffer stores flat
`global_state` vectors. The change is additive — trajectory becomes a separate field:

```
buffers.py  — Transition gains:
  trajectory:      (traj_len, n_agents, 2)
  next_trajectory: (traj_len, n_agents, 2)

maddpg_wrapper.py — extract state.trajectory when storing transitions

networks.py — CTMCritic replaces Critic, takes trajectory as additional input

agents.py — update_critic passes trajectory through to CTMCritic
```

`AssemblyState.trajectory` is already maintained as a circular buffer in the env — no
environment changes needed. It just needs to be extracted and stored.

**Memory cost:** `15 steps × 20 agents × 2 coords × 4 bytes × 240,000 buffer capacity ≈ 576 MB`
This is the main practical constraint. Reducible by storing float16 or reducing traj_len.

### Key Design Decisions

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Token content | positions only / positions+vel / positions+vel+grid | positions+approx_vel first |
| Token structure | joint per step (all agents) / per-agent per step | joint — preserves inter-agent context |
| Sequence length | 15+1=16 / shorter | match traj_len, tune down if slow |
| Num CTM ticks | 15 / 8 / 5 | start at 8, sweep |
| Velocity source | finite difference / store in trajectory | finite difference first |

---

## Option 3: Credit Assignment via CTM

### The Problem

MADDPG actor loss for agent i: `-∂Q(s, a₁, ..., aₙ) / ∂aᵢ`

The gradient of the **joint** Q with respect to agent i's action is noisy. Q conflates all
20 agents' contributions. Agent 7 correctly enters a target cell — its gradient signal is
diluted by the simultaneous behavior of 19 other agents. In a 20-agent system the marginal
contribution of any single agent to the joint Q is small relative to the noise.

This is the credit assignment problem: given a joint outcome, which agent deserves credit?
The MLP critic provides no answer — it outputs one number for the team.

### The Idea

Set `d_model = n_agents`. Each CTM neuron represents one agent. The CTM's synchronization
mechanism — which computes pairwise outer products of neuron activations — then directly
computes agent-agent correlations as a byproduct of its normal operation.

The CTM's internal ticks perform progressive credit attribution:
- Early ticks: build global picture of joint state
- Middle ticks: synchronization reveals which agents are correlated (co-occupying cells,
  approaching collision, independently scattered)
- Late ticks: per-neuron (per-agent) activations stabilize into individual credit estimates

Two output heads are added:
1. **Joint Q head** — as before, for Bellman learning
2. **Credit head** — one scalar per neuron (per agent), decomposing the joint Q

### Architecture

```
Inputs (reshaped as per-agent tokens — not a flat vector):
  per_agent_input: (B, n_agents, obs_dim + 2)   each agent's obs + its action

Step 1 — Per-agent feature extraction:
  Dense(d_model)(per_agent_input_i) → agent_feature_i   (B, n_agents, d_model)
  → initial activated_state for CTM neurons

Step 2 — CTM (d_model = n_agents = 20):
  tick 1–5:  cross-attention over kv builds global context
             (kv = trajectory tokens from Option 2, or current global state)
  tick 6–10: synch_out = outer(neuron_i, neuron_j) for all agent pairs
             agents that are spatially coupled → correlated neurons
             agents independently scattered → decorrelated neurons
  tick 11–15: per-neuron activations stabilize
  activated_state: (B, n_agents)   one value per neuron = one per agent

Step 3 — Output heads:
  Q_joint:  Dense(activated_state_all) → scalar Q       for Bellman
  credits:  [Dense(activated_state_i) → scalar]_i       per-agent credit  (B, n_agents)
  certainty: from Q_joint entropy as normal
```

### Why Synchronization is the Right Mechanism

The CTM computes `synch[i,j] = EMA(activation_i * activation_j)` — a running estimate of
the correlation between neuron i and neuron j. With neurons = agents:

- Agents co-occupying adjacent target cells → neurons fire together → high positive synch
- Agents on collision course → characteristic pre-collision synch pattern
- Agents independently distributed → near-zero synch (decorrelated)

The CTM learns to use these synch patterns as the basis for credit attribution. Agents with
high mutual synch share credit (their outcomes are entangled). Decorrelated agents are
attributed independently. This is the correct inductive bias for swarm tasks — no
handcrafting required.

### Credit Decomposition Loss

Credits have no ground-truth labels. The constraint that makes them meaningful is that they
must sum to the joint Q:

```
L_bellman     = MSE(Q_joint, bellman_target)
                  standard TD loss

L_consistency = MSE(sum_i(credits_i), Q_joint.detach())
                  forces credits to decompose Q (Q_joint is detached to avoid
                  the consistency loss pulling Q_joint toward the credits)

L_spread      = -entropy(softmax(abs(credits)))
                  prevents degenerate solution where one agent captures all credit

total_critic_loss = L_bellman + λ₁ * L_consistency + λ₂ * L_spread
```

`L_consistency` is load-bearing. Without it credits are unconstrained and the head learns
nothing useful. With it the CTM must explain the joint Q through per-agent contributions.

### Actor Loss Change

Currently:
```python
actor_loss_i = -mean( Q(s, actor_i(obs_i), a_{-i}) )
```

With credit assignment:
```python
# baseline: batch-mean credit for agent i (approximates expected credit)
baseline_i    = mean(credits_i)
advantage_i   = credits_i - baseline_i

actor_loss_i  = -mean(advantage_i)
```

The actor gradient now points in the direction that increases agent i's **marginal
contribution**, not the noisy joint Q. In a 20-agent system this is a qualitatively
cleaner signal — agent 7's actor only learns from agent 7's credit, not from the joint
outcome of all 20 agents simultaneously.

---

## Is Option 2 a Clean Path to Option 3?

**Yes — and deliberately so.** The two options are layered along orthogonal axes:

| | What it adds | What it changes |
|-|-------------|-----------------|
| Option 2 | Temporal depth via trajectory | Backbone, buffer, loss |
| Option 3 | Structural decomposition via neuron=agent | d_model, output heads, actor loss |

Option 2 establishes the CTM as the critic and solves all the infrastructure work:
- Trajectory in the buffer
- CTM backbone wired into `update_critic`
- Dual-tick loss working
- Training stable

Option 3 then adds two things on top of a working CTM critic:
1. Set `d_model = n_agents` (one line in model config)
2. Add the credit head and `L_consistency` loss (one new output Dense + one loss term)
3. Change actor loss to use `advantage_i` instead of `-Q`

Nothing from Option 2 needs to be undone. The temporal backbone still provides `kv`. The
CTM internal mechanics are unchanged. The dual-tick loss still applies to `Q_joint`. Credit
assignment is purely additive.

The one architectural decision that needs to be made at Option 2 time, to avoid rework:

> Structure the per-agent input as tokens `(B, n_agents, per_agent_dim)` rather than a
> flat concatenated vector. Option 2 works fine either way, but Option 3 requires the
> per-agent token structure to initialize per-neuron activated states. Building this in
> during Option 2 means Option 3 is a clean add-on with no refactoring.

### Implementation Sequence

```
Option 2
├── buffers.py       add trajectory / next_trajectory fields to Transition
├── maddpg_wrapper   extract state.trajectory when storing transitions
├── jaxCTM/          add TrajectoryEncoder (backbone replacement)
├── networks.py      CTMCritic: TrajectoryEncoder + CTM + dual-tick MSE loss
├── agents.py        update_critic uses CTMCritic, pass trajectory through
└── validate         temporal Q produces better Bellman estimates than MLP critic

Option 3  (on top of working Option 2)
├── model config     d_model = n_agents (was arbitrary, now fixed = 20)
├── networks.py      add credit head to CTMCritic
├── loss             add L_consistency + L_spread terms to critic loss
├── agents.py        actor loss uses advantage_i from credit head
└── validate         actor gradient variance decreases vs baseline
```

No step in Option 3 removes or replaces anything from Option 2. Every addition is
orthogonal. The path is clean.

---

## Combined Architecture (Target State)

```
Trajectory (B, 15, n_agents, 2)
  + current (obs, actions) per agent
          ↓
TrajectoryEncoder: per-step joint MLP → kv (B, 16, d_input)
          ↓
CTM  [d_model = n_agents = 20]
  neurons ↔ agents
  cross-attends over 16-step trajectory at each tick
  synchronization = pairwise agent-agent correlations evolving over ticks
          ↓
Output heads:
  Q_joint   → Bellman loss (MSE at tick_best + tick_certain)
  credits_i → consistency loss (Σ credits ≈ Q_joint)
  certainty → uncertainty signal
          ↓
Actor update uses advantage_i = credits_i - baseline_i
```

One model. Temporal memory + structural decomposition + uncertainty quantification.
