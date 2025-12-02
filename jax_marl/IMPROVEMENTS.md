# MADDPG Convergence Improvements

This document tracks planned and implemented improvements to help the MADDPG algorithm converge faster.

---

## ✅ Implemented

### 1. Observation & Reward Normalization (Dec 2, 2025)

**Rationale:** Neural networks train better on normalized inputs. Scale mismatch between features can slow convergence.

**Changes Made:**
| Component | File | Change |
|-----------|------|--------|
| Fixed Obs Normalization | `observations.py` | Fixed `l_max=2.5` to match `arena_size/2` |
| Layer Norm in Networks | `maddpg.py` | Changed `use_layer_norm=True` (was False) |
| Reward Clipping | `rewards.py` | Added `reward_clip_min=-10.0`, `reward_clip_max=10.0` |

**Verification:**
- All observations now in range `[-1, 1]` (100% within `[-3, 3]`)
- Reward clipping working correctly (extreme ±100 rewards clipped to ±10)
- Layer norm enabled by default
- Test file: `jax_cus_gym/tests/test_normalization.py`

---

### 2. Twin Delayed DDPG (TD3) Enhancements (Dec 2, 2025)

**Rationale:** DDPG overestimates Q-values, leading to poor policies. TD3 fixes this systematically.

**Changes Made:**
| Component | File | Change |
|-----------|------|--------|
| Twin Critics | `networks.py` | `CriticTwin` now used when `use_td3=True` |
| TD3 Critic Loss | `agents.py` | Added `compute_critic_loss_td3()` using min(Q1, Q2) for targets |
| TD3 Actor Loss | `agents.py` | Added `compute_actor_loss_td3()` using only Q1 for policy gradient |
| Target Smoothing | `agents.py` | Added `select_target_action_with_smoothing()` |
| Delayed Updates | `maddpg.py` | Actor updates every `policy_delay` critic updates |
| Config Options | `maddpg.py`, `assembly_cfg.py` | Added `use_td3`, `policy_delay`, `target_noise`, `target_noise_clip` |

**Configuration (in `assembly_cfg.py`):**
```python
use_td3: bool = True           # Enable TD3 (default: True)
policy_delay: int = 2          # Update actor every N critic updates
target_noise: float = 0.2      # Stddev of noise for target smoothing
target_noise_clip: float = 0.5 # Clip range for target noise
```

**Verification:**
- All 20 MADDPG tests pass
- JIT-compiled update function supports TD3
- Backwards compatible: set `use_td3=False` for standard DDPG

---

## 🔄 In Progress

*None currently*

---

## 📋 Planned Improvements

### 3. Prioritized Experience Replay (PER)

**Rationale:** Sample transitions based on TD-error priority. Focuses learning on difficult/surprising experiences. Can provide 2-3x speedup in sample efficiency.

**Components:**
- Priority based on TD-error magnitude
- Importance sampling weights to correct bias
- Sum-tree data structure for efficient sampling

---

### 4. Episode-Based Noise Decay

**Rationale:** Current step-based decay is functionally equivalent to episode-based decay since episode lengths are fixed (`max_steps=200`). Episode-based is more intuitive and matches the PyTorch MARL implementation.

**Current Implementation:**
```python
# Step-based (indirect episode counting)
total_steps = training_state.total_steps + num_steps * n_envs
decay_progress = min(1.0, total_steps / noise_decay_steps)
new_noise = initial - decay_progress * (initial - final)
```

**Proposed Change:**
```python
# Episode-based (direct)
decay_progress = min(1.0, episode / noise_decay_episodes)
new_noise = initial - decay_progress * (initial - final)
```

**Why Current Works Fine:**
- With fixed `max_steps=200` and `n_envs=8`, each episode = 1,600 steps
- Step-based and episode-based produce identical decay schedules
- Step-based is future-proof if early termination is ever added

**Important Note:** `noise_decay_steps` must account for `n_parallel_envs`:
```python
# Formula: noise_decay_steps = desired_episodes × max_steps × n_parallel_envs
# Example: 1333 episodes × 200 steps × 4 envs = 1,066,400 steps
noise_decay_steps: int = 1066400
```

**Conclusion:** Keep step-based, but tune `noise_decay_steps` based on parallel envs.

---

### 5. Warmup Strategy Comparison

**Current JAX Implementation:**
```python
# Explicit step-based warmup
if buffer_size >= warmup_steps:  # default: 5000
    # Start training
```

**PyTorch MARL Implementation:**
```python
# Implicit batch-size warmup
if len(agent_buffer) >= batch_size:  # default: 2048
    # Start training
```

**Comparison:**
| Aspect | PyTorch | JAX |
|--------|---------|-----|
| Threshold | 2,048 transitions | 5,000 transitions |
| Episodes to reach (single env) | ~11 episodes | N/A |
| Episodes to reach (4 parallel envs) | N/A | ~7 episodes |

**Analysis:**
- JAX collects MORE data before training (5000 vs 2048 transitions)
- But reaches threshold in FEWER episodes due to parallel envs
- More warmup data = more diverse initial training samples
- Current JAX approach is slightly more conservative (good)

**Conclusion:** Current implementation is fine. No changes needed.

---

### 6. Learning Rate Scheduling

**Rationale:** Early training with high LR can cause instability; late training benefits from fine-tuning with lower LR.

**Options:**
- Linear warmup
- Cosine annealing
- Reduce on plateau

---

### 7. N-Step Returns

**Rationale:** Propagates reward signal faster through the value function, reducing the number of updates needed.

**Current:** 1-step TD: `r + γ * Q(s', a')`

**Improvement:** N-step: `r_0 + γr_1 + γ²r_2 + ... + γⁿQ(s_n, a_n)`

---

### 8. Gradient Improvements

**Rationale:** More stable gradient flow in multi-agent setting.

**Options:**
- Per-parameter gradient clipping (instead of global norm)
- Gradient penalty for critic stability
- Spectral normalization in critic

---

### 7. Entropy Bonus (SAC-style)

**Rationale:** Encourages exploration throughout training, not just early. Leads to more robust policies.

**Approach:** Add entropy term to actor objective: `maximize Q(s,a) + α * H(π(·|s))`

---

### 8. Replay Buffer Improvements

**Options:**
- **Combined Experience Replay (CER):** Always include most recent transition in batch
- **Episode-based sampling:** Sample full episodes for better temporal consistency

---

### 9. Target Network Tuning

**Current:** Soft update every step with `tau=0.01`

**Options:**
- Less frequent hard updates
- Adaptive tau (start large, decrease over time)

---

### 10. Batch Size Warmup

**Rationale:** Early training has high variance; smaller batches prevent overfitting to initial experiences.

**Approach:** Start with smaller batches, gradually increase as buffer fills.

---

## 📊 Priority Order

Based on expected impact and implementation complexity:

1. ~~**Observation/Reward Normalization**~~ ✅ Implemented
2. ~~**TD3 Enhancements**~~ ✅ Implemented
3. **Prioritized Experience Replay** - High impact, moderate complexity
4. **Learning Rate Scheduling** - Easy to add with optax
5. **N-Step Returns** - Moderate impact, requires buffer changes
6. Others as needed based on training diagnostics

---

## 🔧 Configuration Reference

Current defaults in `assembly_cfg.py`:
```python
# Algorithm (MADDPG)
hidden_dim: int = 256
lr_actor: float = 1e-4
lr_critic: float = 1e-3
gamma: float = 0.95
tau: float = 0.01
buffer_size: int = 50000
batch_size: int = 2048
warmup_steps: int = 5000
noise_scale_initial: float = 0.9
noise_scale_final: float = 0.5      # Matches PyTorch MARL floor
noise_decay_steps: int = 100000
updates_per_step: int = 30

# TD3 Enhancements
use_td3: bool = True
policy_delay: int = 2
target_noise: float = 0.2
target_noise_clip: float = 0.5
```

Current defaults in `observations.py`:
```python
normalize_obs: bool = True
l_max: float = 2.5  # Matches arena_size/2
vel_max: float = 0.8
```
