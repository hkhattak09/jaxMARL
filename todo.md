# TODO: Reward Function & Prior Policy Fix

## Summary
Fixed two components in the JAX implementation to match the original C++ MARL code:
1. **Reward function** - exploration/uniformity calculation in `jax_cus_gym/rewards.py`
2. **Prior policy** - repulsion strength in `jax_cus_gym/assembly_env.py`

---

## Change 1: Reward Function Fix

### 1. Added `rho_cos_dec` Function (Cosine Decay Weighting)
**Location:** `jax_cus_gym/rewards.py` lines ~60-93

The C++ code uses a smooth cosine-decay weighting function for distance-based weighting:

```python
def rho_cos_dec(z, r, delta=0.5):
    # z < delta * r       -> 1.0 (close range, full weight)
    # delta * r <= z < r  -> cosine decay (smooth transition)
    # z >= r              -> 0.0 (out of range, no weight)
```

Formula for transition zone:
$$\rho(z) = \frac{1}{2}\left(1 + \cos\left(\frac{\pi(z/r - \delta)}{1-\delta}\right)\right)$$

### 2. Fixed `compute_exploration_reward` Function
**Location:** `jax_cus_gym/rewards.py` lines ~170-240

#### OLD (Incorrect) Logic:
```python
# Computed centroid of ABSOLUTE grid positions
centroids = Σ(weights × grid_centers) / Σ(weights)
# Checked distance from agent to centroid
dist_to_centroid = ||positions - centroids||
is_exploring = dist_to_centroid < 0.05  # WRONG!
```

#### NEW (Correct) Logic:
```python
# Compute RELATIVE positions from agent to each grid
rel_pos = grid_centers - positions  # (n_agents, n_grid, 2)

# Weight by cosine-decay based on distance
psi_weights = rho_cos_dec(distances, d_sen, delta)

# Compute weighted centroid of RELATIVE positions
v_exp = Σ(ψ × rel_pos) / Σ(ψ)  # (n_agents, 2)

# Check if norm is small (agent is centered among grids)
is_uniform = ||v_exp|| < 0.05  # CORRECT!
```

### 3. Added `cosine_decay_delta` Parameter
**Location:** `RewardParams` dataclass

```python
cosine_decay_delta: float = 0.5  # Delta for cosine decay weighting function
```

---

## Why This Matters

The **key difference** is what we're measuring:

| Aspect | Old (Wrong) | New (Correct) |
|--------|-------------|---------------|
| Centroid of | Absolute grid positions | Relative vectors to grids |
| Measures | Distance from agent to grid centroid | Whether agent is centered among sensed grids |
| Intuition | "Am I near the center of all grids?" | "Are grids evenly distributed around me?" |

The correct interpretation: An agent is "uniform" when the **weighted average of relative vectors** pointing to nearby grids is approximately zero. This means grids are distributed evenly around the agent (not all on one side).

---

## Tests to Run

```bash
# Run reward tests
cd /Users/hassan/repos/jaxMARL
python -m pytest jax_cus_gym/tests/test_rewards.py -v

# Run all jax_cus_gym tests
python -m pytest jax_cus_gym/tests/ -v

# Quick sanity check - test the new function directly
python -c "
import jax.numpy as jnp
from jax_cus_gym.rewards import rho_cos_dec, compute_exploration_reward

# Test rho_cos_dec
z = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
r = 1.0
delta = 0.5
weights = rho_cos_dec(z, r, delta)
print('rho_cos_dec test:')
print(f'  distances: {z}')
print(f'  weights:   {weights}')
print(f'  Expected:  [1.0, 1.0, 1.0, ~0.5, 0.0, 0.0]')
print()

# Test exploration reward - agent at center of 4 symmetric grids
positions = jnp.array([[0.0, 0.0]])  # 1 agent at origin
grid_centers = jnp.array([
    [0.5, 0.0],   # right
    [-0.5, 0.0],  # left  
    [0.0, 0.5],   # up
    [0.0, -0.5],  # down
])
in_target = jnp.array([True])
neighbor_indices = jnp.zeros((1, 1), dtype=jnp.int32)

reward = compute_exploration_reward(
    positions, grid_centers, in_target, neighbor_indices,
    collision_threshold=0.15, exploration_threshold=0.05,
    d_sen=3.0, cosine_decay_delta=0.5
)
print('Exploration reward test (symmetric grids):')
print(f'  Agent at origin with 4 symmetric grids')
print(f'  Reward: {reward}')
print(f'  Expected: 1.0 (agent is centered, v_exp ≈ 0)')
"
```

---

## Expected Behavior After Reward Fix

1. **Agent centered among grids** → `||v_exp|| ≈ 0` → Reward = 1.0
2. **Agent off-center (grids mostly on one side)** → `||v_exp|| > 0.05` → Reward = 0.0
3. **Cosine decay** ensures smooth weighting (closer grids matter more)
4. **JAX-compatible** - fully vectorized, no Python loops, JIT-friendly

---

## Change 2: Prior Policy Fix

### What Changed
**Location:** `jax_cus_gym/assembly_env.py` function `compute_prior_policy`

Changed `repulsion_strength` default from **1.0** to **3.0** to match C++ MARL.

#### OLD:
```python
def compute_prior_policy(
    ...
    repulsion_strength: float = 1.0,  # WRONG - too weak
    ...
)
```

#### NEW:
```python
def compute_prior_policy(
    ...
    repulsion_strength: float = 3.0,  # Matches C++ MARL default
    ...
)
```

### Why This Matters
The prior policy implements a Reynolds-style flocking model with 3 forces:

| Force | Strength | Purpose |
|-------|----------|---------|
| **Attraction** | 2.0 | Move toward target grid cells |
| **Repulsion** | **3.0** | Avoid collisions with nearby agents |
| **Synchronization** | 2.0 | Align velocity with neighbors |

With `repulsion_strength = 1.0`, agents had **weak collision avoidance** compared to the original C++ code. This could lead to:
- More collisions during training
- Different emergent behaviors
- Inconsistent results between JAX and C++ implementations

### Prior Policy Formula (Matches C++)

**Attraction:**
$$\vec{F}_{\text{attract}} = 2.0 \cdot \frac{\vec{p}_{\text{target}} - \vec{p}_{\text{agent}}}{||\vec{p}_{\text{target}} - \vec{p}_{\text{agent}}||}$$

**Repulsion** (for each neighbor within `r_avoid`):
$$\vec{F}_{\text{repel}} = 3.0 \cdot \left(\frac{r_{\text{avoid}}}{d} - 1\right) \cdot \hat{n}_{\text{away}}$$

**Synchronization:**
$$\vec{F}_{\text{sync}} = 2.0 \cdot (\vec{v}_{\text{avg}} - \vec{v}_{\text{agent}})$$

**Total:**
$$\vec{F}_{\text{total}} = \text{clip}\left(\vec{F}_{\text{attract}} + \vec{F}_{\text{repel}} + \vec{F}_{\text{sync}}, -1, 1\right)$$

---

## Files Modified

- `jax_cus_gym/rewards.py`
  - Added `rho_cos_dec()` function
  - Added `cosine_decay_delta` to `RewardParams`
  - Rewrote `compute_exploration_reward()` to match C++ logic

- `jax_cus_gym/assembly_env.py`
  - Changed `repulsion_strength` default from 1.0 to 3.0

---

## Tests to Run

```bash
# Run all tests
cd /Users/hassan/repos/jaxMARL
python -m pytest jax_cus_gym/tests/ -v

# Specific tests for the changes
python -m pytest jax_cus_gym/tests/test_rewards.py -v
python -m pytest jax_cus_gym/tests/test_assembly_env.py -v

# Run just the new C++ MARL matching tests
python -m pytest jax_cus_gym/tests/test_rewards.py -v -k "rho_cos_dec or exploration_reward"
python -m pytest jax_cus_gym/tests/test_assembly_env.py -v -k "repulsion_strength or in_target_behavior"
```

### New Tests Added

#### In `test_rewards.py`:
- `test_rho_cos_dec_basic` - Tests cosine decay function at key distance values
- `test_rho_cos_dec_different_params` - Tests with different r and delta values
- `test_rho_cos_dec_vectorized` - Tests vectorization with 2D arrays
- `test_exploration_reward_symmetric_grids` - Agent centered among symmetric grids → reward
- `test_exploration_reward_asymmetric_grids` - Agent off-center → no reward
- `test_exploration_reward_not_in_target` - Not in target → no reward
- `test_exploration_reward_multiple_agents` - Tests with multiple agents

#### In `test_assembly_env.py`:
- `test_prior_policy_repulsion_strength` - Verifies repulsion_strength=3.0 gives strong repulsion
- `test_prior_policy_in_target_behavior` - Verifies zero attraction when in target

---

## Verification Checklist

### Reward Function
- [ ] `python -m pytest jax_cus_gym/tests/test_rewards.py -v` passes
- [ ] `rho_cos_dec` returns correct values (1.0 close, decay in middle, 0.0 far)
- [ ] Symmetric grid arrangement gives reward (v_exp ≈ 0)
- [ ] Asymmetric grid arrangement gives no reward (v_exp > 0.05)

### Prior Policy
- [ ] `python -m pytest jax_cus_gym/tests/test_assembly_env.py -v` passes
- [ ] Close agents have strong repulsion force
- [ ] Prior policy is JIT-compilable (no Python loops in hot path)
- [ ] No performance regression
- [ ] No performance regression (check JIT compilation works)
