# Continuous Thought Machine (CTM) — JAX/Flax Implementation

This document describes the architecture, data flow, and component interactions of the
Continuous Thought Machine (CTM) implemented in JAX/Flax. It is written for an LLM
that needs to understand the system in order to use, extend, or port it.

---

## High-Level Concept

The CTM models **thinking as a temporal process**. Rather than producing a single
forward-pass output, the model iterates through a fixed number of internal "ticks"
(default: 15), refining its prediction at each tick. The key ideas are:

1. **Internal recurrence** — computation unfolds over ticks, not over sequence positions.
2. **Neuron-level models (NLMs)** — each neuron processes its own activation history
   independently, giving every neuron a distinct temporal memory.
3. **Synchronization** — pairwise outer products of selected neurons produce a
   compact relational representation that drives attention and prediction.
4. **Certainty-based learning** — the loss encourages the model to be confident (low
   entropy) at the same tick where it is most accurate.

---

## Module Overview

```
jaxCTM/
├── layers.py      # SuperLinear (neuron-level linear transforms)
├── model.py       # ContinuousThoughtMachine (full CTM)
├── loss.py        # get_loss, calculate_accuracy
├── data.py        # prepare_data, torch_to_jax
├── train.py       # TrainState, train_step, eval_step, train loop
├── visualize.py   # make_gif
└── __init__.py    # Public API
```

---

## Architecture Deep-Dive

### 1. Input Processing — CNN Backbone

```
Input: (B, H, W, C)   [channels-last, e.g. (B, 28, 28, 1) for MNIST]
  ↓
Conv2D(features=d_input, kernel=3×3) → BatchNorm → ReLU → MaxPool(2×2)
  ↓
Conv2D(features=d_input, kernel=3×3) → BatchNorm → ReLU → MaxPool(2×2)
  ↓
Reshape: (B, num_spatial_positions, d_input)   [e.g. (B, 49, 128)]
  ↓
Dense(d_input) → LayerNorm
Output: kv  (B, S, d_input)   — key-value memory for attention
```

This is computed **once per forward pass** before the recurrent loop begins.
`kv` is the external memory the CTM reads via cross-attention at every tick.

---

### 2. State Initialization

Two learned parameters seed the recurrent state:

| Parameter | Shape | Purpose |
|-----------|-------|---------|
| `start_activated_state` | `(d_model,)` | Initial neuron activations |
| `start_trace` | `(d_model, memory_length)` | Initial activation history window |

At the start of a forward pass both are broadcast to batch size:
```
activated_state : (B, d_model)
state_trace     : (B, d_model, memory_length)
```

---

### 3. Synchronization

Synchronization converts neuron activations into a compact relational vector via
pairwise outer products, retaining only the upper-triangular entries.

```python
# For a subset of n neurons selected from activated_state:
outer = selected[:, :, None] * selected[:, None, :]   # (B, n, n)
pairwise = outer[:, triu_row, triu_col]                # (B, n*(n+1)//2)
```

Two synchronization streams exist:

| Stream | Neurons used | Size | Used for |
|--------|-------------|------|---------|
| `synch_action` | last `n_synch_action` neurons | `n_synch_action*(n_synch_action+1)//2` | Attention query |
| `synch_out` | first `n_synch_out` neurons | `n_synch_out*(n_synch_out+1)//2` | Output prediction |

Both streams use **exponential moving average decay** controlled by learned scalar
parameters `decay_params_action` and `decay_params_out` (one per upper-triangular entry,
clamped to `[0, 15]`):

```python
r = exp(-decay_param)          # decay rate in [exp(-15), 1]
decay_alpha = r * decay_alpha + pairwise_product
decay_beta  = r * decay_beta  + 1.0
synchronisation = decay_alpha / sqrt(decay_beta)
```

This running normalization keeps the synchronization values well-scaled regardless
of how many ticks have elapsed.

---

### 4. Per-Tick Recurrent Loop (15 iterations by default)

Each tick executes the following steps in order:

#### Step 1 — Action Synchronization
```
activated_state → select last n_synch_action neurons
               → outer product → upper-triangular → EMA decay
               → synch_action : (B, synch_repr_size_action)
```

#### Step 2 — Attention Query + Cross-Attention
```
synch_action → Dense(d_input) → q : (B, 1, d_input)

Multi-head cross-attention (manual einsum, H heads):
  Q = q_proj(q)    K = k_proj(kv)    V = v_proj(kv)
  reshape to (B, H, seq, head_dim)
  scores = QK^T / sqrt(head_dim)     softmax → weights
  context = weights @ V
  context → out_proj → attn_out : (B, d_input)
```

The attention reads from the CNN's spatial feature map `kv`, allowing the model
to focus on different image regions at each tick.

#### Step 3 — Synapses (state update)
```
[attn_out | activated_state] → Dropout → Dense(d_model * 2)
                             → GLU(split in half, gate with sigmoid)
                             → LayerNorm
                             → state : (B, d_model)
```

GLU: `output = A * sigmoid(B)` where `[A, B] = split(state, 2, axis=-1)`.
This gates information flow, allowing selective memory updates.

#### Step 4 — Trace (sliding window memory)
```
state_trace : (B, d_model, memory_length)
new_trace   = concat([state_trace[:, :, 1:], state[:, :, None]], axis=-1)
            → slides window, dropping oldest entry, appending current state
```

Each neuron now carries a length-`memory_length` history of its own activations.

#### Step 5 — Neuron-Level Model (trace_processor)
```
state_trace : (B, d_model, memory_length)
  ↓
SuperLinear(out_dims=2*memory_hidden_dims, N=d_model)  [N independent transforms]
  ↓  GLU
SuperLinear(out_dims=2, N=d_model)
  ↓  GLU → squeeze
activated_state : (B, d_model)
```

`SuperLinear` applies **one independent linear transform per neuron** using a shared
weight tensor and `einsum('BDM,MHD->BDH', x, w)`. This gives each neuron its own
learned function over its activation history — hence "neuron-level model".

#### Step 6 — Output Synchronization
```
activated_state → select first n_synch_out neurons
               → outer product → upper-triangular → EMA decay
               → synch_out : (B, synch_repr_size_out)
```

#### Step 7 — Prediction + Certainty
```
synch_out → Dense(out_dims) → current_prediction : (B, out_dims)
current_prediction → normalized_entropy → certainty : (B, 2)
  certainty[:, 0] = normalized_entropy       (uncertainty)
  certainty[:, 1] = 1 - normalized_entropy   (certainty)
```

---

### 5. Output Shapes

After all iterations are stacked:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `predictions` | `(B, out_dims, T)` | Logits at each tick |
| `certainties` | `(B, 2, T)` | [entropy, 1-entropy] at each tick |
| `synch_out` | `(B, synch_repr_size_out)` | Final tick synchronization |

When `track=True`, additional per-tick diagnostics are returned (see `model.py`).

---

### 6. Loss Function

The loss averages **two terms** to encourage both accuracy and confidence:

```python
losses   : (B, T)  cross-entropy at every tick

tick_1   = argmin(losses, axis=T)          # tick where loss is lowest
tick_2   = argmax(certainties[:, 1, :])    # tick where model is most certain

loss = mean(losses[:, tick_1] + losses[:, tick_2]) / 2
```

This dual objective means the model is penalized if it is confident at a tick where
it is wrong, and also penalized if its best tick is not its most confident tick.

---

### 7. Decay Parameter Clamping

`decay_params_action` and `decay_params_out` are clamped to `[0, 15]` after every
gradient update. Since `r = exp(-decay_param)`, this keeps `r` in `[exp(-15), 1]`,
preventing the EMA from either never decaying (r=1) or decaying instantly (r→0).

---

## Key Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `iterations` | 15 | Number of internal ticks |
| `d_model` | 128 | Number of neurons |
| `d_input` | 128 | CNN feature / attention dimension |
| `memory_length` | 10 | Activation history window per neuron |
| `heads` | 2 | Attention heads (`d_input` must be divisible) |
| `n_synch_out` | 16 | Neurons used for output synchronization |
| `n_synch_action` | 16 | Neurons used for action synchronization |
| `memory_hidden_dims` | 8 | Hidden dim in trace_processor SuperLinear |
| `out_dims` | 10 | Output classes (10 for MNIST) |
| `dropout_rate` | 0.0 | Dropout in synapses and SuperLinear |

Total parameters on MNIST configuration: **~1,248,778**

---

## Data Format

JAX convolutions use **channels-last** layout. All image tensors must be
`(B, H, W, C)`. The `torch_to_jax()` utility converts PyTorch's channels-first
`(B, C, H, W)` to this format.

```python
images = jnp.array(inputs.numpy()).transpose(0, 2, 3, 1)
```

---

## Training Infrastructure

```python
class TrainState(train_state.TrainState):
    batch_stats: any   # carries BatchNorm running statistics
```

- `train_step` is `@jax.jit` compiled. It runs the full CTM forward pass
  (with the internal for-loop unrolled into a single XLA graph), computes
  gradients, applies the optimizer, then clamps decay params.
- `eval_step` is `@jax.jit` compiled with `deterministic=True` (BatchNorm
  uses running averages, dropout is disabled).
- Dropout randomness is derived via `jrandom.fold_in(PRNGKey(0), state.step)`
  — zero overhead, deterministic per step.

---

## Visualization

`make_gif()` produces a per-tick animation showing:
- Input image
- Attention heatmap (spatial attention over CNN feature map)
- Class probability bar chart
- Certainty curve over all ticks
- 16 neuron traces (pre-activation in gray, post-activation in red/blue)

---

## Minimal Usage Example

```python
import jax
import jax.numpy as jnp
from jax import random as jrandom
import optax
from flax.training import train_state

from jaxCTM import ContinuousThoughtMachine, TrainState, train_step, eval_step

# Build model
model = ContinuousThoughtMachine(
    iterations=15, d_model=128, d_input=128, memory_length=10,
    heads=2, n_synch_out=16, n_synch_action=16,
    memory_hidden_dims=8, out_dims=10,
)

# Initialize
key = jrandom.PRNGKey(42)
dummy = jnp.ones((1, 28, 28, 1))
variables = model.init(key, dummy, deterministic=True)

# Create training state
optimizer = optax.adamw(learning_rate=1e-4, eps=1e-8)
state = TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=optimizer,
    batch_stats=variables['batch_stats'],
)

# Single training step
images = jnp.ones((256, 28, 28, 1))   # (B, H, W, C) channels-last
labels = jnp.zeros(256, dtype=jnp.int32)
state, loss, accuracy, where_most_certain = train_step(state, images, labels)
```

---

## Porting Notes

When adapting this architecture to a new task:

1. **Replace the CNN backbone** in `compute_features()` with any encoder that
   produces `(B, S, d_input)` key-value pairs.
2. **Change `out_dims`** to match your output space.
3. **Adjust `n_synch_out` / `n_synch_action`** — larger values increase the
   synchronization representation size quadratically.
4. **Tune `iterations`** — more ticks give more "thinking time" at higher compute cost.
5. The recurrent loop is a Python `for` loop unrolled at JIT trace time. For very
   large `iterations`, consider switching to `jax.lax.scan`.
