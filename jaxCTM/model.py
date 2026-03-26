import math

import jax
import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn

from .layers import SuperLinear


def compute_normalized_entropy(logits, reduction='mean'):
    """Normalized Shannon entropy of a softmax distribution.

    Maps raw logits to a certainty signal in [0, 1]:
        0 = maximum uncertainty (uniform distribution)
        1 = maximum certainty (point mass)

    Args:
        logits: (..., num_classes) raw logits.
        reduction: When 'mean' and logits has >2 dims, average over trailing dims.

    Returns:
        normalized_entropy: (B,) values in [0, 1].
    """
    preds = jax.nn.softmax(logits, axis=-1)
    log_preds = jax.nn.log_softmax(logits, axis=-1)
    entropy = -jnp.sum(preds * log_preds, axis=-1)
    num_classes = logits.shape[-1]
    max_entropy = jnp.log(jnp.array(num_classes, dtype=jnp.float32))
    normalized_entropy = entropy / max_entropy
    if len(logits.shape) > 2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.reshape(
            normalized_entropy.shape[0], -1
        ).mean(-1)
    return normalized_entropy


class ContinuousThoughtMachine(nn.Module):
    """Continuous Thought Machine (CTM) in Flax/JAX.

    Models reasoning as a temporal process: the model iterates through
    `iterations` internal ticks, updating its belief at each step.

    Architecture summary
    --------------------
    1. CNN backbone extracts spatial features from the input image once,
       producing key-value pairs ``kv`` for cross-attention.
    2. At each tick:
       a. Compute action synchronization (pairwise outer product of last
          ``n_synch_action`` neurons) with EMA decay.
       b. Project synchronization to an attention query; run cross-attention
          over ``kv`` to read from the image.
       c. Concatenate attention output with current neuron activations;
          pass through synapse layer (GLU + LayerNorm) to update state.
       d. Slide the per-neuron activation trace window.
       e. Run trace_processor (two SuperLinear + GLU layers) to compute
          new neuron activations from each neuron's own history.
       f. Compute output synchronization (first ``n_synch_out`` neurons).
       g. Project output synchronization to class logits; compute certainty.
    3. Stack predictions and certainties across all ticks.

    The recurrent loop is a plain Python ``for`` loop, which JAX/XLA unrolls
    at JIT trace time into a single fused computation graph.

    Args
    ----
    iterations       : int   — number of internal ticks (default 15)
    d_model          : int   — number of neurons
    d_input          : int   — CNN feature / attention embedding dimension
    memory_length    : int   — length of per-neuron activation history window
    heads            : int   — attention heads (d_input must be divisible)
    n_synch_out      : int   — neurons used for output synchronization
    n_synch_action   : int   — neurons used for attention-query synchronization
    out_dims         : int   — number of output classes
    memory_hidden_dims: int  — hidden size inside trace_processor SuperLinear
    dropout_rate     : float — dropout probability (0 = disabled)

    Inputs
    ------
    x            : (B, H, W, C)  — channels-last images
    deterministic: bool          — True disables dropout and uses BN running stats
    track        : bool          — True returns per-tick diagnostic tensors

    Outputs (default)
    -----------------
    predictions  : (B, out_dims, iterations)
    certainties  : (B, 2, iterations)  — [norm_entropy, 1 - norm_entropy]
    synch_out    : (B, synch_repr_size_out)  — final-tick output synchronization

    Outputs (track=True)
    --------------------
    predictions, certainties,
    (synch_out_track, synch_action_track),
    pre_activations,    # (T, B, d_model)
    post_activations,   # (T, B, d_model)
    attention           # (T, B, heads, 1, num_positions)
    """

    iterations: int
    d_model: int
    d_input: int
    memory_length: int
    heads: int
    n_synch_out: int
    n_synch_action: int
    out_dims: int
    memory_hidden_dims: int
    dropout_rate: float = 0.0

    def setup(self):
        d = self.d_input

        # --- CNN Backbone ---
        self.conv1 = nn.Conv(features=d, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.bn1   = nn.BatchNorm()
        self.conv2 = nn.Conv(features=d, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.bn2   = nn.BatchNorm()

        # --- Key-value projection (applied once to CNN output) ---
        self.kv_proj = nn.Dense(features=d)
        self.kv_ln   = nn.LayerNorm()

        # --- Attention query projection (from synch_action) ---
        self.q_proj_linear = nn.Dense(features=d)

        # --- Multi-head attention internal projections ---
        assert d % self.heads == 0, "d_input must be divisible by heads"
        self.head_dim      = d // self.heads
        self.attn_q_proj   = nn.Dense(features=d)
        self.attn_k_proj   = nn.Dense(features=d)
        self.attn_v_proj   = nn.Dense(features=d)
        self.attn_out_proj = nn.Dense(features=d)

        # --- Synapse layer (state update at each tick) ---
        self.synapse_drop   = nn.Dropout(rate=self.dropout_rate)
        self.synapse_linear = nn.Dense(features=self.d_model * 2)
        self.synapse_ln     = nn.LayerNorm()

        # --- Trace processor (neuron-level models over activation history) ---
        self.trace_sl1 = SuperLinear(
            out_dims=2 * self.memory_hidden_dims,
            N=self.d_model,
            dropout_rate=self.dropout_rate,
        )
        self.trace_sl2 = SuperLinear(
            out_dims=2,
            N=self.d_model,
            dropout_rate=self.dropout_rate,
        )

        # --- Output projection ---
        self.output_projector = nn.Dense(features=self.out_dims)

        # --- Synchronization sizes ---
        self.synch_repr_size_action = (self.n_synch_action * (self.n_synch_action + 1)) // 2
        self.synch_repr_size_out    = (self.n_synch_out    * (self.n_synch_out    + 1)) // 2

        # Upper-triangular indices (static, computed once at setup)
        self.triu_idx_action = jnp.triu_indices(self.n_synch_action)
        self.triu_idx_out    = jnp.triu_indices(self.n_synch_out)

        # --- Learned initial recurrent state ---
        scale_state = 1.0 / math.sqrt(self.d_model)
        scale_trace = 1.0 / math.sqrt(self.d_model + self.memory_length)

        self.start_activated_state = self.param(
            'start_activated_state',
            lambda rng, shape: jrandom.uniform(rng, shape, minval=-scale_state, maxval=scale_state),
            (self.d_model,),
        )
        self.start_trace = self.param(
            'start_trace',
            lambda rng, shape: jrandom.uniform(rng, shape, minval=-scale_trace, maxval=scale_trace),
            (self.d_model, self.memory_length),
        )

        # --- Learned decay parameters (one per upper-triangular entry) ---
        # Clamped to [0, 15] after every gradient step.
        # r = exp(-decay_param) ∈ [exp(-15), 1] controls EMA decay speed.
        self.decay_params_action = self.param(
            'decay_params_action',
            nn.initializers.zeros_init(),
            (self.synch_repr_size_action,),
        )
        self.decay_params_out = self.param(
            'decay_params_out',
            nn.initializers.zeros_init(),
            (self.synch_repr_size_out,),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _max_pool_2x2(self, x):
        return nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')

    def compute_features(self, x, use_running_average: bool):
        """CNN backbone: two conv-BN-ReLU-pool blocks → flattened kv pairs.

        Args:
            x: (B, H, W, C) input images.
            use_running_average: passed to BatchNorm.

        Returns:
            kv: (B, num_spatial_positions, d_input)
        """
        h = self.conv1(x)
        h = self.bn1(h, use_running_average=use_running_average)
        h = nn.relu(h)
        h = self._max_pool_2x2(h)
        h = self.conv2(h)
        h = self.bn2(h, use_running_average=use_running_average)
        h = nn.relu(h)
        h = self._max_pool_2x2(h)
        B = h.shape[0]
        h = h.reshape(B, -1, self.d_input)
        kv = self.kv_ln(self.kv_proj(h))
        return kv

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        """Pairwise outer-product synchronization with EMA decay.

        Selects a subset of neurons, computes their pairwise outer products,
        keeps the upper-triangular entries, and applies an exponential moving
        average to smooth the signal across ticks.

        Args:
            activated_state: (B, d_model) current neuron activations.
            decay_alpha: (B, synch_size) or None on first call.
            decay_beta:  (B, synch_size) or None on first call.
            r: (1, synch_size) per-entry decay rates = exp(-decay_param).
            synch_type: 'action' (uses last n neurons) or 'out' (uses first n neurons).

        Returns:
            synchronisation: (B, synch_size)
            decay_alpha:     updated EMA numerator
            decay_beta:      updated EMA denominator
        """
        if synch_type == 'action':
            n   = self.n_synch_action
            selected = activated_state[:, -n:]
            idx = self.triu_idx_action
        else:
            n   = self.n_synch_out
            selected = activated_state[:, :n]
            idx = self.triu_idx_out

        outer = selected[:, :, None] * selected[:, None, :]  # (B, n, n)
        pairwise_product = outer[:, idx[0], idx[1]]           # (B, synch_size)

        if decay_alpha is None:
            decay_alpha = pairwise_product
            decay_beta  = jnp.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta  = r * decay_beta  + 1.0

        synchronisation = decay_alpha / jnp.sqrt(decay_beta)
        return synchronisation, decay_alpha, decay_beta

    def multi_head_attention(self, q, kv):
        """Scaled dot-product multi-head cross-attention.

        Mirrors PyTorch's nn.MultiheadAttention: learned Q/K/V projections,
        scaled dot-product attention, output projection.

        Args:
            q:  (B, 1, d_input) query (one query token per sample).
            kv: (B, S, d_input) keys/values from CNN feature map.

        Returns:
            attn_out:     (B, 1, d_input)
            attn_weights: (B, heads, 1, S)
        """
        B, S, _ = kv.shape
        H = self.heads
        D = self.head_dim

        q_proj = self.attn_q_proj(q)    # (B, 1, d_input)
        k_proj = self.attn_k_proj(kv)   # (B, S, d_input)
        v_proj = self.attn_v_proj(kv)   # (B, S, d_input)

        q_heads = q_proj.reshape(B, 1, H, D).transpose(0, 2, 1, 3)  # (B, H, 1, D)
        k_heads = k_proj.reshape(B, S, H, D).transpose(0, 2, 1, 3)  # (B, H, S, D)
        v_heads = v_proj.reshape(B, S, H, D).transpose(0, 2, 1, 3)  # (B, H, S, D)

        scale       = jnp.sqrt(jnp.array(D, dtype=q_proj.dtype))
        attn_logits = jnp.einsum('bhqd,bhkd->bhqk', q_heads, k_heads) / scale  # (B,H,1,S)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v_heads)          # (B,H,1,D)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, 1, self.d_input)    # (B,1,d_input)
        attn_out = self.attn_out_proj(attn_out)                                   # (B,1,d_input)

        return attn_out, attn_weights

    def compute_certainty(self, current_prediction):
        """Convert logits to a [uncertainty, certainty] pair.

        Returns:
            (B, 2) — column 0 = normalized entropy, column 1 = 1 - entropy.
        """
        ne = compute_normalized_entropy(current_prediction)
        return jnp.stack([ne, 1.0 - ne], axis=-1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, x, deterministic: bool = True, track: bool = False):
        B = x.shape[0]
        use_running_average = deterministic

        # Clamp decay params (mirrors PyTorch forward pre-hook)
        decay_params_action = jnp.clip(self.decay_params_action, 0.0, 15.0)
        decay_params_out    = jnp.clip(self.decay_params_out,    0.0, 15.0)

        # --- Extract image features (computed once, shared across all ticks) ---
        kv = self.compute_features(x, use_running_average=use_running_average)

        # --- Initialize recurrent state ---
        state_trace = jnp.broadcast_to(
            self.start_trace[None, :, :], (B, self.d_model, self.memory_length)
        )
        activated_state = jnp.broadcast_to(
            self.start_activated_state[None, :], (B, self.d_model)
        )

        # Decay rates — (1, synch_size), broadcast-ready
        r_action = jnp.exp(-decay_params_action)[None, :]
        r_out    = jnp.exp(-decay_params_out)[None, :]

        # Seed the output EMA with the initial activated state
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
            activated_state, None, None, r_out, synch_type='out'
        )
        decay_alpha_action = None
        decay_beta_action  = None

        # --- Accumulators ---
        all_predictions  = []
        all_certainties  = []

        pre_acts_track      = []
        post_acts_track     = []
        attn_track          = []
        synch_out_track     = []
        synch_action_track  = []

        # --- Recurrent loop (unrolled by XLA at JIT trace time) ---
        for _ in range(self.iterations):

            # 1. Action synchronization → attention query
            synch_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action,
                r_action, synch_type='action'
            )

            # 2. Cross-attention
            q = self.q_proj_linear(synch_action)[:, None, :]   # (B, 1, d_input)
            attn_out, attn_weights = self.multi_head_attention(q, kv)
            attn_out = attn_out.squeeze(1)                      # (B, d_input)

            # 3. Synapses — GLU state update
            pre_synapse = jnp.concatenate([attn_out, activated_state], axis=-1)
            state = self.synapse_drop(pre_synapse, deterministic=deterministic)
            state = self.synapse_linear(state)
            a, b  = jnp.split(state, 2, axis=-1)
            state = a * jax.nn.sigmoid(b)           # GLU
            state = self.synapse_ln(state)

            # 4. Slide trace window (drop oldest, append current)
            state_trace = jnp.concatenate(
                [state_trace[:, :, 1:], state[:, :, None]], axis=-1
            )

            # 5. Trace processor — neuron-level models
            tp = self.trace_sl1(state_trace, deterministic=deterministic)
            tp_a, tp_b = jnp.split(tp, 2, axis=-1)
            tp = tp_a * jax.nn.sigmoid(tp_b)        # GLU
            tp = self.trace_sl2(tp, deterministic=deterministic)
            tp_a2, tp_b2 = jnp.split(tp, 2, axis=-1)
            activated_state = (tp_a2 * jax.nn.sigmoid(tp_b2)).squeeze(-1)  # GLU + squeeze

            # 6. Output synchronization
            synch_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out,
                r_out, synch_type='out'
            )

            # 7. Prediction + certainty
            current_prediction = self.output_projector(synch_out)
            current_certainty  = self.compute_certainty(current_prediction)

            all_predictions.append(current_prediction)
            all_certainties.append(current_certainty)

            if track:
                pre_acts_track.append(state_trace[:, :, -1])
                post_acts_track.append(activated_state)
                attn_track.append(attn_weights)
                synch_out_track.append(synch_out)
                synch_action_track.append(synch_action)

        predictions = jnp.stack(all_predictions, axis=-1)   # (B, out_dims, T)
        certainties = jnp.stack(all_certainties, axis=-1)   # (B, 2, T)

        if track:
            return (
                predictions,
                certainties,
                (jnp.stack(synch_out_track), jnp.stack(synch_action_track)),
                jnp.stack(pre_acts_track),    # (T, B, d_model)
                jnp.stack(post_acts_track),   # (T, B, d_model)
                jnp.stack(attn_track),        # (T, B, heads, 1, S)
            )

        return predictions, certainties, synch_out
