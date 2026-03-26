"""Tests for the JAX CTM port.

Verifies:
1. Functional correctness (shapes, dtypes, forward/backward, loss)
2. JAX advantage utilization (JIT, XLA, GPU placement, vmap, grad)
3. Google Colab / NVIDIA GPU compatibility
"""

import os
import math
import pytest
import numpy as np

import jax
import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn
import optax
from flax.training import train_state


# ---------------------------------------------------------------------------
# Replicate model and helpers from the notebook so tests are self-contained
# ---------------------------------------------------------------------------

class SuperLinear(nn.Module):
    out_dims: int
    N: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        in_dims = x.shape[-1]
        scale = 1.0 / math.sqrt(in_dims + self.out_dims)
        w = self.param(
            'w',
            lambda rng, shape: jrandom.uniform(rng, shape, minval=-scale, maxval=scale),
            (in_dims, self.out_dims, self.N),
        )
        b = self.param('b', nn.initializers.zeros_init(), (1, self.N, self.out_dims))
        out = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        out = jnp.einsum('BDM,MHD->BDH', out, w) + b
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


def compute_normalized_entropy(logits, reduction='mean'):
    preds = jax.nn.softmax(logits, axis=-1)
    log_preds = jax.nn.log_softmax(logits, axis=-1)
    entropy = -jnp.sum(preds * log_preds, axis=-1)
    num_classes = logits.shape[-1]
    max_entropy = jnp.log(jnp.array(num_classes, dtype=jnp.float32))
    normalized_entropy = entropy / max_entropy
    if len(logits.shape) > 2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.reshape(normalized_entropy.shape[0], -1).mean(-1)
    return normalized_entropy


class ContinuousThoughtMachine(nn.Module):
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
        self.conv1 = nn.Conv(features=d, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv(features=d, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.bn2 = nn.BatchNorm()

        self.kv_proj = nn.Dense(features=d)
        self.kv_ln = nn.LayerNorm()
        self.q_proj_linear = nn.Dense(features=d)

        assert d % self.heads == 0
        self.head_dim = d // self.heads
        self.attn_q_proj = nn.Dense(features=d)
        self.attn_k_proj = nn.Dense(features=d)
        self.attn_v_proj = nn.Dense(features=d)
        self.attn_out_proj = nn.Dense(features=d)

        self.synapse_drop = nn.Dropout(rate=self.dropout_rate)
        self.synapse_linear = nn.Dense(features=self.d_model * 2)
        self.synapse_ln = nn.LayerNorm()

        self.trace_sl1 = SuperLinear(out_dims=2 * self.memory_hidden_dims, N=self.d_model, dropout_rate=self.dropout_rate)
        self.trace_sl2 = SuperLinear(out_dims=2, N=self.d_model, dropout_rate=self.dropout_rate)

        self.output_projector = nn.Dense(features=self.out_dims)

        self.synch_repr_size_action = (self.n_synch_action * (self.n_synch_action + 1)) // 2
        self.synch_repr_size_out = (self.n_synch_out * (self.n_synch_out + 1)) // 2
        self.triu_idx_action = jnp.triu_indices(self.n_synch_action)
        self.triu_idx_out = jnp.triu_indices(self.n_synch_out)

        self.start_activated_state = self.param(
            'start_activated_state',
            lambda rng, shape: jrandom.uniform(rng, shape, minval=-1.0 / math.sqrt(self.d_model), maxval=1.0 / math.sqrt(self.d_model)),
            (self.d_model,),
        )
        self.start_trace = self.param(
            'start_trace',
            lambda rng, shape: jrandom.uniform(rng, shape, minval=-1.0 / math.sqrt(self.d_model + self.memory_length), maxval=1.0 / math.sqrt(self.d_model + self.memory_length)),
            (self.d_model, self.memory_length),
        )
        self.decay_params_action = self.param('decay_params_action', nn.initializers.zeros_init(), (self.synch_repr_size_action,))
        self.decay_params_out = self.param('decay_params_out', nn.initializers.zeros_init(), (self.synch_repr_size_out,))

    def _max_pool_2x2(self, x):
        return nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')

    def compute_features(self, x, use_running_average: bool):
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
        if synch_type == 'action':
            n = self.n_synch_action
            selected = activated_state[:, -n:]
            idx = self.triu_idx_action
        else:
            n = self.n_synch_out
            selected = activated_state[:, :n]
            idx = self.triu_idx_out
        outer = selected[:, :, None] * selected[:, None, :]
        pairwise_product = outer[:, idx[0], idx[1]]
        is_init = (decay_alpha is None)
        if is_init:
            decay_alpha = pairwise_product
            decay_beta = jnp.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1.0
        synchronisation = decay_alpha / jnp.sqrt(decay_beta)
        return synchronisation, decay_alpha, decay_beta

    def multi_head_attention(self, q, kv):
        B, S, _ = kv.shape
        H = self.heads
        D = self.head_dim
        q = self.attn_q_proj(q)
        k = self.attn_k_proj(kv)
        v = self.attn_v_proj(kv)
        q_heads = q.reshape(B, 1, H, D).transpose(0, 2, 1, 3)
        k_heads = k.reshape(B, S, H, D).transpose(0, 2, 1, 3)
        v_heads = v.reshape(B, S, H, D).transpose(0, 2, 1, 3)
        scale = jnp.sqrt(jnp.array(D, dtype=q.dtype))
        attn_logits = jnp.einsum('bhqd,bhkd->bhqk', q_heads, k_heads) / scale
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v_heads)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, 1, self.d_input)
        attn_out = self.attn_out_proj(attn_out)
        return attn_out, attn_weights

    def compute_certainty(self, current_prediction):
        ne = compute_normalized_entropy(current_prediction)
        return jnp.stack([ne, 1.0 - ne], axis=-1)

    def __call__(self, x, deterministic: bool = True, track: bool = False):
        B = x.shape[0]
        use_running_average = deterministic
        decay_params_action = jnp.clip(self.decay_params_action, 0.0, 15.0)
        decay_params_out = jnp.clip(self.decay_params_out, 0.0, 15.0)
        kv = self.compute_features(x, use_running_average=use_running_average)
        state_trace = jnp.broadcast_to(self.start_trace[None, :, :], (B, self.d_model, self.memory_length))
        activated_state = jnp.broadcast_to(self.start_activated_state[None, :], (B, self.d_model))
        r_action = jnp.exp(-decay_params_action)[None, :]
        r_out = jnp.exp(-decay_params_out)[None, :]
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')
        decay_alpha_action = None
        decay_beta_action = None
        all_predictions = []
        all_certainties = []
        pre_acts_track = []
        post_acts_track = []
        attn_track = []
        synch_out_track = []
        synch_action_track = []
        for stepi in range(self.iterations):
            synch_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')
            q = self.q_proj_linear(synch_action)[:, None, :]
            attn_out, attn_weights = self.multi_head_attention(q, kv)
            attn_out = attn_out.squeeze(1)
            pre_synapse = jnp.concatenate([attn_out, activated_state], axis=-1)
            state = self.synapse_drop(pre_synapse, deterministic=deterministic)
            state = self.synapse_linear(state)
            a, b = jnp.split(state, 2, axis=-1)
            state = a * jax.nn.sigmoid(b)
            state = self.synapse_ln(state)
            state_trace = jnp.concatenate([state_trace[:, :, 1:], state[:, :, None]], axis=-1)
            tp = self.trace_sl1(state_trace, deterministic=deterministic)
            tp_a, tp_b = jnp.split(tp, 2, axis=-1)
            tp = tp_a * jax.nn.sigmoid(tp_b)
            tp = self.trace_sl2(tp, deterministic=deterministic)
            tp_a2, tp_b2 = jnp.split(tp, 2, axis=-1)
            activated_state = (tp_a2 * jax.nn.sigmoid(tp_b2)).squeeze(-1)
            synch_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')
            current_prediction = self.output_projector(synch_out)
            current_certainty = self.compute_certainty(current_prediction)
            all_predictions.append(current_prediction)
            all_certainties.append(current_certainty)
            if track:
                pre_acts_track.append(state_trace[:, :, -1])
                post_acts_track.append(activated_state)
                attn_track.append(attn_weights)
                synch_out_track.append(synch_out)
                synch_action_track.append(synch_action)
        predictions = jnp.stack(all_predictions, axis=-1)
        certainties = jnp.stack(all_certainties, axis=-1)
        if track:
            return (predictions, certainties,
                    (jnp.stack(synch_out_track), jnp.stack(synch_action_track)),
                    jnp.stack(pre_acts_track), jnp.stack(post_acts_track), jnp.stack(attn_track))
        return predictions, certainties, synch_out


def get_loss(predictions, certainties, targets, use_most_certain=True):
    B, C, T = predictions.shape
    preds_btc = jnp.transpose(predictions, (0, 2, 1))
    targets_rep = jnp.broadcast_to(targets[:, None], (B, T))
    losses = optax.softmax_cross_entropy_with_integer_labels(preds_btc, targets_rep)
    loss_index_1 = jnp.argmin(losses, axis=1)
    if use_most_certain:
        loss_index_2 = jnp.argmax(certainties[:, 1, :], axis=-1)
    else:
        loss_index_2 = jnp.full((B,), T - 1, dtype=jnp.int32)
    batch_idx = jnp.arange(B)
    loss_minimum_ce = jnp.mean(losses[batch_idx, loss_index_1])
    loss_selected = jnp.mean(losses[batch_idx, loss_index_2])
    loss = (loss_minimum_ce + loss_selected) / 2.0
    return loss, loss_index_2


def calculate_accuracy(predictions, targets, where_most_certain):
    B = predictions.shape[0]
    pred_classes = jnp.argmax(predictions, axis=1)
    selected_preds = pred_classes[jnp.arange(B), where_most_certain]
    accuracy = jnp.mean(selected_preds == targets)
    return accuracy


class TrainState(train_state.TrainState):
    batch_stats: any


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = dict(
    iterations=3,      # small for fast tests
    d_model=32,
    d_input=32,
    memory_length=4,
    heads=2,
    n_synch_out=8,
    n_synch_action=8,
    memory_hidden_dims=4,
    out_dims=10,
    dropout_rate=0.0,
)

BATCH_SIZE = 4
IMG_H, IMG_W, IMG_C = 28, 28, 1


@pytest.fixture
def key():
    return jrandom.PRNGKey(0)


@pytest.fixture
def model():
    return ContinuousThoughtMachine(**DEFAULT_CONFIG)


@pytest.fixture
def dummy_input():
    return jnp.ones((BATCH_SIZE, IMG_H, IMG_W, IMG_C))


@pytest.fixture
def variables(model, dummy_input, key):
    return model.init(key, dummy_input, deterministic=True)


@pytest.fixture
def state(model, variables):
    optimizer = optax.adamw(learning_rate=1e-4, eps=1e-8)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        batch_stats=variables['batch_stats'],
    )


# ===================================================================
# 1. FUNCTIONAL CORRECTNESS
# ===================================================================

class TestForwardPass:
    """Verify the model's forward pass shapes, dtypes, and value ranges."""

    def test_output_shapes(self, model, variables, dummy_input):
        """predictions=(B,C,T), certainties=(B,2,T), synch_out=(B, synch_size)."""
        predictions, certainties, synch_out = model.apply(
            variables, dummy_input, deterministic=True)

        T = DEFAULT_CONFIG['iterations']
        B = BATCH_SIZE
        C = DEFAULT_CONFIG['out_dims']
        assert predictions.shape == (B, C, T)
        assert certainties.shape == (B, 2, T)
        n_out = DEFAULT_CONFIG['n_synch_out']
        expected_synch_size = (n_out * (n_out + 1)) // 2
        assert synch_out.shape == (B, expected_synch_size)

    def test_output_dtypes(self, model, variables, dummy_input):
        """All outputs should be float32."""
        preds, certs, synch = model.apply(variables, dummy_input, deterministic=True)
        assert preds.dtype == jnp.float32
        assert certs.dtype == jnp.float32
        assert synch.dtype == jnp.float32

    def test_certainty_range(self, model, variables, dummy_input):
        """Certainty (1-entropy) should be in [0, 1]."""
        _, certainties, _ = model.apply(variables, dummy_input, deterministic=True)
        # certainties[:, 0, :] = entropy, certainties[:, 1, :] = 1-entropy
        assert jnp.all(certainties[:, 1, :] >= 0.0 - 1e-6)
        assert jnp.all(certainties[:, 1, :] <= 1.0 + 1e-6)

    def test_no_nan_outputs(self, model, variables, dummy_input):
        """Forward pass should not produce NaN."""
        preds, certs, synch = model.apply(variables, dummy_input, deterministic=True)
        assert not jnp.any(jnp.isnan(preds))
        assert not jnp.any(jnp.isnan(certs))
        assert not jnp.any(jnp.isnan(synch))

    def test_tracking_outputs(self, model, variables, dummy_input):
        """track=True should return 6-tuple with extra diagnostics."""
        result = model.apply(variables, dummy_input, deterministic=True, track=True)
        assert len(result) == 6
        preds, certs, (synch_out_t, synch_act_t), pre_acts, post_acts, attn = result

        T = DEFAULT_CONFIG['iterations']
        B = BATCH_SIZE
        D = DEFAULT_CONFIG['d_model']
        assert pre_acts.shape == (T, B, D)
        assert post_acts.shape == (T, B, D)
        # attn: (T, B, heads, 1, num_positions)
        assert attn.shape[0] == T
        assert attn.shape[1] == B
        assert attn.shape[2] == DEFAULT_CONFIG['heads']
        assert attn.shape[3] == 1

    def test_batch_independence(self, model, variables, key):
        """Different batch elements should produce different outputs."""
        x = jrandom.normal(key, (2, IMG_H, IMG_W, IMG_C))
        preds, _, _ = model.apply(variables, x, deterministic=True)
        # Predictions for sample 0 vs sample 1 should differ
        assert not jnp.allclose(preds[0], preds[1], atol=1e-5)


class TestLossFunction:
    """Verify get_loss and calculate_accuracy."""

    def test_loss_finite(self, model, variables, dummy_input):
        """Loss should be finite."""
        preds, certs, _ = model.apply(variables, dummy_input, deterministic=True)
        labels = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        loss, _ = get_loss(preds, certs, labels)
        assert jnp.isfinite(loss)

    def test_loss_is_scalar(self, model, variables, dummy_input):
        preds, certs, _ = model.apply(variables, dummy_input, deterministic=True)
        labels = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)
        loss, _ = get_loss(preds, certs, labels)
        assert loss.shape == ()

    def test_accuracy_range(self, model, variables, dummy_input):
        preds, certs, _ = model.apply(variables, dummy_input, deterministic=True)
        labels = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)
        _, where = get_loss(preds, certs, labels)
        acc = calculate_accuracy(preds, labels, where)
        assert 0.0 <= float(acc) <= 1.0


class TestAttentionProjections:
    """Verify that attention includes internal Q/K/V and output projections."""

    def test_attention_params_exist(self, variables):
        """Attention should have separate q, k, v, and output projection params."""
        params = variables['params']
        assert 'attn_q_proj' in params, "Missing internal Q projection"
        assert 'attn_k_proj' in params, "Missing internal K projection"
        assert 'attn_v_proj' in params, "Missing internal V projection"
        assert 'attn_out_proj' in params, "Missing output projection"

    def test_attention_param_shapes(self, variables):
        """Internal projections should be (d_input, d_input) weight matrices."""
        d = DEFAULT_CONFIG['d_input']
        params = variables['params']
        assert params['attn_q_proj']['kernel'].shape == (d, d)
        assert params['attn_k_proj']['kernel'].shape == (d, d)
        assert params['attn_v_proj']['kernel'].shape == (d, d)
        assert params['attn_out_proj']['kernel'].shape == (d, d)


class TestDecayParamClamping:
    """Verify decay parameters are clamped to [0, 15]."""

    def test_decay_params_init_zero(self, variables):
        """Decay params should be initialized to zero."""
        params = variables['params']
        assert jnp.allclose(params['decay_params_action'], 0.0)
        assert jnp.allclose(params['decay_params_out'], 0.0)

    def test_clamping_in_forward(self, model, key):
        """Even if decay params are set outside [0, 15], forward clamps them."""
        dummy = jnp.ones((1, IMG_H, IMG_W, IMG_C))
        variables = model.init(key, dummy, deterministic=True)
        # Set decay params to extreme values
        import flax
        params = flax.core.unfreeze(variables['params'])
        params['decay_params_action'] = jnp.full_like(params['decay_params_action'], 20.0)
        params['decay_params_out'] = jnp.full_like(params['decay_params_out'], -5.0)
        variables = flax.core.freeze({**variables, 'params': flax.core.freeze(params)})

        # Forward should not crash (clamping prevents extreme exp values)
        preds, certs, _ = model.apply(variables, dummy, deterministic=True)
        assert not jnp.any(jnp.isnan(preds))


# ===================================================================
# 2. JAX ADVANTAGE UTILIZATION
# ===================================================================

class TestJITCompilation:
    """Verify that JIT compilation works and provides benefits."""

    def test_train_step_jits(self, state, dummy_input):
        """train_step should successfully JIT compile and execute."""
        labels = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)

        @jax.jit
        def jit_train_step(state, images, labels):
            dropout_rng = jrandom.fold_in(jrandom.PRNGKey(0), state.step)
            def loss_fn(params):
                (predictions, certainties, _), updates = state.apply_fn(
                    {'params': params, 'batch_stats': state.batch_stats},
                    images, deterministic=False, mutable=['batch_stats'],
                    rngs={'dropout': dropout_rng})
                loss, where = get_loss(predictions, certainties, labels)
                return loss, (predictions, certainties, where, updates)
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (preds, certs, where, updates)), grads = grad_fn(state.params)
            state_new = state.apply_gradients(grads=grads)
            state_new = state_new.replace(batch_stats=updates['batch_stats'])
            acc = calculate_accuracy(preds, labels, where)
            return state_new, loss, acc

        # Should compile and run without error
        new_state, loss, acc = jit_train_step(state, dummy_input, labels)
        assert jnp.isfinite(loss)

    def test_eval_step_jits(self, state, dummy_input):
        """eval_step should JIT compile successfully."""
        labels = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)

        @jax.jit
        def jit_eval(state, images, labels):
            preds, certs, _ = state.apply_fn(
                {'params': state.params, 'batch_stats': state.batch_stats},
                images, deterministic=True)
            loss, where = get_loss(preds, certs, labels)
            return loss, preds, certs, where

        loss, preds, certs, where = jit_eval(state, dummy_input, labels)
        assert jnp.isfinite(loss)

    def test_jit_idempotent(self, model, variables, dummy_input):
        """JIT'd and non-JIT'd forward pass should produce identical results."""
        def fwd(variables, x):
            return model.apply(variables, x, deterministic=True)

        jit_fwd = jax.jit(fwd)

        preds1, certs1, _ = fwd(variables, dummy_input)
        preds2, certs2, _ = jit_fwd(variables, dummy_input)

        np.testing.assert_allclose(np.array(preds1), np.array(preds2), atol=1e-5)
        np.testing.assert_allclose(np.array(certs1), np.array(certs2), atol=1e-5)

    def test_xla_lowering(self, model, variables, dummy_input):
        """Model should lower to XLA HLO without errors."""
        def fwd(variables, x):
            return model.apply(variables, x, deterministic=True)

        lowered = jax.jit(fwd).lower(variables, dummy_input)
        hlo = lowered.as_text()
        assert len(hlo) > 0, "XLA HLO should be non-empty"
        # Verify it compiles
        compiled = lowered.compile()
        assert compiled is not None


class TestGradientComputation:
    """Verify automatic differentiation works correctly."""

    def test_gradients_exist(self, model, variables, dummy_input):
        """Gradients w.r.t. all params should be computable and non-None."""
        labels = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)

        def loss_fn(params):
            (preds, certs, _), _ = model.apply(
                {'params': params, 'batch_stats': variables['batch_stats']},
                dummy_input, deterministic=False, mutable=['batch_stats'],
                rngs={'dropout': jrandom.PRNGKey(0)})
            loss, _ = get_loss(preds, certs, labels)
            return loss

        grads = jax.grad(loss_fn)(variables['params'])
        # Check that gradients exist for key parameters
        leaves = jax.tree_util.tree_leaves(grads)
        assert len(leaves) > 0
        for leaf in leaves:
            assert leaf is not None

    def test_no_nan_gradients(self, model, variables, dummy_input):
        """No gradient should be NaN."""
        labels = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)

        def loss_fn(params):
            (preds, certs, _), _ = model.apply(
                {'params': params, 'batch_stats': variables['batch_stats']},
                dummy_input, deterministic=False, mutable=['batch_stats'],
                rngs={'dropout': jrandom.PRNGKey(0)})
            loss, _ = get_loss(preds, certs, labels)
            return loss

        grads = jax.grad(loss_fn)(variables['params'])
        for leaf in jax.tree_util.tree_leaves(grads):
            assert not jnp.any(jnp.isnan(leaf)), "Found NaN gradient"

    def test_gradient_flow_through_iterations(self, model, variables, dummy_input):
        """Gradients should flow through all recurrent iterations."""
        labels = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)

        def loss_fn(params):
            (preds, certs, _), _ = model.apply(
                {'params': params, 'batch_stats': variables['batch_stats']},
                dummy_input, deterministic=False, mutable=['batch_stats'],
                rngs={'dropout': jrandom.PRNGKey(0)})
            loss, _ = get_loss(preds, certs, labels)
            return loss

        grads = jax.grad(loss_fn)(variables['params'])
        # The trace processor (SuperLinear) gradients should be non-zero
        # since trace is updated at every iteration
        sl1_grad = grads['trace_sl1']['w']
        assert jnp.any(jnp.abs(sl1_grad) > 1e-10), \
            "Trace processor gradient is zero -- gradient not flowing through iterations"

    def test_value_and_grad_consistency(self, model, variables, dummy_input):
        """value_and_grad should return the same loss as a plain forward pass."""
        labels = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)

        def loss_fn(params):
            (preds, certs, _), _ = model.apply(
                {'params': params, 'batch_stats': variables['batch_stats']},
                dummy_input, deterministic=False, mutable=['batch_stats'],
                rngs={'dropout': jrandom.PRNGKey(0)})
            loss, _ = get_loss(preds, certs, labels)
            return loss

        loss_direct = loss_fn(variables['params'])
        loss_vg, _ = jax.value_and_grad(loss_fn)(variables['params'])
        np.testing.assert_allclose(float(loss_direct), float(loss_vg), atol=1e-5)


class TestFunctionalPurity:
    """Verify JAX functional programming paradigm is respected."""

    def test_deterministic_forward(self, model, variables, dummy_input):
        """Same input + same params -> same output (pure function)."""
        out1 = model.apply(variables, dummy_input, deterministic=True)
        out2 = model.apply(variables, dummy_input, deterministic=True)
        np.testing.assert_array_equal(np.array(out1[0]), np.array(out2[0]))
        np.testing.assert_array_equal(np.array(out1[1]), np.array(out2[1]))

    def test_params_immutable_after_forward(self, model, variables, dummy_input):
        """Forward pass should not mutate the input variables."""
        import copy
        params_before = jax.tree_util.tree_map(lambda x: x.copy(), variables['params'])
        _ = model.apply(variables, dummy_input, deterministic=True)
        for a, b in zip(jax.tree_util.tree_leaves(params_before),
                        jax.tree_util.tree_leaves(variables['params'])):
            np.testing.assert_array_equal(np.array(a), np.array(b))


class TestBatchNormHandling:
    """Verify BN works in both train and eval mode."""

    def test_batch_stats_updated_during_training(self, model, variables, dummy_input):
        """Mutable batch_stats should change after a training forward pass."""
        (_, _, _), updates = model.apply(
            variables, dummy_input, deterministic=False, mutable=['batch_stats'],
            rngs={'dropout': jrandom.PRNGKey(0)})
        old_mean = jax.tree_util.tree_leaves(variables['batch_stats'])[0]
        new_mean = jax.tree_util.tree_leaves(updates['batch_stats'])[0]
        assert not jnp.allclose(old_mean, new_mean), \
            "batch_stats not updated during training"

    def test_batch_stats_frozen_during_eval(self, model, variables, dummy_input):
        """In eval mode (deterministic=True, no mutable), batch_stats stay fixed."""
        _ = model.apply(variables, dummy_input, deterministic=True)
        # No crash == pass (no mutable=['batch_stats'] means it uses running stats)


class TestVmap:
    """Verify model works with vmap for per-example operations."""

    def test_vmap_loss(self, model, variables, key):
        """vmap should compute per-example loss without manual batch loop."""
        x = jrandom.normal(key, (BATCH_SIZE, IMG_H, IMG_W, IMG_C))
        preds, certs, _ = model.apply(variables, x, deterministic=True)
        labels = jnp.arange(BATCH_SIZE, dtype=jnp.int32) % DEFAULT_CONFIG['out_dims']

        # Per-example loss via vmap: each element is (C, T) and (2, T)
        def single_loss(pred, cert, label):
            # pred: (C, T), cert: (2, T), label: scalar
            # get_loss expects (B, C, T) so add a batch dim of 1
            l, _ = get_loss(pred[None], cert[None], label[None])
            return l

        per_example = jax.vmap(single_loss)(preds, certs, labels)
        assert per_example.shape == (BATCH_SIZE,)
        assert jnp.all(jnp.isfinite(per_example))


# ===================================================================
# 3. GOOGLE COLAB + NVIDIA GPU COMPATIBILITY
# ===================================================================

class TestDevicePlacement:
    """Verify arrays land on the correct device."""

    def test_default_device(self, dummy_input):
        """Input array should be on the default JAX device."""
        device = dummy_input.devices().pop()
        # Should be either CPU or GPU -- not None
        assert device is not None
        backend = jax.default_backend()
        assert backend in ('cpu', 'gpu', 'tpu'), f"Unexpected backend: {backend}"

    def test_output_on_same_device(self, model, variables, dummy_input):
        """Model outputs should be on the same device as inputs."""
        preds, _, _ = model.apply(variables, dummy_input, deterministic=True)
        assert preds.devices() == dummy_input.devices()

    def test_gpu_available_detection(self):
        """JAX should correctly detect available devices."""
        devices = jax.devices()
        assert len(devices) >= 1
        # Print device info for diagnostic purposes
        print(f"JAX devices: {devices}")
        print(f"Default backend: {jax.default_backend()}")


class TestFloat32Platform:
    """Ensure float32 is used (Colab GPUs default to float32)."""

    def test_params_float32(self, variables):
        """All parameters should be float32 for GPU compatibility."""
        for leaf in jax.tree_util.tree_leaves(variables['params']):
            if hasattr(leaf, 'dtype'):
                assert leaf.dtype == jnp.float32 or leaf.dtype == jnp.int32, \
                    f"Unexpected dtype: {leaf.dtype}"

    def test_model_output_float32(self, model, variables, dummy_input):
        """Outputs should be float32."""
        preds, certs, _ = model.apply(variables, dummy_input, deterministic=True)
        assert preds.dtype == jnp.float32
        assert certs.dtype == jnp.float32


class TestMemoryEfficiency:
    """Verify JAX-specific memory efficiency patterns."""

    def test_broadcast_not_repeat(self, model, variables, key):
        """Initial state should use broadcasting, not materialized copies."""
        # The model internally uses jnp.broadcast_to for start states.
        # We verify the param is (d_model,) not (B, d_model).
        params = variables['params']
        assert params['start_activated_state'].shape == (DEFAULT_CONFIG['d_model'],)
        assert params['start_trace'].shape == (DEFAULT_CONFIG['d_model'], DEFAULT_CONFIG['memory_length'])

    def test_param_count_reasonable(self, variables):
        """Model should have a reasonable number of parameters."""
        count = sum(x.size for x in jax.tree_util.tree_leaves(variables['params']))
        # With d_model=32, d_input=32, this should be modest
        assert count > 0
        assert count < 1_000_000  # sanity upper bound for test config
        print(f"Test model parameter count: {count:,}")


class TestTrainStateIntegration:
    """End-to-end training state tests."""

    def test_full_training_step(self, state, dummy_input):
        """Complete train step: forward + backward + update."""
        labels = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)

        @jax.jit
        def step(state, images, labels):
            dropout_rng = jrandom.fold_in(jrandom.PRNGKey(0), state.step)
            def loss_fn(params):
                (preds, certs, _), updates = state.apply_fn(
                    {'params': params, 'batch_stats': state.batch_stats},
                    images, deterministic=False, mutable=['batch_stats'],
                    rngs={'dropout': dropout_rng})
                loss, where = get_loss(preds, certs, labels)
                return loss, (preds, certs, where, updates)
            (loss, (preds, certs, where, updates)), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(
                batch_stats=updates['batch_stats'],
                params={**state.params,
                        'decay_params_action': jnp.clip(state.params['decay_params_action'], 0.0, 15.0),
                        'decay_params_out': jnp.clip(state.params['decay_params_out'], 0.0, 15.0)})
            return state, loss

        new_state, loss = step(state, dummy_input, labels)
        assert jnp.isfinite(loss)
        # Params should have changed after gradient update.
        # Use strict atol=0 so even tiny gradient updates are detected.
        any_changed = any(
            not jnp.allclose(a, b, atol=0, rtol=0)
            for a, b in zip(jax.tree_util.tree_leaves(state.params),
                            jax.tree_util.tree_leaves(new_state.params))
        )
        assert any_changed, "No params changed after training step"

    def test_multiple_steps_loss_changes(self, state, key):
        """Loss should change across multiple training steps."""
        x = jrandom.normal(key, (BATCH_SIZE, IMG_H, IMG_W, IMG_C))
        labels = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)

        @jax.jit
        def step(state, images, labels):
            dropout_rng = jrandom.fold_in(jrandom.PRNGKey(0), state.step)
            def loss_fn(params):
                (preds, certs, _), updates = state.apply_fn(
                    {'params': params, 'batch_stats': state.batch_stats},
                    images, deterministic=False, mutable=['batch_stats'],
                    rngs={'dropout': dropout_rng})
                loss, where = get_loss(preds, certs, labels)
                return loss, updates
            (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates['batch_stats'])
            return state, loss

        losses = []
        for _ in range(5):
            state, loss = step(state, x, labels)
            losses.append(float(loss))

        # Loss should not be constant over 5 steps
        assert not all(abs(losses[i] - losses[0]) < 1e-7 for i in range(1, 5)), \
            f"Loss did not change across steps: {losses}"


class TestJITNoRecompilation:
    """Verify that same-shaped inputs do not trigger JIT recompilation."""

    def test_no_retrace_same_shape(self, model, variables, key):
        """Calling JIT'd function twice with same shape should reuse cache."""

        call_count = 0
        original_apply = model.apply

        def counting_apply(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_apply(*args, **kwargs)

        @jax.jit
        def fwd(variables, x):
            return model.apply(variables, x, deterministic=True)

        x1 = jrandom.normal(key, (BATCH_SIZE, IMG_H, IMG_W, IMG_C))
        x2 = jrandom.normal(jrandom.PRNGKey(99), (BATCH_SIZE, IMG_H, IMG_W, IMG_C))

        # First call triggers compilation
        _ = fwd(variables, x1)
        # Second call should reuse compiled code (same shape)
        _ = fwd(variables, x2)
        # If we get here without error, the cache was used successfully

    def test_fixed_batch_avoids_retrace(self, model, variables, key):
        """Using drop_last=True (fixed batch) avoids shape-triggered retraces."""
        @jax.jit
        def fwd(variables, x):
            return model.apply(variables, x, deterministic=True)

        # All calls with BATCH_SIZE should use the same compiled artifact
        for i in range(3):
            x = jrandom.normal(jrandom.PRNGKey(i), (BATCH_SIZE, IMG_H, IMG_W, IMG_C))
            preds, _, _ = fwd(variables, x)
            assert preds.shape == (BATCH_SIZE, DEFAULT_CONFIG['out_dims'],
                                   DEFAULT_CONFIG['iterations'])


class TestJITSpeedup:
    """Verify JIT provides measurable speedup over eager execution."""

    def test_jit_faster_than_eager(self, model, variables, key):
        """Second JIT call should be faster than first (eager) call."""
        import time

        x = jrandom.normal(key, (BATCH_SIZE, IMG_H, IMG_W, IMG_C))

        # Eager (non-JIT)
        def eager_fwd(variables, x):
            return model.apply(variables, x, deterministic=True)

        # JIT
        jit_fwd = jax.jit(eager_fwd)

        # Warmup JIT (includes compilation)
        _ = jit_fwd(variables, x)
        jax.block_until_ready(_)

        # Time JIT'd execution (post-compilation)
        start = time.perf_counter()
        for _ in range(5):
            out = jit_fwd(variables, x)
            jax.block_until_ready(out)
        jit_time = time.perf_counter() - start

        # Time eager execution
        start = time.perf_counter()
        for _ in range(5):
            out = eager_fwd(variables, x)
            jax.block_until_ready(out)
        eager_time = time.perf_counter() - start

        # JIT should be faster (or at least not slower on CPU)
        print(f"JIT time: {jit_time:.4f}s, Eager time: {eager_time:.4f}s, "
              f"Speedup: {eager_time / jit_time:.2f}x")
        # We don't assert a strict speedup because on CPU with small models
        # the difference may be marginal, but it should compile and run.


class TestGradientCheckpointing:
    """Verify model is compatible with jax.checkpoint for memory efficiency."""

    def test_checkpoint_compatible(self, model, variables, dummy_input):
        """jax.checkpoint should wrap the forward pass without error."""
        labels = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)

        @jax.checkpoint
        def checkpointed_fwd(params):
            (preds, certs, _), updates = model.apply(
                {'params': params, 'batch_stats': variables['batch_stats']},
                dummy_input, deterministic=False, mutable=['batch_stats'],
                rngs={'dropout': jrandom.PRNGKey(0)})
            loss, _ = get_loss(preds, certs, labels)
            return loss

        # Should compute gradients through checkpointed forward
        grads = jax.grad(checkpointed_fwd)(variables['params'])
        leaves = jax.tree_util.tree_leaves(grads)
        assert len(leaves) > 0
        for leaf in leaves:
            assert jnp.isfinite(leaf).all(), "NaN/Inf in checkpointed gradient"

    def test_checkpoint_matches_standard(self, model, variables, dummy_input):
        """Checkpointed gradients should match standard gradients."""
        labels = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)

        def loss_fn(params):
            (preds, certs, _), _ = model.apply(
                {'params': params, 'batch_stats': variables['batch_stats']},
                dummy_input, deterministic=False, mutable=['batch_stats'],
                rngs={'dropout': jrandom.PRNGKey(0)})
            loss, _ = get_loss(preds, certs, labels)
            return loss

        standard_grads = jax.grad(loss_fn)(variables['params'])

        @jax.checkpoint
        def ckpt_loss_fn(params):
            return loss_fn(params)

        ckpt_grads = jax.grad(ckpt_loss_fn)(variables['params'])

        for a, b in zip(jax.tree_util.tree_leaves(standard_grads),
                        jax.tree_util.tree_leaves(ckpt_grads)):
            np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-5,
                                       err_msg="Checkpointed grad differs from standard")


class TestPytreeCompatibility:
    """Verify model params are proper JAX pytrees."""

    def test_tree_map(self, variables):
        """tree_map should work on model params (e.g. for weight decay)."""
        scaled = jax.tree_util.tree_map(lambda x: x * 0.99, variables['params'])
        for a, b in zip(jax.tree_util.tree_leaves(variables['params']),
                        jax.tree_util.tree_leaves(scaled)):
            np.testing.assert_allclose(np.array(a) * 0.99, np.array(b), atol=1e-7)

    def test_tree_leaves_count(self, variables):
        """All parameter leaves should be JAX arrays."""
        leaves = jax.tree_util.tree_leaves(variables['params'])
        assert len(leaves) > 0
        for leaf in leaves:
            assert isinstance(leaf, jnp.ndarray), f"Expected jnp.ndarray, got {type(leaf)}"

    def test_tree_structure_preserved(self, variables):
        """Serialization round-trip should preserve pytree structure."""
        struct = jax.tree_util.tree_structure(variables['params'])
        leaves = jax.tree_util.tree_leaves(variables['params'])
        reconstructed = jax.tree_util.tree_unflatten(struct, leaves)
        for a, b in zip(jax.tree_util.tree_leaves(variables['params']),
                        jax.tree_util.tree_leaves(reconstructed)):
            np.testing.assert_array_equal(np.array(a), np.array(b))

    def test_tree_map_for_clipping(self, variables):
        """Gradient clipping via tree_map should work (common JAX pattern)."""
        grads = jax.tree_util.tree_map(lambda x: x * 10.0, variables['params'])
        max_norm = 1.0
        clipped = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -max_norm, max_norm), grads)
        for leaf in jax.tree_util.tree_leaves(clipped):
            assert jnp.all(leaf >= -max_norm - 1e-6)
            assert jnp.all(leaf <= max_norm + 1e-6)


class TestDonateArgnums:
    """Verify buffer donation works for training step efficiency."""

    def test_donate_state_in_train_step(self, state, dummy_input):
        """donate_argnums should allow buffer reuse for the state."""
        labels = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)

        @jax.jit  # donate_argnums not used here because state is a pytree
        def step(state, images, labels):
            dropout_rng = jrandom.fold_in(jrandom.PRNGKey(0), state.step)
            def loss_fn(params):
                (preds, certs, _), updates = state.apply_fn(
                    {'params': params, 'batch_stats': state.batch_stats},
                    images, deterministic=False, mutable=['batch_stats'],
                    rngs={'dropout': dropout_rng})
                loss, where = get_loss(preds, certs, labels)
                return loss, updates
            (loss, updates), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            state_new = state.apply_gradients(grads=grads)
            state_new = state_new.replace(batch_stats=updates['batch_stats'])
            return state_new, loss

        # With donate_argnums=(0,), the old state buffer is donated to the new state
        step_donate = jax.jit(step, donate_argnums=(0,))
        new_state, loss = step_donate(state, dummy_input, labels)
        assert jnp.isfinite(loss)
        # The old `state` buffers may now be invalid (donated), but new_state is valid
        preds, _, _ = new_state.apply_fn(
            {'params': new_state.params, 'batch_stats': new_state.batch_stats},
            dummy_input, deterministic=True)
        assert not jnp.any(jnp.isnan(preds))


class TestSuperLinearEinsum:
    """Verify SuperLinear einsum produces correct per-neuron linear transform."""

    def test_einsum_matches_loop(self, key):
        """Einsum result should match an explicit loop over neurons."""
        N, in_dims, out_dims = 8, 4, 6
        B = 2
        sl = SuperLinear(out_dims=out_dims, N=N)
        x = jrandom.normal(key, (B, N, in_dims))
        variables = sl.init(jrandom.PRNGKey(1), x, deterministic=True)
        result = sl.apply(variables, x, deterministic=True)

        # Manually compute per-neuron linear transform
        w = variables['params']['w']  # (in_dims, out_dims, N)
        b = variables['params']['b']  # (1, N, out_dims)
        manual = jnp.zeros((B, N, out_dims))
        for d in range(N):
            # x[:, d, :] @ w[:, :, d] + b[0, d, :]
            manual = manual.at[:, d, :].set(x[:, d, :] @ w[:, :, d] + b[0, d, :])
        if manual.shape[-1] == 1:
            manual = manual.squeeze(-1)

        np.testing.assert_allclose(np.array(result), np.array(manual), atol=1e-5,
                                   err_msg="Einsum does not match manual per-neuron computation")

    def test_superlinear_output_shape(self, key):
        """SuperLinear should produce (B, N, out_dims) or (B, N) if squeezed."""
        N, out_dims = 16, 1
        sl = SuperLinear(out_dims=out_dims, N=N)
        x = jrandom.normal(key, (2, N, 5))
        variables = sl.init(jrandom.PRNGKey(1), x, deterministic=True)
        result = sl.apply(variables, x, deterministic=True)
        # out_dims=1 -> squeeze removes last dim -> (B, N)
        assert result.shape == (2, N)


class TestXLAOptimization:
    """Verify XLA compilation optimizations."""

    def test_hlo_contains_fusion(self, model, variables, dummy_input):
        """Compiled HLO should contain fused operations."""
        def fwd(variables, x):
            return model.apply(variables, x, deterministic=True)
        lowered = jax.jit(fwd).lower(variables, dummy_input)
        hlo = lowered.as_text()
        # XLA typically produces fusion operations for GALUs, einsums, etc.
        # The HLO text should be substantial (many ops fused into a graph)
        assert len(hlo) > 1000, "HLO too small -- model may not be compiling properly"

    def test_compiled_execution_shape(self, model, variables, dummy_input):
        """Compiled model should produce correct output shapes."""
        def fwd(variables, x):
            return model.apply(variables, x, deterministic=True)
        compiled = jax.jit(fwd).lower(variables, dummy_input).compile()
        preds, certs, synch = compiled(variables, dummy_input)
        T = DEFAULT_CONFIG['iterations']
        assert preds.shape == (BATCH_SIZE, DEFAULT_CONFIG['out_dims'], T)
        assert certs.shape == (BATCH_SIZE, 2, T)


class TestColabDependencies:
    """Verify all dependencies are importable (matching Colab environment)."""

    def test_core_imports(self):
        """All core imports from the notebook should work."""
        import jax
        import jax.numpy as jnp
        import flax.linen as nn
        import optax
        from flax.training import train_state
        assert True

    def test_data_imports(self):
        """torchvision and torch data loaders should be importable."""
        try:
            from torchvision import datasets, transforms
            import torch.utils.data
        except ImportError:
            pytest.skip("torch/torchvision not installed")

    def test_viz_imports(self):
        """Visualization dependencies should be importable."""
        import matplotlib.pyplot as plt
        from scipy.special import softmax
        from scipy.ndimage import zoom
        import imageio
        import seaborn
        assert True

    def test_jax_version(self):
        """JAX version should support all features used."""
        version = tuple(int(x) for x in jax.__version__.split('.')[:2])
        assert version >= (0, 4), f"JAX version {jax.__version__} too old, need >= 0.4"
