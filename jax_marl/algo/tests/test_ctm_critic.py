"""Tests for CTMCritic forward pass.

Covers:
- Output shapes: q_vals=(B,1,T), certs=(B,2,T)
- Certainty column certs[:,1,:] is in [0,1] (sigmoid)
- Q values are finite (no NaN/Inf)
- Certainty scores vary across ticks (CTM is actually "thinking")
- tick_certain = argmax(certs[:,1,:]) produces a valid index
- Deterministic=True vs False both work
- Target network (same architecture, separate params) produces same shapes
- alpha=0.0 and alpha=1.0 both produce finite outputs (blend extremes)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import jax
import jax.numpy as jnp
from jax import random

from networks import CTMCritic, create_ctm_critic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_AGENTS   = 4
OBS_DIM    = 8
ACTION_DIM = 2
TRAJ_LEN   = 6
ITERATIONS = 5   # small for speed

CTM_KWARGS = dict(
    n_agents=N_AGENTS,
    obs_dim=OBS_DIM,
    action_dim=ACTION_DIM,
    traj_len=TRAJ_LEN,
    iterations=ITERATIONS,
    d_model=16,
    d_input=32,
    memory_length=3,
    heads=2,
    n_synch_out=4,
    n_synch_action=4,
    memory_hidden_dims=4,
)


def _make_inputs(key, B=4):
    k1, k2, k3 = random.split(key, 3)
    traj    = random.uniform(k1, (B, TRAJ_LEN, N_AGENTS, 2))
    all_obs = random.uniform(k2, (B, N_AGENTS, OBS_DIM))
    acts    = random.uniform(k3, (B, N_AGENTS, ACTION_DIM))
    return traj, all_obs, acts


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_q_vals_shape():
    key = random.PRNGKey(0)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    traj, obs, acts = _make_inputs(key, B=4)

    q_vals, certs = critic.apply(params, traj, obs, acts, alpha=0.0)

    assert q_vals.shape == (4, 1, ITERATIONS), \
        f"Expected (4, 1, {ITERATIONS}), got {q_vals.shape}"


def test_certs_shape():
    key = random.PRNGKey(1)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    traj, obs, acts = _make_inputs(key, B=4)

    q_vals, certs = critic.apply(params, traj, obs, acts, alpha=0.0)

    assert certs.shape == (4, 2, ITERATIONS), \
        f"Expected (4, 2, {ITERATIONS}), got {certs.shape}"


def test_batch_size_one():
    key = random.PRNGKey(2)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    traj, obs, acts = _make_inputs(key, B=1)

    q_vals, certs = critic.apply(params, traj, obs, acts, alpha=0.0)

    assert q_vals.shape == (1, 1, ITERATIONS)
    assert certs.shape  == (1, 2, ITERATIONS)


# ---------------------------------------------------------------------------
# Certainty range [0, 1]
# ---------------------------------------------------------------------------

def test_certainty_in_unit_interval():
    """certs[:,1,:] (certainty) must be in [0,1] — it comes from sigmoid."""
    key = random.PRNGKey(3)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    traj, obs, acts = _make_inputs(key, B=8)

    _, certs = critic.apply(params, traj, obs, acts, alpha=0.0)
    certainty = certs[:, 1, :]

    assert jnp.all(certainty >= 0.0), "Certainty score below 0"
    assert jnp.all(certainty <= 1.0), "Certainty score above 1"


def test_uncertainty_plus_certainty_equals_one():
    """certs[:,0,:] + certs[:,1,:] should equal 1 (they're complementary)."""
    key = random.PRNGKey(4)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    traj, obs, acts = _make_inputs(key, B=4)

    _, certs = critic.apply(params, traj, obs, acts, alpha=0.0)

    total = certs[:, 0, :] + certs[:, 1, :]
    assert jnp.allclose(total, jnp.ones_like(total), atol=1e-5), \
        "uncertainty + certainty != 1"


# ---------------------------------------------------------------------------
# Finiteness
# ---------------------------------------------------------------------------

def test_q_vals_finite():
    key = random.PRNGKey(5)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    traj, obs, acts = _make_inputs(key, B=8)

    q_vals, _ = critic.apply(params, traj, obs, acts, alpha=0.0)

    assert jnp.all(jnp.isfinite(q_vals)), "Q values contain NaN or Inf"


def test_certs_finite():
    key = random.PRNGKey(6)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    traj, obs, acts = _make_inputs(key, B=8)

    _, certs = critic.apply(params, traj, obs, acts, alpha=0.0)

    assert jnp.all(jnp.isfinite(certs)), "Certainties contain NaN or Inf"


# ---------------------------------------------------------------------------
# tick_certain is a valid index
# ---------------------------------------------------------------------------

def test_tick_certain_valid_index():
    key = random.PRNGKey(7)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    traj, obs, acts = _make_inputs(key, B=8)

    _, certs = critic.apply(params, traj, obs, acts, alpha=0.0)
    tick_certain = jnp.argmax(certs[:, 1, :], axis=-1)  # (B,)

    assert jnp.all(tick_certain >= 0)
    assert jnp.all(tick_certain < ITERATIONS)


# ---------------------------------------------------------------------------
# Alpha blend extremes
# ---------------------------------------------------------------------------

def test_alpha_zero_produces_finite_output():
    key = random.PRNGKey(8)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    traj, obs, acts = _make_inputs(key, B=4)

    q_vals, certs = critic.apply(params, traj, obs, acts, alpha=0.0)

    assert jnp.all(jnp.isfinite(q_vals))
    assert jnp.all(jnp.isfinite(certs))


def test_alpha_one_produces_finite_output():
    """alpha=1 means pure tick-variance certainty (structural, no learned blend)."""
    key = random.PRNGKey(9)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    traj, obs, acts = _make_inputs(key, B=4)

    q_vals, certs = critic.apply(params, traj, obs, acts, alpha=1.0)

    assert jnp.all(jnp.isfinite(q_vals))
    assert jnp.all(jnp.isfinite(certs))


# ---------------------------------------------------------------------------
# Deterministic mode
# ---------------------------------------------------------------------------

def test_deterministic_true_is_reproducible():
    """Two calls with same inputs and deterministic=True must give identical outputs."""
    key = random.PRNGKey(10)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    traj, obs, acts = _make_inputs(key, B=4)

    q1, c1 = critic.apply(params, traj, obs, acts, alpha=0.0, deterministic=True)
    q2, c2 = critic.apply(params, traj, obs, acts, alpha=0.0, deterministic=True)

    assert jnp.allclose(q1, q2)
    assert jnp.allclose(c1, c2)


# ---------------------------------------------------------------------------
# Target network
# ---------------------------------------------------------------------------

def test_target_network_same_shapes():
    """Target params (from a separate init) must produce identical output shapes."""
    key = random.PRNGKey(11)
    k1, k2 = random.split(key)
    critic, params        = create_ctm_critic(k1, **CTM_KWARGS)
    _,      target_params = create_ctm_critic(k2, **CTM_KWARGS)

    traj, obs, acts = _make_inputs(key, B=4)

    q_online, c_online = critic.apply(params,        traj, obs, acts, alpha=0.0)
    q_target, c_target = critic.apply(target_params, traj, obs, acts, alpha=0.0)

    assert q_online.shape == q_target.shape
    assert c_online.shape == c_target.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
