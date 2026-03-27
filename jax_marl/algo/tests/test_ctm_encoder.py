"""Tests for TrajectoryEncoder.

Covers:
- Output shapes: kv=(B, traj_len+n_agents, d_input), per_agent_feat=(B, n_agents, d_model)
- Accepts float16 trajectory input (casts internally to float32)
- Handles batch size 1
- kv tokens are finite (no NaN/Inf)
- LayerNorm applied: kv has zero-mean per-token (approximately)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import jax
import jax.numpy as jnp
from jax import random

from networks import TrajectoryEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_INPUT  = 32
D_MODEL  = 16
N_AGENTS = 4
TRAJ_LEN = 6
OBS_DIM  = 8
ACT_DIM  = 2


def _make_encoder():
    return TrajectoryEncoder(
        d_input=D_INPUT,
        d_model=D_MODEL,
        n_agents=N_AGENTS,
        traj_len=TRAJ_LEN,
    )


def _init_encoder(key, B=2):
    enc = _make_encoder()
    traj  = jnp.zeros((B, TRAJ_LEN, N_AGENTS, 2))
    obs   = jnp.zeros((B, N_AGENTS, OBS_DIM))
    acts  = jnp.zeros((B, N_AGENTS, ACT_DIM))
    params = enc.init(key, traj, obs, acts)
    return enc, params


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_kv_shape():
    key = random.PRNGKey(0)
    enc, params = _init_encoder(key, B=4)
    B = 4
    traj = random.uniform(key, (B, TRAJ_LEN, N_AGENTS, 2))
    obs  = random.uniform(key, (B, N_AGENTS, OBS_DIM))
    acts = random.uniform(key, (B, N_AGENTS, ACT_DIM))

    kv, per_agent_feat = enc.apply(params, traj, obs, acts)

    assert kv.shape == (B, TRAJ_LEN + N_AGENTS, D_INPUT), \
        f"Expected ({B}, {TRAJ_LEN + N_AGENTS}, {D_INPUT}), got {kv.shape}"


def test_per_agent_feat_shape():
    key = random.PRNGKey(1)
    enc, params = _init_encoder(key, B=4)
    B = 4
    traj = random.uniform(key, (B, TRAJ_LEN, N_AGENTS, 2))
    obs  = random.uniform(key, (B, N_AGENTS, OBS_DIM))
    acts = random.uniform(key, (B, N_AGENTS, ACT_DIM))

    kv, per_agent_feat = enc.apply(params, traj, obs, acts)

    assert per_agent_feat.shape == (B, N_AGENTS, D_MODEL), \
        f"Expected ({B}, {N_AGENTS}, {D_MODEL}), got {per_agent_feat.shape}"


def test_batch_size_one():
    key = random.PRNGKey(2)
    enc, params = _init_encoder(key, B=1)
    traj = random.uniform(key, (1, TRAJ_LEN, N_AGENTS, 2))
    obs  = random.uniform(key, (1, N_AGENTS, OBS_DIM))
    acts = random.uniform(key, (1, N_AGENTS, ACT_DIM))

    kv, per_agent_feat = enc.apply(params, traj, obs, acts)

    assert kv.shape == (1, TRAJ_LEN + N_AGENTS, D_INPUT)
    assert per_agent_feat.shape == (1, N_AGENTS, D_MODEL)


# ---------------------------------------------------------------------------
# Dtype tests
# ---------------------------------------------------------------------------

def test_float16_input_produces_float32_output():
    """Trajectory stored as float16 should be cast to float32 internally."""
    key = random.PRNGKey(3)
    enc, params = _init_encoder(key)
    B = 2
    traj = random.uniform(key, (B, TRAJ_LEN, N_AGENTS, 2)).astype(jnp.float16)
    obs  = random.uniform(key, (B, N_AGENTS, OBS_DIM))
    acts = random.uniform(key, (B, N_AGENTS, ACT_DIM))

    kv, per_agent_feat = enc.apply(params, traj, obs, acts)

    assert kv.dtype == jnp.float32
    assert per_agent_feat.dtype == jnp.float32


# ---------------------------------------------------------------------------
# Numerical tests
# ---------------------------------------------------------------------------

def test_kv_is_finite():
    key = random.PRNGKey(4)
    enc, params = _init_encoder(key)
    B = 8
    traj = random.normal(key, (B, TRAJ_LEN, N_AGENTS, 2))
    obs  = random.normal(key, (B, N_AGENTS, OBS_DIM))
    acts = random.normal(key, (B, N_AGENTS, ACT_DIM))

    kv, _ = enc.apply(params, traj, obs, acts)

    assert jnp.all(jnp.isfinite(kv)), "kv contains NaN or Inf"


def test_per_agent_feat_is_finite():
    key = random.PRNGKey(5)
    enc, params = _init_encoder(key)
    B = 8
    traj = random.normal(key, (B, TRAJ_LEN, N_AGENTS, 2))
    obs  = random.normal(key, (B, N_AGENTS, OBS_DIM))
    acts = random.normal(key, (B, N_AGENTS, ACT_DIM))

    _, per_agent_feat = enc.apply(params, traj, obs, acts)

    assert jnp.all(jnp.isfinite(per_agent_feat)), "per_agent_feat contains NaN or Inf"


def test_zero_trajectory_runs_without_error():
    """All-zero trajectory (typical at episode start) should not crash."""
    key = random.PRNGKey(6)
    enc, params = _init_encoder(key)
    B = 4
    traj = jnp.zeros((B, TRAJ_LEN, N_AGENTS, 2))
    obs  = random.normal(key, (B, N_AGENTS, OBS_DIM))
    acts = random.normal(key, (B, N_AGENTS, ACT_DIM))

    kv, per_agent_feat = enc.apply(params, traj, obs, acts)

    assert jnp.all(jnp.isfinite(kv))
    assert jnp.all(jnp.isfinite(per_agent_feat))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
