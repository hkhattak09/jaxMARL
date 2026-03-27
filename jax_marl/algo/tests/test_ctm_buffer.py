"""Tests for CTM trajectory storage in ReplayBuffer.

Covers:
- Buffer allocates trajectory arrays in float16 when store_trajectories=True
- add() stores trajectories with correct dtype and shape
- sample() returns trajectories in BatchTransition with correct shape
- store_trajectories=False leaves trajectory fields as None
- Circular overwrite: new trajectory replaces old at correct slot
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import jax
import jax.numpy as jnp
from jax import random

from buffers import ReplayBuffer, Transition


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CAPACITY  = 64
N_AGENTS  = 4
OBS_DIM   = 6
ACTION_DIM = 2
TRAJ_LEN  = 5
BATCH     = 16


def _make_buffer(store=True, traj_dtype=jnp.float16):
    return ReplayBuffer(
        capacity=CAPACITY,
        n_agents=N_AGENTS,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        use_global_state=False,
        traj_len=TRAJ_LEN,
        store_trajectories=store,
        trajectory_dtype=traj_dtype,
    )


def _rand_transition(key, with_traj=True):
    k1, k2, k3, k4, k5, k6 = random.split(key, 6)
    traj      = random.uniform(k6, (TRAJ_LEN, N_AGENTS, 2)) if with_traj else None
    next_traj = random.uniform(k6, (TRAJ_LEN, N_AGENTS, 2)) if with_traj else None
    return Transition(
        obs=random.uniform(k1, (N_AGENTS, OBS_DIM)),
        actions=random.uniform(k2, (N_AGENTS, ACTION_DIM)),
        rewards=random.uniform(k3, (N_AGENTS,)),
        next_obs=random.uniform(k4, (N_AGENTS, OBS_DIM)),
        dones=jnp.zeros((N_AGENTS,)),
        trajectory=traj,
        next_trajectory=next_traj,
    )


# ---------------------------------------------------------------------------
# Allocation tests
# ---------------------------------------------------------------------------

def test_buffer_allocates_trajectory_arrays():
    buf = _make_buffer(store=True)
    state = buf.init()
    assert state.trajectory is not None
    assert state.next_trajectory is not None
    assert state.trajectory.shape  == (CAPACITY, TRAJ_LEN, N_AGENTS, 2)
    assert state.next_trajectory.shape == (CAPACITY, TRAJ_LEN, N_AGENTS, 2)


def test_buffer_trajectory_dtype_is_float16():
    buf = _make_buffer(store=True, traj_dtype=jnp.float16)
    state = buf.init()
    assert state.trajectory.dtype == jnp.float16
    assert state.next_trajectory.dtype == jnp.float16


def test_buffer_no_trajectory_when_disabled():
    buf = _make_buffer(store=False)
    state = buf.init()
    assert state.trajectory is None
    assert state.next_trajectory is None


# ---------------------------------------------------------------------------
# add() tests
# ---------------------------------------------------------------------------

def test_add_stores_trajectory():
    buf = _make_buffer(store=True)
    state = buf.init()
    key = random.PRNGKey(0)
    t = _rand_transition(key)

    state = buf.add(state, t)

    assert state.size == 1
    stored = state.trajectory[0]          # (traj_len, n_agents, 2)
    assert stored.shape == (TRAJ_LEN, N_AGENTS, 2)
    assert stored.dtype == jnp.float16


def test_add_trajectory_values_match():
    """Values stored should round-trip through float16 cast."""
    buf = _make_buffer(store=True)
    state = buf.init()
    key = random.PRNGKey(1)
    t = _rand_transition(key)

    state = buf.add(state, t)

    expected = t.trajectory.astype(jnp.float16)
    stored = state.trajectory[0]
    assert jnp.allclose(stored, expected, atol=1e-3)


def test_add_increments_position():
    buf = _make_buffer(store=True)
    state = buf.init()
    key = random.PRNGKey(2)

    for i in range(3):
        key, k = random.split(key)
        state = buf.add(state, _rand_transition(k))

    assert int(state.size) == 3
    assert int(state.position) == 3


# ---------------------------------------------------------------------------
# sample() tests
# ---------------------------------------------------------------------------

def test_sample_returns_trajectories():
    buf = _make_buffer(store=True)
    state = buf.init()
    key = random.PRNGKey(3)

    for _ in range(BATCH * 2):
        key, k = random.split(key)
        state = buf.add(state, _rand_transition(k))

    key, sk = random.split(key)
    batch = buf.sample(state, sk, BATCH)

    assert batch.trajectory is not None
    assert batch.next_trajectory is not None
    assert batch.trajectory.shape      == (BATCH, TRAJ_LEN, N_AGENTS, 2)
    assert batch.next_trajectory.shape == (BATCH, TRAJ_LEN, N_AGENTS, 2)


def test_sample_trajectory_dtype_is_float16():
    buf = _make_buffer(store=True)
    state = buf.init()
    key = random.PRNGKey(4)

    for _ in range(BATCH * 2):
        key, k = random.split(key)
        state = buf.add(state, _rand_transition(k))

    key, sk = random.split(key)
    batch = buf.sample(state, sk, BATCH)

    assert batch.trajectory.dtype == jnp.float16


def test_sample_no_trajectory_when_disabled():
    buf = _make_buffer(store=False)
    state = buf.init()
    key = random.PRNGKey(5)

    for _ in range(BATCH * 2):
        key, k = random.split(key)
        state = buf.add(state, _rand_transition(k, with_traj=False))

    key, sk = random.split(key)
    batch = buf.sample(state, sk, BATCH)

    assert batch.trajectory is None
    assert batch.next_trajectory is None


# ---------------------------------------------------------------------------
# Circular buffer tests
# ---------------------------------------------------------------------------

def test_circular_overwrite_trajectory():
    """Writing CAPACITY+1 transitions should overwrite slot 0."""
    buf = _make_buffer(store=True)
    state = buf.init()
    key = random.PRNGKey(6)

    sentinel = jnp.ones((TRAJ_LEN, N_AGENTS, 2), dtype=jnp.float32) * 99.0
    # Write sentinel at slot 0
    t0 = _rand_transition(key)
    t0 = t0._replace(trajectory=sentinel)
    state = buf.add(state, t0)

    # Fill remaining CAPACITY-1 slots + one more to wrap
    for _ in range(CAPACITY):
        key, k = random.split(key)
        state = buf.add(state, _rand_transition(k))

    # Slot 0 should now hold the overwrite, not the sentinel
    stored = state.trajectory[0].astype(jnp.float32)
    assert not jnp.allclose(stored, sentinel, atol=1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
