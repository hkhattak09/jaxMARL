"""Tests for _reorder_trajectory (D7 — reorder at store time).

Covers:
- Identity: traj_idx at last slot → no-op
- Correct chronological reorder from arbitrary write position
- All positions produce the same chronological output given the same data
- vmapped version (as used in store_transitions_batched) works over batch
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # jax_marl/

import pytest
import jax
import jax.numpy as jnp
from jax import random

from jax_marl.algo.maddpg import _reorder_trajectory


TRAJ_LEN = 6
N_AGENTS = 3


def _circular_buffer_state(data_sequence, traj_len):
    """Simulate writing data_sequence into a circular buffer of length traj_len.

    Returns (raw_buf, write_idx) where write_idx is the slot that will be
    written NEXT (i.e. the oldest slot in a full buffer).
    """
    buf = jnp.zeros((traj_len, N_AGENTS, 2))
    for i, val in enumerate(data_sequence):
        slot = i % traj_len
        buf = buf.at[slot].set(val)
    write_idx = len(data_sequence) % traj_len
    # last written slot = (write_idx - 1) % traj_len = traj_idx for reorder
    traj_idx = (write_idx - 1) % traj_len
    return buf, traj_idx


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_reorder_identity_when_last_slot_written():
    """If traj_idx == traj_len-1, the buffer is already chronological."""
    data = [jnp.full((N_AGENTS, 2), float(i)) for i in range(TRAJ_LEN)]
    raw, traj_idx = _circular_buffer_state(data, TRAJ_LEN)

    assert int(traj_idx) == TRAJ_LEN - 1
    reordered = _reorder_trajectory(raw, traj_idx, TRAJ_LEN)

    for t in range(TRAJ_LEN):
        assert jnp.allclose(reordered[t], data[t])


def test_reorder_mid_buffer():
    """Arbitrary write position should produce chronological order."""
    # Write 9 items into a buffer of length 6 → wrap once, traj_idx=2
    data = [jnp.full((N_AGENTS, 2), float(i)) for i in range(9)]
    raw, traj_idx = _circular_buffer_state(data, TRAJ_LEN)

    # The last 6 items in chronological order are data[3..8]
    expected = [data[i] for i in range(3, 9)]

    reordered = _reorder_trajectory(raw, traj_idx, TRAJ_LEN)

    for t in range(TRAJ_LEN):
        assert jnp.allclose(reordered[t], expected[t]), f"step {t} mismatch"


def test_reorder_full_wrap():
    """Write exactly 2*TRAJ_LEN items; oldest slot should be next write position."""
    data = [jnp.full((N_AGENTS, 2), float(i)) for i in range(2 * TRAJ_LEN)]
    raw, traj_idx = _circular_buffer_state(data, TRAJ_LEN)

    expected = [data[i] for i in range(TRAJ_LEN, 2 * TRAJ_LEN)]
    reordered = _reorder_trajectory(raw, traj_idx, TRAJ_LEN)

    for t in range(TRAJ_LEN):
        assert jnp.allclose(reordered[t], expected[t]), f"step {t} mismatch"


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_reorder_output_shape():
    raw = random.uniform(random.PRNGKey(0), (TRAJ_LEN, N_AGENTS, 2))
    out = _reorder_trajectory(raw, jnp.array(3), TRAJ_LEN)
    assert out.shape == (TRAJ_LEN, N_AGENTS, 2)


# ---------------------------------------------------------------------------
# All starting positions produce same chronological content
# ---------------------------------------------------------------------------

def test_reorder_all_offsets_consistent():
    """For a fixed data sequence, reordering from any write position gives same last TRAJ_LEN items."""
    full_sequence = [jnp.full((N_AGENTS, 2), float(i)) for i in range(3 * TRAJ_LEN)]

    for write_count in range(TRAJ_LEN, 3 * TRAJ_LEN):
        raw, traj_idx = _circular_buffer_state(full_sequence[:write_count], TRAJ_LEN)
        expected = [full_sequence[write_count - TRAJ_LEN + t] for t in range(TRAJ_LEN)]
        reordered = _reorder_trajectory(raw, traj_idx, TRAJ_LEN)
        for t in range(TRAJ_LEN):
            assert jnp.allclose(reordered[t], expected[t]), \
                f"write_count={write_count}, t={t}: got {reordered[t]}, want {expected[t]}"


# ---------------------------------------------------------------------------
# vmap over batch (mirrors store_transitions_batched usage)
# ---------------------------------------------------------------------------

def test_reorder_vmapped_over_batch():
    B = 8
    raw_batch = random.uniform(random.PRNGKey(1), (B, TRAJ_LEN, N_AGENTS, 2))
    idxs = jnp.arange(B) % TRAJ_LEN  # different traj_idx per env

    reordered_batch = jax.vmap(
        lambda r, idx: _reorder_trajectory(r, idx, TRAJ_LEN)
    )(raw_batch, idxs)

    assert reordered_batch.shape == (B, TRAJ_LEN, N_AGENTS, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
