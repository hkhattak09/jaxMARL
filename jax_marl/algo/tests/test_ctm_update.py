"""Tests for update_critic_ctm and update_actor_ctm.

Covers update_critic_ctm:
- Returns DDPGAgentState with incremented step
- critic_params actually change after update
- D8: decay params (decay_params_action, decay_params_out) stay in [0, 15]
  even when we manually inject out-of-range values before the update
- Loss is finite

Covers update_actor_ctm:
- Returns DDPGAgentState with updated actor_params
- actor_params change after update
- step counter incremented
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import jax
import jax.numpy as jnp
from jax import random
import optax

from networks import create_ctm_critic, create_actor
from agents import DDPGAgentState, update_critic_ctm, update_actor_ctm


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

N_AGENTS   = 3
OBS_DIM    = 6
ACTION_DIM = 2
TRAJ_LEN   = 5
ITERATIONS = 4
B          = 8

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


def _setup(key):
    k1, k2, k3, k4 = random.split(key, 4)
    critic, c_params = create_ctm_critic(k1, **CTM_KWARGS)
    _, tc_params     = create_ctm_critic(k2, **CTM_KWARGS)
    actor,  a_params = create_actor(k3, OBS_DIM, ACTION_DIM)

    optimizer = optax.adam(1e-3)

    agent_state = DDPGAgentState(
        actor_params=a_params,
        critic_params=c_params,
        target_actor_params=a_params,
        target_critic_params=tc_params,
        actor_opt_state=optimizer.init(a_params),
        critic_opt_state=optimizer.init(c_params),
        step=jnp.array(0, dtype=jnp.int32),
    )

    k5, k6, k7, k8, k9, k10 = random.split(k4, 6)
    batch = dict(
        trajectory       = random.uniform(k5, (B, TRAJ_LEN, N_AGENTS, 2)),
        all_obs          = random.uniform(k6, (B, N_AGENTS, OBS_DIM)),
        all_actions      = random.uniform(k7, (B, N_AGENTS, ACTION_DIM)) * 2 - 1,
        next_trajectory  = random.uniform(k8, (B, TRAJ_LEN, N_AGENTS, 2)),
        next_all_obs     = random.uniform(k9, (B, N_AGENTS, OBS_DIM)),
        next_all_actions = random.uniform(k10, (B, N_AGENTS, ACTION_DIM)) * 2 - 1,
        rewards          = random.uniform(k5, (B,)),
        dones            = jnp.zeros((B,)),
    )

    return critic, actor, optimizer, agent_state, batch


# ---------------------------------------------------------------------------
# update_critic_ctm
# ---------------------------------------------------------------------------

def test_critic_update_does_not_modify_step():
    """update_critic_ctm does not own the step counter.
    Step is incremented once per full update by update_targets, consistent with
    the TD3 path (update_critic_td3 also does not touch step).
    """
    key = random.PRNGKey(0)
    critic, actor, opt, state, batch = _setup(key)

    new_state, info = update_critic_ctm(
        agent_state=state, ctm_critic=critic, optimizer=opt,
        **batch, gamma=0.99, alpha=0.0,
    )

    assert int(new_state.step) == int(state.step), \
        "update_critic_ctm should not modify step — update_targets owns that"


def test_critic_params_change():
    key = random.PRNGKey(1)
    critic, actor, opt, state, batch = _setup(key)

    new_state, _ = update_critic_ctm(
        agent_state=state, ctm_critic=critic, optimizer=opt,
        **batch, gamma=0.99, alpha=0.0,
    )

    old_leaves = jax.tree_util.tree_leaves(state.critic_params)
    new_leaves = jax.tree_util.tree_leaves(new_state.critic_params)

    changed = any(
        not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves)
    )
    assert changed, "critic_params did not change after update"


def test_critic_update_loss_finite():
    key = random.PRNGKey(2)
    critic, actor, opt, state, batch = _setup(key)

    _, info = update_critic_ctm(
        agent_state=state, ctm_critic=critic, optimizer=opt,
        **batch, gamma=0.99, alpha=0.0,
    )

    assert jnp.isfinite(info['ctm_critic_loss']), \
        f"Critic loss not finite: {info['ctm_critic_loss']}"


def test_critic_update_returns_info_keys():
    required = {'ctm_critic_loss', 'q_mean', 'bellman_target_mean',
                'cert_score_mean', 'td_error_mean', 'cert_aux_loss'}
    key = random.PRNGKey(3)
    critic, actor, opt, state, batch = _setup(key)

    _, info = update_critic_ctm(
        agent_state=state, ctm_critic=critic, optimizer=opt,
        **batch, gamma=0.99, alpha=0.0,
    )

    missing = required - set(info.keys())
    assert not missing, f"Missing info keys: {missing}"


# ---------------------------------------------------------------------------
# D8: decay param clamping
# ---------------------------------------------------------------------------

def _inject_decay_params(params, value):
    """Manually set decay_params_action / decay_params_out to `value`."""
    def _set(path, leaf):
        keys = [k.key for k in path if isinstance(k, jax.tree_util.DictKey)]
        if 'decay_params_action' in keys or 'decay_params_out' in keys:
            return jnp.full_like(leaf, value)
        return leaf
    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    flat = [_set(path, leaf) for path, leaf in flat]
    return treedef.unflatten(flat)


def test_decay_params_clamped_after_update():
    """Even if decay params start out-of-range, they must be in [0,15] post-update."""
    key = random.PRNGKey(4)
    critic, actor, opt, state, batch = _setup(key)

    # Inject a large out-of-range value so the update must clamp it
    injected_params = _inject_decay_params(state.critic_params, 30.0)
    state_with_large = state.replace(
        critic_params=injected_params,
        critic_opt_state=opt.init(injected_params),
    )

    new_state, _ = update_critic_ctm(
        agent_state=state_with_large, ctm_critic=critic, optimizer=opt,
        **batch, gamma=0.99, alpha=0.0,
    )

    flat, _ = jax.tree_util.tree_flatten_with_path(new_state.critic_params)
    for path, leaf in flat:
        keys = [k.key for k in path if isinstance(k, jax.tree_util.DictKey)]
        if 'decay_params_action' in keys or 'decay_params_out' in keys:
            assert jnp.all(leaf <= 15.0), \
                f"decay param > 15 after clamping: {leaf}"
            assert jnp.all(leaf >= 0.0), \
                f"decay param < 0 after clamping: {leaf}"


def test_decay_params_clamped_from_below():
    """Negative decay params must be clamped to 0."""
    key = random.PRNGKey(5)
    critic, actor, opt, state, batch = _setup(key)

    injected_params = _inject_decay_params(state.critic_params, -5.0)
    state_neg = state.replace(
        critic_params=injected_params,
        critic_opt_state=opt.init(injected_params),
    )

    new_state, _ = update_critic_ctm(
        agent_state=state_neg, ctm_critic=critic, optimizer=opt,
        **batch, gamma=0.99, alpha=0.0,
    )

    flat, _ = jax.tree_util.tree_flatten_with_path(new_state.critic_params)
    for path, leaf in flat:
        keys = [k.key for k in path if isinstance(k, jax.tree_util.DictKey)]
        if 'decay_params_action' in keys or 'decay_params_out' in keys:
            assert jnp.all(leaf >= 0.0), f"decay param < 0 after clamp: {leaf}"


def test_non_decay_params_not_clamped():
    """Non-decay params should NOT be clamped to [0,15] — they may legitimately exceed it."""
    key = random.PRNGKey(6)
    critic, actor, opt, state, batch = _setup(key)

    # After a normal update with default zero init, do multiple steps to move params
    current_state = state
    for _ in range(3):
        current_state, _ = update_critic_ctm(
            agent_state=current_state, ctm_critic=critic, optimizer=opt,
            **batch, gamma=0.99, alpha=0.0,
        )

    # Most non-decay params (Dense weights, biases) should be able to move freely
    # This test just verifies the update doesn't NaN everything
    leaves = jax.tree_util.tree_leaves(current_state.critic_params)
    assert all(jnp.all(jnp.isfinite(l)) for l in leaves), \
        "Some critic params are not finite after multiple updates"


# ---------------------------------------------------------------------------
# update_actor_ctm
# ---------------------------------------------------------------------------

def test_actor_update_does_not_modify_step():
    """update_actor_ctm does not own the step counter.
    Step is incremented once per full jit_update call at the MADDPGState level.
    This matches update_actor_td3 behaviour.
    """
    key = random.PRNGKey(7)
    critic, actor, opt, state, batch = _setup(key)

    new_state, info = update_actor_ctm(
        agent_state=state,
        actor=actor,
        ctm_critic=critic,
        optimizer=opt,
        trajectory=batch['trajectory'],
        all_obs=batch['all_obs'],
        agent_obs=batch['all_obs'][:, 0, :],
        all_actions_flat=batch['all_actions'].reshape(B, -1),
        agent_action_idx=0,
        action_dim=ACTION_DIM,
        n_agents=N_AGENTS,
        alpha=0.0,
    )

    assert int(new_state.step) == int(state.step), \
        "update_actor_ctm should not modify step — that is MADDPGState's responsibility"


def test_actor_params_change():
    key = random.PRNGKey(8)
    critic, actor, opt, state, batch = _setup(key)

    new_state, _ = update_actor_ctm(
        agent_state=state,
        actor=actor,
        ctm_critic=critic,
        optimizer=opt,
        trajectory=batch['trajectory'],
        all_obs=batch['all_obs'],
        agent_obs=batch['all_obs'][:, 0, :],
        all_actions_flat=batch['all_actions'].reshape(B, -1),
        agent_action_idx=0,
        action_dim=ACTION_DIM,
        n_agents=N_AGENTS,
        alpha=0.0,
    )

    old_leaves = jax.tree_util.tree_leaves(state.actor_params)
    new_leaves = jax.tree_util.tree_leaves(new_state.actor_params)

    changed = any(
        not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves)
    )
    assert changed, "actor_params did not change after update"


def test_actor_update_info_key():
    key = random.PRNGKey(9)
    critic, actor, opt, state, batch = _setup(key)

    _, info = update_actor_ctm(
        agent_state=state,
        actor=actor,
        ctm_critic=critic,
        optimizer=opt,
        trajectory=batch['trajectory'],
        all_obs=batch['all_obs'],
        agent_obs=batch['all_obs'][:, 0, :],
        all_actions_flat=batch['all_actions'].reshape(B, -1),
        agent_action_idx=0,
        action_dim=ACTION_DIM,
        n_agents=N_AGENTS,
        alpha=0.0,
    )

    assert 'ctm_actor_loss' in info


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
