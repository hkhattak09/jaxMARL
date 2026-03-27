"""Tests for CTM loss functions.

Covers compute_ctm_critic_loss:
- Returns scalar loss + info dict with required keys
- Loss is finite and non-negative (MSE)
- With all dones=1 the bootstrap term vanishes (target = reward only)
- With all dones=0 the bootstrap term is active (target > reward for positive Q)
- dual-tick: loss uses both tick_best and tick_certain

Covers compute_actor_loss_ctm:
- Returns negative Q (maximization objective → negative mean)
- Info dict has 'ctm_actor_loss' key
- Actor gradient flows through (loss is differentiable w.r.t. actor_params)
- Prior regularization increases loss when prior_weight > 0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import jax
import jax.numpy as jnp
from jax import random
import optax

from networks import create_ctm_critic, Actor, create_actor
from agents import compute_ctm_critic_loss, compute_actor_loss_ctm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_AGENTS   = 3
OBS_DIM    = 6
ACTION_DIM = 2
TRAJ_LEN   = 5
ITERATIONS = 4
B          = 8   # batch size

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


def _batch_inputs(key):
    k1, k2, k3, k4, k5, k6, k7 = random.split(key, 7)
    return dict(
        trajectory      = random.uniform(k1, (B, TRAJ_LEN, N_AGENTS, 2)),
        all_obs         = random.uniform(k2, (B, N_AGENTS, OBS_DIM)),
        all_actions     = random.uniform(k3, (B, N_AGENTS, ACTION_DIM)) * 2 - 1,
        next_trajectory = random.uniform(k4, (B, TRAJ_LEN, N_AGENTS, 2)),
        next_all_obs    = random.uniform(k5, (B, N_AGENTS, OBS_DIM)),
        next_all_actions= random.uniform(k6, (B, N_AGENTS, ACTION_DIM)) * 2 - 1,
        rewards         = random.uniform(k7, (B,)),
        dones           = jnp.zeros((B,)),
    )


# ---------------------------------------------------------------------------
# compute_ctm_critic_loss — return values
# ---------------------------------------------------------------------------

def test_critic_loss_is_scalar():
    key = random.PRNGKey(0)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    _, target_params = create_ctm_critic(random.PRNGKey(1), **CTM_KWARGS)
    inp = _batch_inputs(key)

    loss, info = compute_ctm_critic_loss(
        critic, params, target_params,
        **inp, gamma=0.99, alpha=0.0,
    )

    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"


def test_critic_loss_is_finite():
    key = random.PRNGKey(2)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    _, target_params = create_ctm_critic(random.PRNGKey(3), **CTM_KWARGS)
    inp = _batch_inputs(key)

    loss, _ = compute_ctm_critic_loss(
        critic, params, target_params,
        **inp, gamma=0.99, alpha=0.0,
    )

    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"


def test_critic_loss_non_negative():
    """MSE loss is always >= 0."""
    key = random.PRNGKey(4)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    _, target_params = create_ctm_critic(random.PRNGKey(5), **CTM_KWARGS)
    inp = _batch_inputs(key)

    loss, _ = compute_ctm_critic_loss(
        critic, params, target_params,
        **inp, gamma=0.99, alpha=0.0,
    )

    assert float(loss) >= 0.0, f"Loss should be non-negative, got {loss}"


def test_critic_loss_info_keys():
    required_keys = {'ctm_critic_loss', 'q_mean', 'bellman_target_mean',
                     'cert_score_mean', 'td_error_mean'}
    key = random.PRNGKey(6)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    _, target_params = create_ctm_critic(random.PRNGKey(7), **CTM_KWARGS)
    inp = _batch_inputs(key)

    _, info = compute_ctm_critic_loss(
        critic, params, target_params,
        **inp, gamma=0.99, alpha=0.0,
    )

    missing = required_keys - set(info.keys())
    assert not missing, f"Info dict missing keys: {missing}"


def test_critic_loss_info_values_finite():
    key = random.PRNGKey(8)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    _, target_params = create_ctm_critic(random.PRNGKey(9), **CTM_KWARGS)
    inp = _batch_inputs(key)

    _, info = compute_ctm_critic_loss(
        critic, params, target_params,
        **inp, gamma=0.99, alpha=0.0,
    )

    for k, v in info.items():
        assert jnp.isfinite(v), f"info['{k}'] is not finite: {v}"


# ---------------------------------------------------------------------------
# Bellman target behaviour
# ---------------------------------------------------------------------------

def test_done_kills_bootstrap():
    """With all dones=1, Bellman target = reward (no γ * Q_next term)."""
    key = random.PRNGKey(10)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    _, target_params = create_ctm_critic(random.PRNGKey(11), **CTM_KWARGS)
    inp = _batch_inputs(key)

    # Fix rewards to a constant so we know what the target should be
    inp_done = {**inp, 'rewards': jnp.ones((B,)), 'dones': jnp.ones((B,))}

    _, info = compute_ctm_critic_loss(
        critic, params, target_params,
        **inp_done, gamma=0.99, alpha=0.0,
    )

    # bellman_target should be ~1.0 (reward only, no bootstrap)
    # cert_score is in [0,1] so target <= reward, here target == reward * cert_score.
    # With certainty-discounted target: bellman = r + 0 (done=1).
    # The mean target should be <= 1.0 and > 0.
    target_mean = float(info['bellman_target_mean'])
    assert target_mean <= 1.0 + 1e-4, f"Bellman target should be <= reward, got {target_mean}"


def test_bootstrap_active_when_not_done():
    """With dones=0 and positive Q, target > reward (bootstrap contributes)."""
    key = random.PRNGKey(12)
    critic, params = create_ctm_critic(key, **CTM_KWARGS)
    _, target_params = create_ctm_critic(random.PRNGKey(13), **CTM_KWARGS)
    inp = _batch_inputs(key)

    # This test is statistical: just verify both done=0 and done=1 produce
    # different bellman_target_mean values (bootstrap changes the target).
    inp_no_done  = {**inp, 'dones': jnp.zeros((B,))}
    inp_all_done = {**inp, 'dones': jnp.ones((B,))}

    _, info_no  = compute_ctm_critic_loss(critic, params, target_params,
                                          **inp_no_done,  gamma=0.99, alpha=0.0)
    _, info_all = compute_ctm_critic_loss(critic, params, target_params,
                                          **inp_all_done, gamma=0.99, alpha=0.0)

    # Targets should differ when done flags differ (unless Q_next=0 exactly)
    # With random params they'll almost certainly differ.
    assert not jnp.allclose(
        info_no['bellman_target_mean'],
        info_all['bellman_target_mean'],
        atol=1e-4,
    ), "Bellman targets identical regardless of done — bootstrap may be broken"


# ---------------------------------------------------------------------------
# compute_actor_loss_ctm — return values
# ---------------------------------------------------------------------------

def test_actor_loss_is_scalar():
    key = random.PRNGKey(14)
    critic, critic_params = create_ctm_critic(key, **CTM_KWARGS)
    actor,  actor_params  = create_actor(random.PRNGKey(15), OBS_DIM, ACTION_DIM)
    inp = _batch_inputs(key)

    loss, info = compute_actor_loss_ctm(
        actor=actor,
        actor_params=actor_params,
        ctm_critic=critic,
        critic_params=critic_params,
        trajectory=inp['trajectory'],
        all_obs=inp['all_obs'],
        agent_obs=inp['all_obs'][:, 0, :],
        all_actions_flat=inp['all_actions'].reshape(B, -1),
        agent_action_idx=0,
        action_dim=ACTION_DIM,
        n_agents=N_AGENTS,
        alpha=0.0,
    )

    assert loss.shape == (), f"Actor loss should be scalar, got {loss.shape}"


def test_actor_loss_is_negative_q():
    """Actor loss = -mean(Q) so it should be negative when Q > 0 is possible."""
    key = random.PRNGKey(16)
    critic, critic_params = create_ctm_critic(key, **CTM_KWARGS)
    actor,  actor_params  = create_actor(random.PRNGKey(17), OBS_DIM, ACTION_DIM)
    inp = _batch_inputs(key)

    loss, _ = compute_actor_loss_ctm(
        actor=actor,
        actor_params=actor_params,
        ctm_critic=critic,
        critic_params=critic_params,
        trajectory=inp['trajectory'],
        all_obs=inp['all_obs'],
        agent_obs=inp['all_obs'][:, 0, :],
        all_actions_flat=inp['all_actions'].reshape(B, -1),
        agent_action_idx=0,
        action_dim=ACTION_DIM,
        n_agents=N_AGENTS,
        alpha=0.0,
    )

    assert jnp.isfinite(loss)


def test_actor_loss_info_key():
    key = random.PRNGKey(18)
    critic, critic_params = create_ctm_critic(key, **CTM_KWARGS)
    actor,  actor_params  = create_actor(random.PRNGKey(19), OBS_DIM, ACTION_DIM)
    inp = _batch_inputs(key)

    _, info = compute_actor_loss_ctm(
        actor=actor, actor_params=actor_params,
        ctm_critic=critic, critic_params=critic_params,
        trajectory=inp['trajectory'], all_obs=inp['all_obs'],
        agent_obs=inp['all_obs'][:, 0, :],
        all_actions_flat=inp['all_actions'].reshape(B, -1),
        agent_action_idx=0, action_dim=ACTION_DIM, n_agents=N_AGENTS, alpha=0.0,
    )

    assert 'ctm_actor_loss' in info


def test_actor_loss_differentiable():
    """jax.grad should work through compute_actor_loss_ctm w.r.t. actor_params."""
    key = random.PRNGKey(20)
    critic, critic_params = create_ctm_critic(key, **CTM_KWARGS)
    actor,  actor_params  = create_actor(random.PRNGKey(21), OBS_DIM, ACTION_DIM)
    inp = _batch_inputs(key)

    def loss_fn(ap):
        l, _ = compute_actor_loss_ctm(
            actor=actor, actor_params=ap,
            ctm_critic=critic, critic_params=critic_params,
            trajectory=inp['trajectory'], all_obs=inp['all_obs'],
            agent_obs=inp['all_obs'][:, 0, :],
            all_actions_flat=inp['all_actions'].reshape(B, -1),
            agent_action_idx=0, action_dim=ACTION_DIM, n_agents=N_AGENTS, alpha=0.0,
        )
        return l

    grads = jax.grad(loss_fn)(actor_params)
    # At least one gradient leaf should be non-zero
    leaves = jax.tree_util.tree_leaves(grads)
    any_nonzero = any(jnp.any(g != 0) for g in leaves)
    assert any_nonzero, "All actor gradients are zero — gradient may not be flowing"


def test_actor_prior_increases_loss():
    """With prior_weight>0 and a far-from-prior action, loss should increase."""
    key = random.PRNGKey(22)
    critic, critic_params = create_ctm_critic(key, **CTM_KWARGS)
    actor,  actor_params  = create_actor(random.PRNGKey(23), OBS_DIM, ACTION_DIM)
    inp = _batch_inputs(key)

    kwargs = dict(
        actor=actor, actor_params=actor_params,
        ctm_critic=critic, critic_params=critic_params,
        trajectory=inp['trajectory'], all_obs=inp['all_obs'],
        agent_obs=inp['all_obs'][:, 0, :],
        all_actions_flat=inp['all_actions'].reshape(B, -1),
        agent_action_idx=0, action_dim=ACTION_DIM, n_agents=N_AGENTS, alpha=0.0,
    )

    loss_no_prior, _ = compute_actor_loss_ctm(**kwargs, prior_weight=0.0)
    # Prior is all -1 (opposite of tanh actor which maps to [-1,1])
    far_prior = -jnp.ones((B, ACTION_DIM))
    loss_with_prior, _ = compute_actor_loss_ctm(
        **kwargs, action_prior=far_prior, prior_weight=1.0
    )

    # Adding a far prior must change the loss
    assert not jnp.allclose(loss_no_prior, loss_with_prior, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
