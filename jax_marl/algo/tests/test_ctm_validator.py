"""Tests for CTMValidator (ctm_validation.py).

Pure-Python — no JAX required. Fast to run.
"""

import sys
from pathlib import Path
import pytest

# ctm_validation.py lives in jax_marl/train/
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "train"))
from ctm_validation import CTMValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Cfg:
    """Minimal config stand-in."""
    def __init__(self, enabled=True, anneal_steps=1000, alpha_final=0.0):
        self.use_ctm_critic        = enabled
        self.ctm_alpha_anneal_steps = anneal_steps
        self.ctm_alpha_final        = alpha_final


def _good_info(**overrides):
    """Nominal healthy info dict (all Python floats, updated=True)."""
    base = {
        'updated':             True,
        'ctm_q_mean':          1.0,
        'ctm_bellman_target':  1.5,
        'ctm_cert_score':      0.5,
        'ctm_td_error':        0.1,
        'ctm_cert_aux_loss':   0.3,
        'ctm_tick_diversity':  0.4,
    }
    base.update(overrides)
    return base


def _pump(validator, n, **info_overrides):
    """Feed n healthy updates through the validator."""
    for i in range(n):
        validator.check(_good_info(**info_overrides), episode=i)


# ---------------------------------------------------------------------------
# 1. Disabled / skipped
# ---------------------------------------------------------------------------

def test_disabled_returns_empty():
    v = CTMValidator(_Cfg(enabled=False))
    warnings = v.check(_good_info(), episode=0)
    assert warnings == []


def test_not_updated_returns_empty():
    v = CTMValidator(_Cfg())
    warnings = v.check(_good_info(updated=False), episode=0)
    assert warnings == []


def test_not_updated_does_not_increment_count():
    v = CTMValidator(_Cfg())
    v.check(_good_info(updated=False))
    v.check(_good_info(updated=False))
    assert v._update_count == 0


# ---------------------------------------------------------------------------
# 2. Healthy data produces no warnings
# ---------------------------------------------------------------------------

def test_healthy_no_warnings():
    v = CTMValidator(_Cfg())
    # warm up rolling window past all thresholds
    for _ in range(30):
        warnings = v.check(_good_info())
    assert warnings == []


# ---------------------------------------------------------------------------
# 3. Finiteness — CRITICAL
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,label", [
    ('ctm_q_mean',        'Q-mean'),
    ('ctm_bellman_target','Bellman'),
    ('ctm_td_error',      'TD error'),
    ('ctm_cert_aux_loss', 'cert_aux_loss'),
])
def test_nan_triggers_critical(field, label):
    v = CTMValidator(_Cfg())
    warnings = v.check(_good_info(**{field: float('nan')}))
    assert any('CRITICAL' in w for w in warnings), f"Expected CRITICAL for NaN {field}"


@pytest.mark.parametrize("field", [
    'ctm_q_mean', 'ctm_bellman_target', 'ctm_td_error', 'ctm_cert_aux_loss',
])
def test_inf_triggers_critical(field):
    v = CTMValidator(_Cfg())
    warnings = v.check(_good_info(**{field: float('inf')}))
    assert any('CRITICAL' in w for w in warnings)


# ---------------------------------------------------------------------------
# 4. Q divergence
# ---------------------------------------------------------------------------

def test_q_diverge_warning():
    v = CTMValidator(_Cfg())
    warnings = v.check(_good_info(ctm_q_mean=600.0))
    assert any('WARNING' in w and 'Q-mean' in w for w in warnings)


def test_q_diverge_negative():
    v = CTMValidator(_Cfg())
    warnings = v.check(_good_info(ctm_q_mean=-600.0))
    assert any('WARNING' in w and 'Q-mean' in w for w in warnings)


def test_q_diverge_not_triggered_at_threshold():
    v = CTMValidator(_Cfg())
    # exactly at threshold — should NOT warn (uses >)
    warnings = v.check(_good_info(ctm_q_mean=500.0))
    assert not any('Q-mean' in w and 'WARNING' in w for w in warnings)


def test_nan_q_no_diverge_warning():
    # NaN should give CRITICAL only, not also a divergence WARNING
    v = CTMValidator(_Cfg())
    warnings = v.check(_good_info(ctm_q_mean=float('nan')))
    diverge_warnings = [w for w in warnings if 'WARNING' in w and 'Q-mean' in w]
    assert diverge_warnings == []


# ---------------------------------------------------------------------------
# 5. cert_score saturation (rate-limited)
# ---------------------------------------------------------------------------

def test_cert_score_low_warning():
    v = CTMValidator(_Cfg())
    # Pump 19: history has 19 entries. Explicit call is the 20th — first time
    # len >= 20 is satisfied, rate limit 20-(-500)=520 >= 500, so it fires here.
    _pump(v, 19, ctm_cert_score=0.01)
    warnings = v.check(_good_info(ctm_cert_score=0.01))
    assert any('cert_score' in w for w in warnings)


def test_cert_score_high_warning():
    v = CTMValidator(_Cfg())
    _pump(v, 19, ctm_cert_score=0.999)
    warnings = v.check(_good_info(ctm_cert_score=0.999))
    assert any('cert_score' in w for w in warnings)


def test_cert_score_normal_no_warning():
    v = CTMValidator(_Cfg())
    _pump(v, 30, ctm_cert_score=0.5)
    warnings = v.check(_good_info(ctm_cert_score=0.5))
    assert not any('cert_score' in w for w in warnings)


def test_cert_score_rate_limited():
    # Warn once, then suppress until CERT_WARN_FREQ more updates
    v = CTMValidator(_Cfg())
    _pump(v, 19, ctm_cert_score=0.01)
    first = v.check(_good_info(ctm_cert_score=0.01))   # 20th update — fires
    assert any('cert_score' in w for w in first)
    second = v.check(_good_info(ctm_cert_score=0.01))  # 21st — rate-limited
    assert not any('cert_score' in w for w in second)


def test_cert_score_warns_again_after_freq():
    v = CTMValidator(_Cfg())
    _pump(v, 19, ctm_cert_score=0.01)
    first = v.check(_good_info(ctm_cert_score=0.01))  # fires, sets _last_cert_warn
    assert any('cert_score' in w for w in first)
    # Directly rewind _last_cert_warn to simulate CERT_WARN_FREQ steps elapsed.
    # Pumping that many calls would re-trigger internally at exactly CERT_WARN_FREQ,
    # again leaving the assertion call rate-limited.
    v._last_cert_warn -= v.CERT_WARN_FREQ
    second = v.check(_good_info(ctm_cert_score=0.01))
    assert any('cert_score' in w for w in second)


def test_cert_score_not_warned_before_20_samples():
    v = CTMValidator(_Cfg())
    # Only 19 updates — window not full enough
    _pump(v, 19, ctm_cert_score=0.01)
    # The 19th call is done inside _pump; check manually for the edge case
    v2 = CTMValidator(_Cfg())
    for _ in range(18):
        v2.check(_good_info(ctm_cert_score=0.01))
    warnings = v2.check(_good_info(ctm_cert_score=0.01))
    assert not any('cert_score' in w for w in warnings)


# ---------------------------------------------------------------------------
# 6. Tick diversity (rate-limited, only after DIVERSITY_WARN_STEPS)
# ---------------------------------------------------------------------------

def test_diversity_no_warning_before_threshold():
    v = CTMValidator(_Cfg())
    # _update_count increments at the top of check(), so the Nth call results in
    # count=N. The check fires when count >= DIVERSITY_WARN_STEPS (200).
    # Pump 198 → explicit call is count=199 → 199 >= 200 False → no warn.
    _pump(v, v.DIVERSITY_WARN_STEPS - 2, ctm_tick_diversity=0.0)
    warnings = v.check(_good_info(ctm_tick_diversity=0.0))
    assert not any('tick_diversity' in w for w in warnings)


def test_diversity_warning_after_threshold():
    v = CTMValidator(_Cfg())
    # Pump 199 → explicit call is count=200 → 200 >= 200 True → fires.
    _pump(v, v.DIVERSITY_WARN_STEPS - 1, ctm_tick_diversity=0.0)
    warnings = v.check(_good_info(ctm_tick_diversity=0.0))
    assert any('tick_diversity' in w for w in warnings)


def test_diversity_rate_limited():
    v = CTMValidator(_Cfg())
    _pump(v, v.DIVERSITY_WARN_STEPS - 1, ctm_tick_diversity=0.0)
    first = v.check(_good_info(ctm_tick_diversity=0.0))   # count=200 — fires
    assert any('tick_diversity' in w for w in first)
    second = v.check(_good_info(ctm_tick_diversity=0.0))  # count=201 — rate-limited
    assert not any('tick_diversity' in w for w in second)


def test_diversity_no_warning_when_active():
    v = CTMValidator(_Cfg())
    _pump(v, v.DIVERSITY_WARN_STEPS + 10, ctm_tick_diversity=0.5)
    warnings = v.check(_good_info(ctm_tick_diversity=0.5))
    assert not any('tick_diversity' in w for w in warnings)


# ---------------------------------------------------------------------------
# 7. cert_aux early collapse
# ---------------------------------------------------------------------------

def test_cert_aux_collapse_early():
    v = CTMValidator(_Cfg())
    # Need 50 samples in history, all near-zero, before update 500
    _pump(v, 50, ctm_cert_aux_loss=0.0)
    warnings = v.check(_good_info(ctm_cert_aux_loss=0.0))
    assert any('cert_aux_loss' in w for w in warnings)


def test_cert_aux_collapse_not_after_500():
    v = CTMValidator(_Cfg())
    # Push past update 500 first with normal values
    _pump(v, 500, ctm_cert_aux_loss=0.3)
    # Now collapse — should not warn (too late)
    _pump(v, 50, ctm_cert_aux_loss=0.0)
    warnings = v.check(_good_info(ctm_cert_aux_loss=0.0))
    assert not any('cert_aux_loss' in w for w in warnings)


def test_cert_aux_no_collapse_warning_when_normal():
    v = CTMValidator(_Cfg())
    _pump(v, 55, ctm_cert_aux_loss=0.3)
    warnings = v.check(_good_info(ctm_cert_aux_loss=0.3))
    assert not any('cert_aux_loss' in w for w in warnings)


# ---------------------------------------------------------------------------
# 8. Alpha transition (fires exactly once)
# ---------------------------------------------------------------------------

def test_alpha_transition_fires_once():
    # anneal_steps=10 so alpha crosses 0.5 quickly
    v = CTMValidator(_Cfg(anneal_steps=10))
    saw_info = []
    for i in range(20):
        ws = v.check(_good_info(), episode=i)
        saw_info.extend([w for w in ws if 'Alpha' in w])
    assert len(saw_info) == 1


def test_alpha_transition_not_fired_if_never_crosses():
    # anneal_steps very large — alpha never gets near 0.5 in a short run
    v = CTMValidator(_Cfg(anneal_steps=10_000_000))
    for i in range(100):
        ws = v.check(_good_info(), episode=i)
        assert not any('Alpha' in w for w in ws)


def test_alpha_transition_message_contains_alpha_value():
    v = CTMValidator(_Cfg(anneal_steps=10))
    msgs = []
    for i in range(20):
        msgs.extend(v.check(_good_info(), episode=i))
    alpha_msgs = [w for w in msgs if 'Alpha' in w]
    assert len(alpha_msgs) == 1
    assert 'now' in alpha_msgs[0]  # message includes current alpha value


# ---------------------------------------------------------------------------
# 9. summary()
# ---------------------------------------------------------------------------

EXPECTED_SUMMARY_KEYS = {
    'q_mean_rolling', 'cert_score_rolling', 'cert_aux_loss_rolling',
    'tick_diversity_rolling', 'update_count',
}

def test_summary_keys():
    v = CTMValidator(_Cfg())
    assert set(v.summary().keys()) == EXPECTED_SUMMARY_KEYS


def test_summary_empty_before_any_updates():
    v = CTMValidator(_Cfg())
    s = v.summary()
    assert s['update_count'] == 0
    assert s['q_mean_rolling'] == 0.0


def test_summary_reflects_updates():
    v = CTMValidator(_Cfg())
    _pump(v, 5, ctm_q_mean=10.0)
    s = v.summary()
    assert s['update_count'] == 5
    assert abs(s['q_mean_rolling'] - 10.0) < 1e-6


def test_summary_rolling_window():
    v = CTMValidator(_Cfg(), window=10)
    _pump(v, 10, ctm_q_mean=1.0)
    _pump(v, 10, ctm_q_mean=99.0)
    s = v.summary()
    # window is 10, so only the last 10 (all 99.0) should be in rolling mean
    assert abs(s['q_mean_rolling'] - 99.0) < 1e-4


# ---------------------------------------------------------------------------
# 10. update_count
# ---------------------------------------------------------------------------

def test_update_count_increments_only_on_updated():
    v = CTMValidator(_Cfg())
    v.check(_good_info(updated=True))
    v.check(_good_info(updated=False))
    v.check(_good_info(updated=True))
    assert v._update_count == 2
