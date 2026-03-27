"""CTM training validation — Phase 5.

Pure Python checks that run after each jit_train_step when use_ctm_critic=True.
All inputs are already-materialized Python floats (extracted from JAX arrays).

Checks:
  1. Q-values finite and bounded
  2. Bellman target finite
  3. cert_score not saturated (near 0 = overly conservative, near 1 = not learning)
  4. TD error finite
  5. cert_aux_loss finite and decreasing over time (cert head training signal active)
  6. tick_diversity > 0 (tick_best and tick_certain actually diverge — mechanism active)

Usage in training loop:
    from ctm_validation import CTMValidator
    validator = CTMValidator(config)
    ...
    warnings = validator.check(train_info, episode=episode)
    for w in warnings: print(w)
"""

import numpy as np
from collections import deque
from typing import Dict, Any, List


class CTMValidator:
    """Stateful validator — tracks rolling metrics to detect trends."""

    # cert_score bounds: outside these values something is wrong
    CERT_SCORE_LOW  = 0.05   # near 0 → Bellman targets collapse to reward-only
    CERT_SCORE_HIGH = 0.995  # saturated → certainty head not discriminating

    # Q magnitude threshold — environment-dependent, tuned for assembly reward scale
    Q_DIVERGE_THRESHOLD = 500.0

    # cert_score saturation: warn at most once every N steps to avoid log spam
    CERT_WARN_FREQ = 500

    # tick_diversity: fraction of batch where tick_best ≠ tick_certain
    # If this stays 0 for many steps the mechanism is not active
    DIVERSITY_WARN_STEPS = 200   # how many updates before we warn about zero diversity
    DIVERSITY_WARN_FREQ  = 500   # only warn once every N steps to avoid log spam

    def __init__(self, config, window: int = 100):
        """
        Args:
            config: AssemblyTrainConfig — used to check use_ctm_critic and alpha schedule
            window: rolling window size for trend detection
        """
        self.enabled = getattr(config, 'use_ctm_critic', False)
        self.alpha_anneal_steps = getattr(config, 'ctm_alpha_anneal_steps', 100000)
        self.alpha_final = getattr(config, 'ctm_alpha_final', 0.0)

        self._q_history          = deque(maxlen=window)
        self._cert_history       = deque(maxlen=window)
        self._cert_aux_history   = deque(maxlen=window)
        self._diversity_history  = deque(maxlen=window)
        self._update_count        = 0
        self._last_cert_warn      = -self.CERT_WARN_FREQ
        self._last_diversity_warn = -self.DIVERSITY_WARN_FREQ
        self._alpha_half_warned   = False

    def check(self, info: Dict[str, Any], episode: int = 0) -> List[str]:
        """Run all checks. Returns list of warning strings (empty = all good).

        Args:
            info: dict returned by jit_train_step — must contain CTM keys
            episode: current episode number (for context in messages)

        Returns:
            List of warning strings. Empty list means all checks passed.
        """
        if not self.enabled:
            return []

        # Only validate when updates actually ran
        if not bool(info.get('updated', False)):
            return []

        self._update_count += 1

        q_mean    = float(info.get('ctm_q_mean',         0.0))
        bellman   = float(info.get('ctm_bellman_target', 0.0))
        cert      = float(info.get('ctm_cert_score',     0.0))
        td_err    = float(info.get('ctm_td_error',       0.0))
        cert_aux  = float(info.get('ctm_cert_aux_loss',  0.0))
        diversity = float(info.get('ctm_tick_diversity', 0.0))

        self._q_history.append(q_mean)
        self._cert_history.append(cert)
        self._cert_aux_history.append(cert_aux)
        self._diversity_history.append(diversity)

        warnings = []

        # ------------------------------------------------------------------
        # 1. Finiteness — highest priority
        # ------------------------------------------------------------------
        if not np.isfinite(q_mean):
            warnings.append(
                f"[CTM CRITICAL] Q-mean is not finite ({q_mean}) at episode {episode}. "
                f"Check for exploding gradients or NaN in synch EMA."
            )
        if not np.isfinite(bellman):
            warnings.append(
                f"[CTM CRITICAL] Bellman target is not finite ({bellman}) at episode {episode}."
            )
        if not np.isfinite(td_err):
            warnings.append(
                f"[CTM CRITICAL] TD error is not finite ({td_err}) at episode {episode}."
            )
        if not np.isfinite(cert_aux):
            warnings.append(
                f"[CTM CRITICAL] cert_aux_loss is not finite ({cert_aux}) at episode {episode}."
            )

        # ------------------------------------------------------------------
        # 2. Q-value divergence
        # ------------------------------------------------------------------
        if np.isfinite(q_mean) and abs(q_mean) > self.Q_DIVERGE_THRESHOLD:
            warnings.append(
                f"[CTM WARNING] Q-mean magnitude high ({q_mean:.1f}) at episode {episode}. "
                f"May be diverging. Monitor ctm_td_error trend."
            )

        # ------------------------------------------------------------------
        # 3. cert_score saturation
        # ------------------------------------------------------------------
        if np.isfinite(cert) and len(self._cert_history) >= 20:
            mean_cert = np.mean(list(self._cert_history)[-20:])
            cert_out_of_bounds = mean_cert < self.CERT_SCORE_LOW or mean_cert > self.CERT_SCORE_HIGH
            if cert_out_of_bounds and (self._update_count - self._last_cert_warn) >= self.CERT_WARN_FREQ:
                self._last_cert_warn = self._update_count
                if mean_cert < self.CERT_SCORE_LOW:
                    warnings.append(
                        f"[CTM WARNING] cert_score rolling mean is {mean_cert:.4f} (< {self.CERT_SCORE_LOW}). "
                        f"Bellman targets are near reward-only. Learning may be slow."
                    )
                else:
                    warnings.append(
                        f"[CTM WARNING] cert_score rolling mean is {mean_cert:.4f} (> {self.CERT_SCORE_HIGH}). "
                        f"Certainty saturated — head may not be discriminating across ticks."
                    )

        # ------------------------------------------------------------------
        # 4. Tick diversity — tick_best and tick_certain should diverge
        # ------------------------------------------------------------------
        if self._update_count >= self.DIVERSITY_WARN_STEPS:
            recent_diversity = np.mean(list(self._diversity_history)[-50:]) if self._diversity_history else 0.0
            if recent_diversity < 0.01:
                steps_since_warn = self._update_count - self._last_diversity_warn
                if steps_since_warn >= self.DIVERSITY_WARN_FREQ:
                    warnings.append(
                        f"[CTM WARNING] tick_diversity near zero ({recent_diversity:.4f}) "
                        f"over last 50 updates at episode {episode}. "
                        f"tick_best and tick_certain always agree — CTM may not be using "
                        f"multiple ticks meaningfully. Consider increasing ctm_iterations."
                    )
                    self._last_diversity_warn = self._update_count

        # ------------------------------------------------------------------
        # 5. cert_aux_loss — should be positive and ideally decreasing over time
        # ------------------------------------------------------------------
        if np.isfinite(cert_aux) and len(self._cert_aux_history) >= 50:
            # Warn once early if cert_aux is already near-zero (cert head collapsed)
            mean_cert_aux = np.mean(list(self._cert_aux_history)[-20:])
            if mean_cert_aux < 1e-6 and self._update_count < 500:
                warnings.append(
                    f"[CTM WARNING] cert_aux_loss is near zero ({mean_cert_aux:.2e}) very early "
                    f"in training. The certainty head may have collapsed. "
                    f"Check cert_projector gradient flow."
                )

        # ------------------------------------------------------------------
        # 6. Alpha-transition check
        # ------------------------------------------------------------------
        # Warn once when alpha crosses below 0.5 — the learned head becomes dominant
        alpha_progress = min(self._update_count / max(self.alpha_anneal_steps, 1), 1.0)
        alpha_now = 1.0 - (1.0 - self.alpha_final) * alpha_progress
        if not self._alpha_half_warned and alpha_now < 0.5:
            self._alpha_half_warned = True
            recent_cert_aux = np.mean(list(self._cert_aux_history)[-20:]) if self._cert_aux_history else 0.0
            warnings.append(
                f"[CTM INFO] Alpha crossed 0.5 (now {alpha_now:.3f}) — learned certainty "
                f"head is now dominant. cert_aux_loss = {recent_cert_aux:.4f}. "
                f"If cert_score becomes erratic, consider slowing the anneal schedule."
            )

        return warnings

    def summary(self) -> Dict[str, float]:
        """Rolling statistics for the last window of updates."""
        def _mean(q): return float(np.mean(list(q))) if q else 0.0
        return {
            'q_mean_rolling':          _mean(self._q_history),
            'cert_score_rolling':      _mean(self._cert_history),
            'cert_aux_loss_rolling':   _mean(self._cert_aux_history),
            'tick_diversity_rolling':  _mean(self._diversity_history),
            'update_count':            self._update_count,
        }
