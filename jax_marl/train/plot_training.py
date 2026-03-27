#!/usr/bin/env python3
"""Training dashboard from training_log.json.

Reads the JSON log written during training and produces a multi-panel
summary figure. CTM panels are shown only when CTM keys are present in
the log (i.e. use_ctm_critic=True was active during the run).

Usage:
    python plot_training.py <log_file> [--out <dir>]

    python plot_training.py runs/my_run/training_log.json
    python plot_training.py runs/my_run/training_log.json --out runs/my_run/plots
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth(values, window=20):
    """Trailing moving average, preserves list length."""
    arr = np.array(values, dtype=float)
    if len(arr) < 2:
        return arr
    w = min(window, len(arr))
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='same')


def _valid_pairs(episodes, values):
    """Return (eps, vals) with NaN/None stripped."""
    pairs = [(e, v) for e, v in zip(episodes, values)
             if v is not None and np.isfinite(float(v))]
    if not pairs:
        return [], []
    return zip(*pairs)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_dashboard(log_path: Path, out_dir: Path) -> None:
    with open(log_path) as f:
        entries = json.load(f)

    train = [e for e in entries if e.get('type') != 'eval']
    eval_ = [e for e in entries if e.get('type') == 'eval']

    if not train:
        print("No training entries found in log.")
        return

    eps          = [e['episode']                       for e in train]
    rewards      = [e.get('reward_mean',  np.nan)      for e in train]
    coverage     = [e.get('coverage_rate', np.nan)     for e in train]
    actor_loss   = [e.get('actor_loss',   np.nan)      for e in train]
    critic_loss  = [e.get('critic_loss',  np.nan)      for e in train]

    # CTM keys — only present when use_ctm_critic=True
    has_ctm = any('ctm_q_mean' in e for e in train)
    if has_ctm:
        ctm_td_err   = [e.get('ctm_td_error',       np.nan) for e in train]
        ctm_cert     = [e.get('ctm_cert_score',      np.nan) for e in train]
        ctm_aux      = [e.get('ctm_cert_aux_loss',   np.nan) for e in train]
        ctm_div      = [e.get('ctm_tick_diversity',  np.nan) for e in train]

    n_panels = 4 if has_ctm else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # ------------------------------------------------------------------
    # Panel 1: Reward + Coverage
    # ------------------------------------------------------------------
    ax = axes[0]
    ax_r = ax.twinx()
    re, rv = _valid_pairs(eps, rewards)
    ce, cv = _valid_pairs(eps, coverage)
    if re:
        ax.plot(list(re), _smooth(list(rv)), color='tab:blue', linewidth=1.5,
                label='Reward (smoothed)')
    if ce:
        ax_r.plot(list(ce), _smooth(list(cv)), color='tab:green', linewidth=1.5,
                  alpha=0.8, label='Coverage (smoothed)')
    ax.set_ylabel('Mean Reward', color='tab:blue')
    ax_r.set_ylabel('Coverage Rate', color='tab:green')
    ax_r.set_ylim(0, 1)
    ax.set_title('Reward & Coverage')
    ax.grid(True, alpha=0.3)
    lines = ax.get_lines() + ax_r.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='lower right', fontsize=8)

    # Eval reward dots
    if eval_:
        eval_eps = [e['episode'] for e in eval_]
        eval_rew = [e.get('eval_reward_mean', np.nan) for e in eval_]
        ax.scatter(eval_eps, eval_rew, color='navy', s=20, zorder=5,
                   label='Eval reward', marker='D')

    # ------------------------------------------------------------------
    # Panel 2: Losses
    # ------------------------------------------------------------------
    ax = axes[1]
    ae, av = _valid_pairs(eps, actor_loss)
    ce2, cv2 = _valid_pairs(eps, critic_loss)
    if ae:
        ax.plot(list(ae), _smooth(list(av)), color='tab:orange', linewidth=1.5,
                label='Actor loss')
    if ce2:
        ax.plot(list(ce2), _smooth(list(cv2)), color='tab:red', linewidth=1.5,
                label='Critic loss')
    if has_ctm:
        axe, axv = _valid_pairs(eps, ctm_aux)
        if axe:
            ax.plot(list(axe), _smooth(list(axv)), color='tab:purple', linewidth=1.2,
                    linestyle='--', label='cert_aux_loss')
    ax.set_ylabel('Loss')
    ax.set_title('Losses')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if has_ctm:
        # ------------------------------------------------------------------
        # Panel 3: CTM convergence — TD error + cert_score
        # ------------------------------------------------------------------
        ax = axes[2]
        ax_c = ax.twinx()
        tde, tdv = _valid_pairs(eps, ctm_td_err)
        cse, csv = _valid_pairs(eps, ctm_cert)
        if tde:
            ax.plot(list(tde), _smooth(list(tdv)), color='tab:red', linewidth=1.5,
                    label='TD error')
        if cse:
            ax_c.plot(list(cse), _smooth(list(csv)), color='tab:cyan', linewidth=1.5,
                      alpha=0.9, label='cert_score')
        # Saturation threshold bands
        ax_c.axhline(0.05,  color='tab:cyan', linestyle=':', linewidth=0.8, alpha=0.5)
        ax_c.axhline(0.995, color='tab:cyan', linestyle=':', linewidth=0.8, alpha=0.5)
        ax_c.set_ylim(0, 1)
        ax.set_ylabel('TD Error', color='tab:red')
        ax_c.set_ylabel('cert_score', color='tab:cyan')
        ax.set_title('CTM Convergence  (cert_score dashed lines = saturation bounds)')
        ax.grid(True, alpha=0.3)
        lines = ax.get_lines() + ax_c.get_lines()
        ax.legend([l for l in lines if l.get_label()[0] != '_'],
                  [l.get_label() for l in lines if l.get_label()[0] != '_'],
                  fontsize=8)

        # ------------------------------------------------------------------
        # Panel 4: Tick diversity
        # ------------------------------------------------------------------
        ax = axes[3]
        de, dv = _valid_pairs(eps, ctm_div)
        if de:
            ax.plot(list(de), _smooth(list(dv)), color='tab:brown', linewidth=1.5,
                    label='tick_diversity')
        ax.axhline(0.01, color='gray', linestyle=':', linewidth=0.8, alpha=0.6,
                   label='diversity=0.01 threshold')
        ax.set_ylim(0, 1)
        ax.set_ylabel('tick_diversity')
        ax.set_title('CTM Mechanism Activity  (fraction of steps where tick_best ≠ tick_certain)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Episode')
    fig.suptitle(f'Training Dashboard — {log_path.parent.name}', fontsize=13, fontweight='bold')
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'training_dashboard.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from JSON log.')
    parser.add_argument('log_file', type=Path, help='Path to training_log.json')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output directory (default: same dir as log_file)')
    args = parser.parse_args()

    log_path = args.log_file.resolve()
    if not log_path.exists():
        print(f'Error: {log_path} not found')
        return

    out_dir = args.out.resolve() if args.out else log_path.parent / 'plots'
    plot_dashboard(log_path, out_dir)


if __name__ == '__main__':
    main()
