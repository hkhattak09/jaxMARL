"""CTM evaluation visualizations.

Called during eval episodes when use_ctm_critic=True.
All functions are pure matplotlib — no JAX required.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict


def plot_ctm_tick_progression(
    ctm_step_data: List[Dict],
    out_path,
    episode: int,
    n_ticks: int,
) -> None:
    """Mean Q-value and certainty across ticks, averaged over the eval episode.

    Args:
        ctm_step_data: list of per-step dicts, each containing:
            'q_ticks'    (n_agents, T) — Q at every tick
            'cert_ticks' (n_agents, T) — certainty at every tick
        out_path:  file path to save the figure
        episode:   episode number (for title)
        n_ticks:   number of CTM iterations (T)
    """
    q_all    = np.stack([d['q_ticks']    for d in ctm_step_data], axis=0)  # (S, A, T)
    cert_all = np.stack([d['cert_ticks'] for d in ctm_step_data], axis=0)  # (S, A, T)

    q_mean = q_all.mean(axis=(0, 1))     # (T,)
    q_std  = q_all.std(axis=(0, 1))
    c_mean = cert_all.mean(axis=(0, 1))  # (T,)
    c_std  = cert_all.std(axis=(0, 1))

    ticks = np.arange(n_ticks)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(ticks, q_mean, 'o-', color='tab:blue', linewidth=2, label='Mean Q')
    ax1.fill_between(ticks, q_mean - q_std, q_mean + q_std, alpha=0.2, color='tab:blue')
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Q-value')
    ax1.set_title('Q-value across CTM ticks')
    ax1.set_xticks(ticks)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(ticks, c_mean, 'o-', color='tab:green', linewidth=2, label='Mean certainty')
    ax2.fill_between(ticks, c_mean - c_std, c_mean + c_std, alpha=0.2, color='tab:green')
    ax2.set_xlabel('Tick')
    ax2.set_ylabel('Certainty score')
    ax2.set_title('Certainty across CTM ticks')
    ax2.set_ylim(0, 1)
    ax2.set_xticks(ticks)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Mark the tick with peak certainty
    peak_tick = int(np.argmax(c_mean))
    ax2.axvline(peak_tick, color='red', linestyle='--', linewidth=1.2,
                alpha=0.7, label=f'Peak tick={peak_tick}')
    ax2.legend(fontsize=8)

    fig.suptitle(f'CTM Tick Progression — Episode {episode}', fontweight='bold')
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_tick_scatter(
    ctm_step_data: List[Dict],
    out_path,
    episode: int,
    n_ticks: int,
) -> None:
    """Scatter of tick_best vs tick_certain and divergence histogram.

    Args:
        ctm_step_data: list of per-step dicts, each containing:
            'tick_best'    (n_agents,) — argmax Q tick
            'tick_certain' (n_agents,) — argmax certainty tick
        out_path:  file path to save the figure
        episode:   episode number (for title)
        n_ticks:   number of CTM iterations (T)
    """
    tb   = np.concatenate([d['tick_best']    for d in ctm_step_data])  # (S*A,)
    tc   = np.concatenate([d['tick_certain'] for d in ctm_step_data])  # (S*A,)
    diff = tc.astype(int) - tb.astype(int)

    diversity_frac = np.mean(diff != 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter with jitter so density is visible
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.3, 0.3, size=len(tb))
    ax1.scatter(tb + jitter, tc + jitter, alpha=0.15, s=6, color='tab:blue')
    lims = [-0.5, n_ticks - 0.5]
    ax1.plot(lims, lims, 'r--', linewidth=1.5, alpha=0.8, label='tick_best = tick_certain')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xticks(range(n_ticks))
    ax1.set_yticks(range(n_ticks))
    ax1.set_xlabel('tick_best (highest Q)')
    ax1.set_ylabel('tick_certain (highest certainty)')
    ax1.set_title(f'tick_best vs tick_certain  (diversity={diversity_frac:.1%})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Histogram of divergence
    bin_edges = np.arange(-(n_ticks - 1) - 0.5, n_ticks - 0.5 + 1)
    ax2.hist(diff, bins=bin_edges, color='tab:purple', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, label='zero divergence')
    ax2.set_xlabel('tick_certain − tick_best')
    ax2.set_ylabel('Count')
    ax2.set_title('Tick divergence distribution')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'CTM tick_best vs tick_certain — Episode {episode}', fontweight='bold')
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
