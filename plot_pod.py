"""
plot_pod.py — DCEF Tool 05
Generates a range-binned probability of detection (PoD) curve
with 90% binomial confidence intervals from a DCEF correlated
CSV. This is the primary procurement-relevant output.

Requirements: pandas, matplotlib, numpy, scipy
Usage: python plot_pod.py <correlated.csv> [output.png]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom

ORANGE  = '#f26115'
GREEN   = '#4ade80'
BG      = '#060606'
SURFACE = '#0e0e0e'
DIM     = '#888888'
BORDER  = '#1a1a1a'

BIN_STEP   = 50     # metres per range bin
POD_THRESH = 0.90   # procurement threshold line
CI_ALPHA   = 0.10   # 90% confidence interval


def binom_ci(k, n, alpha=0.10):
    """Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    z   = 1.645  # 90% CI
    p   = k / n
    lo  = (p + z**2/(2*n) - z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
    hi  = (p + z**2/(2*n) + z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
    return max(0, lo), min(1, hi)


def plot_pod(csv_path: str, output_path: str = None):
    if output_path is None:
        output_path = csv_path.replace('.csv', '_pod.png')

    df = pd.read_csv(csv_path)
    df['detection_flag'] = df['detection_flag'].astype(bool)
    print(f'[DCEF] Loaded {len(df)} records | '
          f'Overall PoD: {df["detection_flag"].mean():.1%}')

    # ── Bin by range ──────────────────────────────────────────
    r_min = (df['range_m'].min() // BIN_STEP) * BIN_STEP
    r_max = (df['range_m'].max() // BIN_STEP + 1) * BIN_STEP
    edges = np.arange(r_min, r_max + BIN_STEP, BIN_STEP)
    mids  = (edges[:-1] + edges[1:]) / 2

    pods, ci_los, ci_his, ns = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = df[(df['range_m'] >= lo) & (df['range_m'] < hi)]
        n   = len(sub)
        k   = sub['detection_flag'].sum()
        pod = k / n if n > 0 else np.nan
        lo_ci, hi_ci = binom_ci(k, n)
        pods.append(pod)
        ci_los.append(lo_ci)
        ci_his.append(hi_ci)
        ns.append(n)

    pods    = np.array(pods, dtype=float)
    ci_los  = np.array(ci_los, dtype=float)
    ci_his  = np.array(ci_his, dtype=float)
    valid   = ~np.isnan(pods)

    # ── Plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.set_facecolor(SURFACE)
    ax.grid(color=BORDER, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # CI band
    ax.fill_between(mids[valid], ci_los[valid], ci_his[valid],
                    alpha=0.15, color=ORANGE, label='90% CI')

    # CI border lines
    ax.plot(mids[valid], ci_his[valid], color=ORANGE,
            linewidth=0.6, linestyle='--', alpha=0.35)
    ax.plot(mids[valid], ci_los[valid], color=ORANGE,
            linewidth=0.6, linestyle='--', alpha=0.35)

    # PoD curve
    ax.plot(mids[valid], pods[valid],
            color=ORANGE, linewidth=2.5, zorder=4, label='PoD')
    ax.scatter(mids[valid], pods[valid],
               c=ORANGE, s=50, zorder=5, linewidths=0)

    # 0.90 threshold
    ax.axhline(POD_THRESH, color=GREEN, linewidth=1.0,
               linestyle='--', alpha=0.5, label=f'PoD = {POD_THRESH}')
    ax.text(mids[valid][-1], POD_THRESH + 0.01,
            f'  PoD = {POD_THRESH}', color=GREEN, fontsize=7.5,
            fontfamily='monospace', va='bottom')

    # Sample count annotations
    for mx, pod, n in zip(mids[valid], pods[valid], np.array(ns)[valid]):
        ax.text(mx, pod + 0.02, f'n={n}',
                ha='center', va='bottom', fontsize=6.5,
                color=DIM, fontfamily='monospace')

    # Effective range annotation
    eff_mask = pods >= POD_THRESH
    if eff_mask.any():
        eff_range = mids[valid][eff_mask[valid]][-1]
        ax.axvline(eff_range, color='white', linewidth=0.8,
                   linestyle=':', alpha=0.3)
        ax.text(eff_range + 5, 0.05,
                f'Eff. range\n{eff_range:.0f} m',
                color=DIM, fontsize=7, fontfamily='monospace')

    # Tier badge
    tier = int(df['tier'].iloc[0])
    ax.text(0.98, 0.02, f'TIER {tier}', transform=ax.transAxes,
            fontsize=8, color=ORANGE, va='bottom', ha='right',
            fontfamily='monospace', weight='bold')

    ax.set_xlabel('Slant Range  (m)', color=DIM, fontsize=9)
    ax.set_ylabel('Probability of Detection', color=DIM, fontsize=9)
    ax.set_title('DCEF — Range-Binned Probability of Detection\n'
                 f'{BIN_STEP}m bins · Wilson 90% CI',
                 color='white', fontsize=11, pad=14, weight='bold')
    ax.set_ylim(-0.02, 1.08)
    ax.tick_params(colors=DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    ax.legend(facecolor='#161616', edgecolor=BORDER,
              labelcolor='white', fontsize=8, loc='lower left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=BG, bbox_inches='tight')
    print(f'[DCEF] Saved → {output_path}')
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_pod.py <correlated.csv> [output.png]')
        sys.exit(1)
    plot_pod(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
