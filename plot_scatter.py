"""
plot_scatter.py — DCEF Tool 03
Generates a 2D horizontal position error scatter plot from a
DCEF correlated CSV. Points are colour-coded by range band.
Concentric CEP rings and a crosshair mark the truth origin.

Requirements: pandas, matplotlib, numpy
Usage: python plot_scatter.py <correlated.csv> [output.png]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── DCEF brand palette ────────────────────────────────────────
ORANGE  = '#f26115'
BLUE    = '#4a9eff'
PURPLE  = '#a78bfa'
GREEN   = '#4ade80'
BG      = '#060606'
SURFACE = '#0e0e0e'
DIM     = '#888888'
BORDER  = '#1a1a1a'

RANGE_BINS  = [(300, 400), (400, 500), (500, 600)]
BIN_COLORS  = [BLUE, ORANGE, PURPLE]
BIN_LABELS  = ['300–400 m', '400–500 m', '500–600 m']


def plot_scatter(csv_path: str, output_path: str = None):
    if output_path is None:
        output_path = csv_path.replace('.csv', '_scatter.png')

    df = pd.read_csv(csv_path)
    print(f'[DCEF] Loaded {len(df)} records from {csv_path}')

    # ── Compute relative offsets in metres from truth ─────────
    R_LAT  = 111_320.0
    R_LON  = 111_320.0 * np.cos(np.radians(df['lat_truth'].mean()))

    df['dx'] = (df['sensor_lon'] - df['lon_truth']) * R_LON
    df['dy'] = (df['sensor_lat'] - df['lat_truth']) * R_LAT

    # ── Figure ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG)
    ax.set_facecolor(SURFACE)

    # Grid
    ax.grid(color=BORDER, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # CEP rings
    max_err = df['pos_error_m'].max()
    ring_steps = np.arange(5, max_err + 5, 5)
    for r in ring_steps:
        circle = plt.Circle((0, 0), r, color=ORANGE,
                             fill=False, linewidth=0.6,
                             linestyle='--', alpha=0.18, zorder=1)
        ax.add_patch(circle)
        ax.text(r * 0.71, r * 0.71, f'{r:.0f}m',
                color=ORANGE, fontsize=6, alpha=0.4,
                ha='center', va='center')

    # Points by range bin
    for (lo, hi), colour, label in zip(RANGE_BINS, BIN_COLORS, BIN_LABELS):
        mask = (df['range_m'] >= lo) & (df['range_m'] < hi)
        sub  = df[mask]
        ax.scatter(sub['dx'], sub['dy'],
                   c=colour, s=20, alpha=0.7,
                   label=label, linewidths=0, zorder=3)

    # Truth origin crosshair
    ax.axhline(0, color='white', linewidth=0.8, alpha=0.4, linestyle='--', zorder=2)
    ax.axvline(0, color='white', linewidth=0.8, alpha=0.4, linestyle='--', zorder=2)
    ax.scatter([0], [0], c='white', s=40, zorder=5, marker='+', linewidths=1.5)

    # Mean error marker
    mx, my = df['dx'].mean(), df['dy'].mean()
    ax.scatter([mx], [my], c=ORANGE, s=60, zorder=6,
               marker='x', linewidths=2, label=f'Mean offset ({mx:.1f}, {my:.1f})m')

    # Stats annotation
    stats = (f"n = {len(df)}\n"
             f"Mean error: {df['pos_error_m'].mean():.2f} m\n"
             f"95th pct:   {df['pos_error_m'].quantile(0.95):.2f} m\n"
             f"Max error:  {df['pos_error_m'].max():.2f} m")
    ax.text(0.02, 0.98, stats, transform=ax.transAxes,
            fontsize=7.5, color=DIM, va='top', ha='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#161616',
                      edgecolor=BORDER, alpha=0.9))

    # Tier badge
    tier = int(df['tier'].iloc[0])
    ax.text(0.98, 0.98, f'TIER {tier}', transform=ax.transAxes,
            fontsize=8, color=ORANGE, va='top', ha='right',
            fontfamily='monospace', weight='bold')

    # Labels & formatting
    ax.set_xlabel('East / West offset  (m)', color=DIM, fontsize=9)
    ax.set_ylabel('North / South offset  (m)', color=DIM, fontsize=9)
    ax.set_title('DCEF — 2D Position Error Scatter\nSensor-reported vs. Telemetry Ground Truth',
                 color='white', fontsize=11, pad=14, weight='bold')
    ax.tick_params(colors=DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    ax.legend(facecolor='#161616', edgecolor=BORDER,
              labelcolor='white', fontsize=7.5, loc='lower right')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=BG, bbox_inches='tight')
    print(f'[DCEF] Saved → {output_path}')
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_scatter.py <correlated.csv> [output.png]')
        sys.exit(1)
    plot_scatter(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
