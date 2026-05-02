"""
plot_polar.py — DCEF Tool 04
Generates a polar error plot showing directional sensor bias
from a DCEF correlated CSV. Each spoke = mean position error
for tracks where the drone was flying in that heading.

Requirements: pandas, matplotlib, numpy
Usage: python plot_polar.py <correlated.csv> [output.png]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ORANGE  = '#f26115'
BG      = '#060606'
SURFACE = '#0e0e0e'
DIM     = '#888888'
BORDER  = '#1a1a1a'

N_BINS  = 16   # 22.5° steps


def plot_polar(csv_path: str, output_path: str = None):
    if output_path is None:
        output_path = csv_path.replace('.csv', '_polar.png')

    df = pd.read_csv(csv_path)
    print(f'[DCEF] Loaded {len(df)} records')

    # Bin headings
    edges  = np.linspace(0, 360, N_BINS + 1)
    labels = (edges[:-1] + edges[1:]) / 2
    df['hdg_bin'] = pd.cut(df['hdg_deg'], bins=edges,
                           labels=labels, include_lowest=True)
    binned = df.groupby('hdg_bin', observed=True)['pos_error_m'].mean()

    # Align to N_BINS; fill missing bins with 0
    angles  = np.radians(labels)
    values  = np.array([binned.get(lbl, 0) for lbl in labels])
    # Close the polygon
    angles  = np.append(angles, angles[0])
    values  = np.append(values, values[0])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'},
                           facecolor=BG)
    ax.set_facecolor(SURFACE)

    # Grid styling
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.spines['polar'].set_color(BORDER)
    for gridline in ax.yaxis.get_gridlines():
        gridline.set_color(BORDER)
        gridline.set_linewidth(0.4)
    ax.set_rlabel_position(45)

    # Compass labels
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                      ['N','NE','E','SE','S','SW','W','NW'],
                      color=DIM, fontsize=8)

    # Filled polygon
    ax.fill(angles, values, alpha=0.18, color=ORANGE)
    ax.plot(angles, values, color=ORANGE, linewidth=1.8)

    # Mean error ring
    mean_err = df['pos_error_m'].mean()
    mean_circle = np.full(361, mean_err)
    ax.plot(np.radians(np.arange(361)), mean_circle,
            color='white', linewidth=0.8, linestyle='--', alpha=0.25,
            label=f'Mean {mean_err:.2f} m')

    # Peak annotation
    peak_idx = np.argmax(values[:-1])
    peak_ang = np.degrees(angles[peak_idx])
    peak_val = values[peak_idx]
    ax.annotate(f'{peak_val:.1f}m @ {peak_ang:.0f}°',
                xy=(angles[peak_idx], peak_val),
                xytext=(angles[peak_idx], peak_val + mean_err * 0.4),
                color=ORANGE, fontsize=7, fontfamily='monospace',
                ha='center')

    ax.set_title('DCEF — Polar Directional Bias\nMean position error by vehicle heading',
                 color='white', fontsize=10, pad=20, weight='bold')

    tier = int(df['tier'].iloc[0])
    fig.text(0.98, 0.98, f'TIER {tier}', color=ORANGE, fontsize=8,
             fontfamily='monospace', weight='bold', ha='right', va='top')

    ax.legend(facecolor='#161616', edgecolor=BORDER,
              labelcolor='white', fontsize=7.5, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=BG, bbox_inches='tight')
    print(f'[DCEF] Saved → {output_path}')
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_polar.py <correlated.csv> [output.png]')
        sys.exit(1)
    plot_polar(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
