#!/usr/bin/env python3
"""
Standalone epoch-curve plot: all 18 individual sample curves (thin, faded)
plus the 6 per-variant mean curves (thick) -- so sample-to-sample variance is
visible, not just the averaged Figure 2 panel.

Reads:
  - output/epoch_figure_arxiv/epoch_logs_arxiv_{N}_{E}_{T}_sample{s}.csv
    (produced by run_epoch_figure_arxiv.py)

Writes:
  - output/figures/epoch_curves_standalone.png

Usage:
  python code/plot_epoch_curves_standalone.py
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

REPO_ROOT = Path(__file__).resolve().parent.parent
EPOCH_DIR = REPO_ROOT / 'output/epoch_figure_arxiv'
OUT_DIR = REPO_ROOT / 'output/figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Palette as given: Blue, Purple, Yellow, Green, Red, Orange.
# 6 curves need 6 distinct colors -- warm (red/orange/yellow) for the top
# tier, cool (green/blue/purple) for the bottom tier.
VARIANTS = [
    ('N8', 'E10b', 'T12b', 'top', '#EA4335'),     # Red
    ('N8', 'E11a', 'T12a', 'top', '#FF6D00'),     # Orange
    ('N8', 'E11b', 'T12b', 'top', '#FBBC05'),     # Yellow
    ('N7', 'E10b', 'T12b', 'bottom', '#34A853'),  # Green
    ('N7', 'E10c', 'T12b', 'bottom', '#4285F4'),  # Blue
    ('N7', 'E11b', 'T12b', 'bottom', '#A142F4'),  # Purple
]
SAMPLES = [0, 1, 2]

fig, ax = plt.subplots(figsize=(9, 6.5), facecolor='white')

for N, E, T, tier, color in VARIANTS:
    exp_id = f'{N}_{E}_{T}'
    curves = []
    for s in SAMPLES:
        p = EPOCH_DIR / f'epoch_logs_arxiv_{exp_id}_sample{s}.csv'
        d = pd.read_csv(p).set_index('epoch')['test_top1'] * 100  # fraction -> percent
        curves.append(d)
        ax.plot(d.index, d.values, color=color, linewidth=1.6, alpha=0.35, zorder=2)

    mean_curve = pd.concat(curves, axis=1).mean(axis=1)
    linestyle = '-' if tier == 'top' else '--'
    ax.plot(mean_curve.index, mean_curve.values, color=color, linestyle=linestyle,
             linewidth=5, label=f'{N}/{E}/{T}', zorder=3)

ax.set_xlabel('Epochs', fontsize=18)
ax.set_ylabel('GNN Accuracy', fontsize=18)

# Title + a second, smaller line underneath explaining thin-vs-thick,
# rather than folding it into the legend.
# Both title lines as ax.text() in the same (axes-relative) coordinate
# system, rather than mixing fig.suptitle (figure coordinates, not
# governed by tight_layout) with an axes title -- that combination can't
# be tightened predictably. This way the gap between the two lines, and
# between them and the plot, is directly controllable.
ax.text(0.5, 1.13, 'GNN Performance on Node Classification',
        transform=ax.transAxes, ha='center', fontsize=20, fontweight='bold')
ax.text(0.5, 1.055, 'Individual Samples = Thin Lines     Mean of Samples = Thick Line',
        transform=ax.transAxes, ha='center', fontsize=14, style='italic')

ax.legend(fontsize=14, loc='lower right', frameon=True)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25)

# Y axis: whole-number percents, 0-100 with headroom to 101 so the "100"
# tick isn't clipped at the top edge.
ax.set_ylim(0, 101)
ax.set_yticks(range(0, 101, 10))
ax.set_yticklabels([f'{v}%' for v in range(0, 101, 10)])

# X axis: every 10 epochs.
ax.set_xticks(range(0, 101, 10))

ax.tick_params(labelsize=15)

plt.tight_layout()
out = OUT_DIR / 'epoch_curves_standalone.png'
fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
print(f'saved -> {out}')
