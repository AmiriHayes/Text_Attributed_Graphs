#!/usr/bin/env python3
"""
Builds the two-panel, paper-ready Amazon figure: left = epoch accuracy curves
for 6 selected M1 variants (2 per node type, mean test_top1 across 3
samples), right = Amazon M1 raw_gnn heatmap (variant x sample) from
run_post_audit. Same styling as the finalized ArXiv standalone figure: Times
New Roman, thick/thin line convention, % y-axis 0-100, epochs every 10,
blank heatmap axis labels.

Reads:
  - output/epoch_figure_amazon/epoch_logs_amazon_{N}_{E}_{T}_sample{s}.csv
  - output/run_post_audit/construction_performance_table_amazon.csv

Writes:
  - output/figures/fig_amazon_construction_performance.png
  - output/figures/fig_amazon_construction_performance.pdf

Usage:
  python code/build_fig_amazon.py
"""
import sys
from pathlib import Path

sys.path.insert(0, 'code')
from generate_ablation_heatmaps import build_pivot, compute_row_order, HELD_OUT_SAMPLES, HELD_OUT_FALLBACK_N

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

REPO_ROOT = Path(__file__).resolve().parent.parent
EPOCH_DIR = REPO_ROOT / 'output/epoch_figure_amazon'
OUT_DIR = REPO_ROOT / 'output/figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ('N7', 'E11b', 'T12a', '#EA4335'),   # Red
    ('N7', 'E10c', 'T12a', '#FF6D00'),   # Orange
    ('N8', 'E10a', 'T12b', '#4285F4'),   # Blue
    ('N8', 'E11b', 'T12a', '#A142F4'),   # Purple
    ('N9', 'E11b', 'T12a', '#FBBC05'),   # Yellow
    ('N9', 'E10b', 'T12a', '#34A853'),   # Green
]
NODE_STYLE = {'N7': '-', 'N9': '-.', 'N8': '--'}
SAMPLES = [0, 1, 2]

fig = plt.figure(figsize=(15, 6.5), facecolor='white')
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.28)

# ── LEFT: epoch accuracy curves ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for N, E, T, color in VARIANTS:
    exp_id = f'{N}_{E}_{T}'
    curves = []
    for s in SAMPLES:
        p = EPOCH_DIR / f'epoch_logs_amazon_{exp_id}_sample{s}.csv'
        d = pd.read_csv(p).set_index('epoch')['test_top1'] * 100
        curves.append(d)
        ax1.plot(d.index, d.values, color=color, linewidth=1.6, alpha=0.35, zorder=2)

    mean_curve = pd.concat(curves, axis=1).mean(axis=1)
    ax1.plot(mean_curve.index, mean_curve.values, color=color, linestyle=NODE_STYLE[N],
              linewidth=5, label=f'{N}/{E}/{T}', zorder=3)

ax1.text(0.5, 1.05, 'Individual Samples = Thin Lines     Mean of Samples = Thick Line',
         transform=ax1.transAxes, ha='center', fontsize=11.5, style='italic')

ax1.set_xlabel('Epochs', fontsize=15)
ax1.set_ylabel('GNN Accuracy', fontsize=15)
ax1.set_ylim(0, 70)
ax1.set_yticks(range(0, 71, 10))
ax1.set_yticklabels([f'{v}%' for v in range(0, 71, 10)])
ax1.set_xticks(range(0, 101, 10))
ax1.legend(fontsize=11, loc='lower right', frameon=True)
ax1.spines[['top', 'right']].set_visible(False)
ax1.grid(alpha=0.25)
ax1.tick_params(labelsize=12)

# ── RIGHT: raw_gnn heatmap, variant x sample, M1-only, run_post_audit ───
# T12e dropped: every N7/N8/N9 T12e row is exactly 0.0 (confirmed, not a
# bug -- zero node features means the GNN has nothing to learn category
# signal from beyond graph structure, which doesn't encode it for any of
# the non-leaking edge types), so those rows add no information here.
raw_df = pd.read_csv(REPO_ROOT / 'output/run_post_audit/construction_performance_table_amazon.csv')
raw_df['sample_idx'] = pd.to_numeric(raw_df['sample_idx'], errors='coerce').astype('Int64')
df = raw_df[(raw_df['Task_Idx'] == 'M1') & (raw_df['Text_Idx'] != 'T12e')]

test_df = df[df['run_split'] == 'test']
samples = [s for s in HELD_OUT_SAMPLES if s in test_df['sample_idx'].values]
if not samples:
    all_test = sorted(test_df['sample_idx'].unique())
    samples = all_test[-HELD_OUT_FALLBACK_N:]

row_order = compute_row_order(df, samples)
# Top1 (raw top-1 test accuracy), not S_GNN_step1 (a pseudo-R2 score --
# a stricter, differently-scaled metric that would visually mismatch the
# left panel's accuracy curves despite both being labeled "accuracy").
pivot = build_pivot(df, 'Top1', 100.0, samples, row_order)

ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(
    pivot, ax=ax2, cmap='RdYlGn',
    linewidths=0.3, linecolor='#e0e0e0',
    cbar=False,  # red=low, green=high is self-explanatory; colorbar was
                 # eating vertical space and squashing the matrix
    yticklabels=True, xticklabels=False,
    annot=True, fmt='.0f', annot_kws={'fontsize': 6.5},
)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8.5)

# Single figure-level title covering both panels (heatmap doesn't need
# its own -- that mismatch was what threw off vertical alignment).
fig.suptitle('GNN Performance on Node Classification', fontsize=19, fontweight='bold', y=1.02)

plt.tight_layout()

# Stretch the heatmap taller after layout settles: the left panel's plot
# area starts lower because of the axis-note subtitle sitting above it,
# so the heatmap (no such subtitle) reads shorter at the same nominal
# grid height. Extend it up and down to visually match.
pos = ax2.get_position()
stretch_top, stretch_bottom = 0.035, 0.025
ax2.set_position([pos.x0, pos.y0 - stretch_bottom,
                   pos.width, pos.height + stretch_top + stretch_bottom])

out_png = OUT_DIR / 'fig_amazon_construction_performance.png'
out_pdf = OUT_DIR / 'fig_amazon_construction_performance.pdf'
fig.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')  # vector -- use this one in LaTeX
print(f'saved -> {out_png}')
print(f'saved -> {out_pdf}')
print(f'  left panel: {len(VARIANTS)} variants x {len(SAMPLES)} samples averaged')
print(f'  right panel: {len(row_order)} M1 variants x {len(samples)} test samples ({samples[0]}-{samples[-1]})')
