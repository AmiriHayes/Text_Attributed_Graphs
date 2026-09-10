#!/usr/bin/env python3
"""
Builds the two-panel, paper-ready ArXiv figure: left = epoch accuracy curves
for 6 selected M1 variants (3 top-tier N8, 3 bottom-tier N7, mean test_top1
across 3 samples), right = ArXiv M1 raw_gnn heatmap (variant x sample) from
run_post_audit. Styling: Times New Roman, thick/thin line convention, %
y-axis 0-100, epochs every 10, blank heatmap axis labels.

Reads:
  - output/epoch_figure_arxiv/epoch_logs_arxiv_{N}_{E}_{T}_sample{s}.csv
  - output/run_post_audit/construction_performance_table_arxiv.csv

Writes:
  - output/figures/fig2_arxiv_construction_performance.png

Usage:
  python code/build_fig2.py
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
EPOCH_DIR = REPO_ROOT / 'output/epoch_figure_arxiv'
OUT_DIR = REPO_ROOT / 'output/figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ('N8', 'E10b', 'T12b', '#EA4335'),   # Red
    ('N8', 'E11a', 'T12a', '#FF6D00'),   # Orange
    ('N8', 'E11b', 'T12b', '#FBBC05'),   # Yellow
    ('N7', 'E10b', 'T12b', '#34A853'),   # Green
    ('N7', 'E10c', 'T12b', '#4285F4'),   # Blue
    ('N7', 'E11b', 'T12b', '#A142F4'),   # Purple
]
NODE_STYLE = {'N8': '-', 'N7': '--'}
SAMPLES = [0, 1, 2]

fig = plt.figure(figsize=(15, 6.5), facecolor='white')
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.28)

# ── LEFT: epoch accuracy curves ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for N, E, T, color in VARIANTS:
    exp_id = f'{N}_{E}_{T}'
    curves = []
    for s in SAMPLES:
        p = EPOCH_DIR / f'epoch_logs_arxiv_{exp_id}_sample{s}.csv'
        d = pd.read_csv(p).set_index('epoch')['test_top1'] * 100
        curves.append(d)
        ax1.plot(d.index, d.values, color=color, linewidth=1.6, alpha=0.35, zorder=2)

    mean_curve = pd.concat(curves, axis=1).mean(axis=1)
    ax1.plot(mean_curve.index, mean_curve.values, color=color, linestyle=NODE_STYLE[N],
              linewidth=5, label=f'{N}/{E}/{T}', zorder=3)

ax1.text(0.5, 1.10, 'GNN Performance on Node Classification',
         transform=ax1.transAxes, ha='center', fontsize=17, fontweight='bold')
ax1.text(0.5, 1.03, 'Individual Samples = Thin Lines     Mean of Samples = Thick Line',
         transform=ax1.transAxes, ha='center', fontsize=11.5, style='italic')

ax1.set_xlabel('Epochs', fontsize=15)
ax1.set_ylabel('GNN Accuracy', fontsize=15)
ax1.set_ylim(0, 101)
ax1.set_yticks(range(0, 101, 10))
ax1.set_yticklabels([f'{v}%' for v in range(0, 101, 10)])
ax1.set_xticks(range(0, 101, 10))
ax1.legend(fontsize=11, loc='lower right', frameon=True)
ax1.spines[['top', 'right']].set_visible(False)
ax1.grid(alpha=0.25)
ax1.tick_params(labelsize=12)

# ── RIGHT: raw_gnn heatmap, variant x sample, M1-only, run_post_audit ───
raw_df = pd.read_csv(REPO_ROOT / 'output/run_post_audit/construction_performance_table_arxiv.csv')
raw_df['sample_idx'] = pd.to_numeric(raw_df['sample_idx'], errors='coerce').astype('Int64')
df = raw_df[raw_df['Task_Idx'] == 'M1']

test_df = df[df['run_split'] == 'test']
samples = [s for s in HELD_OUT_SAMPLES if s in test_df['sample_idx'].values]
if not samples:
    all_test = sorted(test_df['sample_idx'].unique())
    samples = all_test[-HELD_OUT_FALLBACK_N:]

row_order = compute_row_order(df, samples)
pivot = build_pivot(df, 'S_GNN_step1', 100.0, samples, row_order)

ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(
    pivot, ax=ax2, cmap='YlGnBu',
    linewidths=0.3, linecolor='#e0e0e0',
    cbar_kws={'label': 'Raw GNN Score (x100)', 'shrink': 0.75, 'pad': 0.02},
    yticklabels=True, xticklabels=True,
)
ax2.text(0.5, 1.06, 'raw_gnn score heatmap across all variants and subsets',
         transform=ax2.transAxes, ha='center', fontsize=13, fontweight='bold')
ax2.set_xlabel('', fontsize=1)
ax2.set_ylabel('', fontsize=1)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=6.5)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)

plt.tight_layout()
out = OUT_DIR / 'fig2_arxiv_construction_performance.png'
fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
print(f'saved -> {out}')
print(f'  left panel: {len(VARIANTS)} variants x {len(SAMPLES)} samples averaged')
print(f'  right panel: {len(row_order)} M1 variants x {len(samples)} test samples ({samples[0]}-{samples[-1]})')
