#!/usr/bin/env python3
"""
generate_ablation_heatmaps.py — Three-panel ablation heatmaps from run_final/.

One figure per dataset, three side-by-side panels:
  Left   — S_GNN_step1   (raw GNN accuracy/F1/R², sequential colourmap)
  Centre — Final_Score   (clamped GNN lift × 100, sequential colourmap)
  Right  — unclamped_score (signed GNN lift × 100, diverging colourmap)

Layout:
  Y-axis  : all variants sorted by mean unclamped_score descending (shared across panels)
  X-axis  : held-out test samples 20–29 (first 10 of the 30 new samples)
  Source  : output/run_final/ CSVs (all five datasets, canonical schema)

Output:
  output/analysis/ablation_heatmap_{dataset}.png

Usage:
    python3 code/generate_ablation_heatmaps.py
"""

import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# ── Config ─────────────────────────────────────────────────────────────────────

DATASETS     = ['history', 'amazon', 'arxiv', 'electronics', 'toys']
RUN_FINAL    = Path('output/run_final')
OUT_DIR      = Path('output/analysis')
VARIANT_COLS = ['Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx']

# Held-out test samples: 20–29 for datasets with 50 samples (electronics/toys).
# For pre-populated datasets (history/amazon/arxiv) that only have samples 0–9,
# fall back to the last 10 available test samples so all five plots are generated.
HELD_OUT_SAMPLES = list(range(20, 30))
HELD_OUT_FALLBACK_N = 10   # use last N test samples when 20-29 not available

PANELS = [
    # (score_col,         scale_factor, label,                          diverging, sequential_cmap)
    ('S_GNN_step1',       100.0,        'Raw GNN Score (×100)',          False,     'YlGnBu'),
    ('Final_Score',       100.0,        'Clamped GNN Lift (×100)',       False,     'YlOrRd'),
    ('unclamped_score',   1.0,          'Non-Clamped GNN Lift (×100)',   True,      None),
]

# unclamped_score is already stored as fraction × 100 in run_final
# (trainer emits it that way). S_GNN_step1 and Final_Score are [0,1] → ×100.


# ── Helpers ────────────────────────────────────────────────────────────────────

def load(dataset: str) -> pd.DataFrame:
    path = RUN_FINAL / f'construction_performance_table_{dataset}.csv'
    return pd.read_csv(path)


def variant_label(row) -> str:
    return f"{row['Task_Idx']}_{row['Node_Idx']}_{row['Edge_Idx']}_{row['Text_Idx']}"


def build_pivot(
    df: pd.DataFrame,
    score_col: str,
    scale: float,
    samples: list,
    row_order: list,
) -> pd.DataFrame:
    """
    Pivot to (variant × sample) for the given test samples.
    Scores are multiplied by `scale` before pivoting.
    Missing cells stay NaN.
    """
    sub = df[
        (df['run_split'] == 'test') &
        (df['sample_idx'].isin(samples))
    ].copy()

    sub['variant'] = sub.apply(variant_label, axis=1)
    sub['_val']    = sub[score_col].astype(float) * scale

    pivot = sub.pivot_table(
        index='variant',
        columns='sample_idx',
        values='_val',
        aggfunc='mean',
        dropna=False,
    )
    pivot = pivot.reindex(columns=sorted(samples))
    pivot = pivot.reindex(row_order)
    return pivot


def compute_row_order(df: pd.DataFrame, samples: list) -> list:
    """
    Sort variants by mean unclamped_score (test, samples 20-29) descending.
    Variants already have unclamped_score in run_final (stored as fraction×100 by trainer,
    or as raw fraction for pre-populated rows — normalise to ×100 display scale).
    """
    sub = df[
        (df['run_split'] == 'test') &
        (df['sample_idx'].isin(samples))
    ].copy()

    sub['variant'] = sub.apply(variant_label, axis=1)

    # unclamped_score: trainer stores as fraction (same scale as Final_Score, i.e. [−1, 1])
    # multiply by 100 for display, then rank
    means = (
        sub.groupby('variant')['unclamped_score']
        .mean()
        .fillna(-999)
    )
    return list(means.sort_values(ascending=False).index)


# ── Plotter ────────────────────────────────────────────────────────────────────

def plot_ablation(dataset: str, df: pd.DataFrame):
    test_df   = df[df['run_split'] == 'test']
    samples   = [s for s in HELD_OUT_SAMPLES if s in test_df['sample_idx'].values]

    # Fall back to last HELD_OUT_FALLBACK_N samples when 20-29 not present
    if not samples:
        all_test = sorted(test_df['sample_idx'].unique())
        samples  = all_test[-HELD_OUT_FALLBACK_N:]
        print(f'  Fallback: using last {len(samples)} test samples '
              f'({samples[0]}–{samples[-1]})')

    row_order = compute_row_order(df, samples)
    n_var     = len(row_order)
    n_samp    = len(samples)

    if n_var == 0 or n_samp == 0:
        print(f'  [{dataset}] SKIP — no test samples found')
        return

    x_label = f'{samples[0]}–{samples[-1]}'
    fig_h = max(10, n_var * 0.30)
    fig_w = 3 * max(6, n_samp * 0.6) + 1.5   # three panels + spacing

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(
        f'{dataset.title()} — Ablation Scoring Comparison\n'
        f'Y: variants sorted by mean non-clamped score  |  test samples {x_label}',
        fontsize=13, fontweight='bold', y=0.995,
    )

    gs = gridspec.GridSpec(
        1, 3,
        figure=fig,
        wspace=0.35,
        left=0.16, right=0.97,
        top=0.93, bottom=0.06,
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax_idx, (score_col, scale, label, diverging, seq_cmap) in enumerate(PANELS):
        ax     = axes[ax_idx]
        pivot  = build_pivot(df, score_col, scale, samples, row_order)

        vals = pivot.values[~np.isnan(pivot.values)]
        if len(vals) == 0:
            ax.set_title(f'{label}\n(no data)', fontsize=9)
            ax.axis('off')
            continue

        vmin_v = float(vals.min())
        vmax_v = float(vals.max())

        if diverging:
            cmap   = 'RdBu_r'
            abs_m  = max(abs(vmin_v), abs(vmax_v))
            vmin_v, vmax_v = -abs_m, abs_m
            center = 0.0
        else:
            cmap   = seq_cmap
            center = None

        show_yticklabels = (ax_idx == 0)
        show_cbar        = True

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=vmin_v,
            vmax=vmax_v,
            center=center,
            linewidths=0.3,
            linecolor='#e0e0e0',
            cbar=show_cbar,
            cbar_kws={'label': 'Score', 'shrink': 0.6, 'pad': 0.02},
            yticklabels=show_yticklabels,
            xticklabels=True,
        )

        ax.set_title(label, fontsize=9, fontweight='bold', pad=8)
        ax.set_xlabel('Sample', fontsize=8)
        ax.set_ylabel('Variant' if ax_idx == 0 else '', fontsize=8)

        if show_yticklabels:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)

    out_path = OUT_DIR / f'ablation_heatmap_{dataset}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved ({n_var} variants × {n_samp} samples): {out_path.name}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('ABLATION HEATMAPS — output/run_final/ → output/analysis/')
    print('Panels: S_GNN_step1 | Final_Score (clamped) | unclamped_score')
    print(f'X-axis: test samples {HELD_OUT_SAMPLES[0]}–{HELD_OUT_SAMPLES[-1]}')
    print('=' * 72)

    for dataset in DATASETS:
        print(f'\n{dataset.upper()}')
        df = load(dataset)

        # Report sample coverage
        test_samples = sorted(df[df['run_split'] == 'test']['sample_idx'].unique())
        held = [s for s in HELD_OUT_SAMPLES if s in test_samples]
        print(f'  test samples available: {test_samples[0]}–{test_samples[-1]} '
              f'(n={len(test_samples)})  |  held-out 20-29: {len(held)} present')

        plot_ablation(dataset, df)

    print('\n' + '=' * 72)
    print(f'DONE  →  {OUT_DIR}')
    print('=' * 72)


if __name__ == '__main__':
    main()
