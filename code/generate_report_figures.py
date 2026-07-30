#!/usr/bin/env python3
"""
generate_report_figures.py — All supplementary report figures from run_final/.

Outputs (all to output/analysis/):
  fig_01_variant_barcharts.png      — Variant mean score bar chart per dataset
  fig_02_feature_importance.png     — Feature importance for all 6 trees
  fig_03_rho_table.png              — 5×5 ρ table with significance markers
  fig_04_ted_table.png              — 5×5 normalized TED table
  fig_05_rho_ted_scatter.png        — Joint ρ/TED scatter (off-diagonal cells)
  fig_06_per_task_rho.png           — Per-task-type ρ breakdown (5 datasets × task)
  fig_07_band_accuracy.png          — Band accuracy grouped bar + overall ρ overlay
  fig_08_strategy1_summary.png      — Strategy-1 validation summary table figure

Usage:
    python3 code/generate_report_figures.py
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from decision_tree_analysis import fit_tree, compare_trees, predict_and_validate, LABEL_MAP

# ── Config ─────────────────────────────────────────────────────────────────────

DATASETS     = ['history', 'amazon', 'arxiv', 'electronics', 'toys']
RUN_FINAL    = Path('output/run_final')
OUT_DIR      = Path('output/analysis')
VARIANT_COLS = ['Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx']
SCORE_COL    = 'Final_Score'
ZERO_THRESH  = 0.95

TASK_COLORS  = {'M1': '#3b82d4', 'M3': '#7c5cd8', 'M4': '#e67e22', 'M5': '#27ae60'}
TASK_LABELS  = {'M1': 'Node Categ. (M1)', 'M3': 'Edge Categ. (M3)',
                'M4': 'Edge Scalar (M4)', 'M5': 'Global Categ. (M5)'}
DS_DISPLAY   = {d: LABEL_MAP.get(d, d.title()) for d in DATASETS}
AXIS_DISPLAY = {'Task_Idx': 'Task', 'Node_Idx': 'Node',
                'Edge_Idx': 'Edge', 'Text_Idx': 'Text'}

PER_DATASET_EXCLUDE = {'arxiv': {'M2', 'M6'}}

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10,
                     'axes.labelsize': 9, 'figure.dpi': 150})


# ── Data helpers ───────────────────────────────────────────────────────────────

def load(dataset: str) -> pd.DataFrame:
    return pd.read_csv(RUN_FINAL / f'construction_performance_table_{dataset}.csv')


def build_summary(df: pd.DataFrame, dataset: str,
                  exclude_tasks: set = None) -> pd.DataFrame:
    sc = SCORE_COL if SCORE_COL in df.columns else 'normalized_score'
    if exclude_tasks:
        df = df[~df['Task_Idx'].isin(exclude_tasks)].copy()
    rows = []
    for key, grp in df.groupby(VARIANT_COLS):
        vd = dict(zip(VARIANT_COLS, key))
        tr = grp[grp['run_split'] == 'train'][sc].dropna()
        te = grp[grp['run_split'] == 'test'][sc].dropna()
        if len(tr) == 0 or len(te) == 0:
            continue
        n_tr = len(grp[grp['run_split'] == 'train'])
        n0   = int((tr == 0.0).sum())
        if n_tr > 0 and n0 / n_tr > ZERO_THRESH:
            continue
        rows.append({**vd,
                     'task_type':  grp['task_type'].iloc[0] if 'task_type' in grp.columns else '',
                     'train_mean': float(tr.mean()),
                     'test_mean':  float(te.mean()),
                     'n_train':    len(tr), 'n_test': len(te)})
    return pd.DataFrame(rows)


def load_all() -> dict:
    raw, summ = {}, {}
    for ds in DATASETS:
        raw[ds]  = load(ds)
        excl     = PER_DATASET_EXCLUDE.get(ds)
        summ[ds] = build_summary(raw[ds], ds, exclude_tasks=excl)
    return raw, summ


def fit_trees(summ: dict) -> tuple:
    tr_trees, te_trees = {}, {}
    for ds, s in summ.items():
        v = s.dropna(subset=['train_mean', 'test_mean'])
        if len(v) >= 5:
            tr_trees[ds] = fit_tree(v, 'train_mean', max_depth=8)
            te_trees[ds] = fit_tree(v, 'test_mean',  max_depth=8)
    # combined
    combined = pd.concat([s.assign(dataset=ds) for ds, s in summ.items()], ignore_index=True)
    v = combined.dropna(subset=['train_mean', 'test_mean'])
    feats = [c for c in VARIANT_COLS if c in v.columns]
    if len(v) >= 5:
        tr_trees['combined'] = fit_tree(v, 'train_mean', max_depth=8, features=feats)
        te_trees['combined'] = fit_tree(v, 'test_mean',  max_depth=8, features=feats)
    return tr_trees, te_trees


def axis_importances(tree_res: dict) -> dict:
    imp  = dict(zip(tree_res['ohe_names'], tree_res['tree'].feature_importances_))
    axis = {}
    for k, v in imp.items():
        col, _ = tree_res['ohe_to_axis_val'][k]
        axis[col] = axis.get(col, 0.0) + v
    return axis


# ── Figure 1 — Variant bar charts ──────────────────────────────────────────────

def fig_variant_barcharts(summ: dict):
    fig, axes = plt.subplots(1, 5, figsize=(22, 7), sharey=False)
    fig.suptitle('Variant Mean Score — sorted by unclamped score, coloured by task type',
                 fontsize=11, fontweight='bold', y=1.01)

    for ax, ds in zip(axes, DATASETS):
        s = summ[ds].copy()
        if s.empty:
            ax.set_title(DS_DISPLAY[ds]); ax.axis('off'); continue

        # Use unclamped_score mean if available, else train_mean for ordering
        raw = load(ds)
        sc  = SCORE_COL if SCORE_COL in raw.columns else 'normalized_score'
        unc_means = {}
        excl = PER_DATASET_EXCLUDE.get(ds, set())
        sub  = raw[~raw['Task_Idx'].isin(excl)]
        for key, grp in sub.groupby(VARIANT_COLS):
            te = grp[grp['run_split'] == 'test']
            if 'unclamped_score' in te.columns:
                v = te['unclamped_score'].dropna()
                if len(v):
                    lbl = '_'.join(str(k) for k in key)
                    unc_means[lbl] = float(v.mean())

        s['variant'] = s.apply(
            lambda r: f"{r['Task_Idx']}_{r['Node_Idx']}_{r['Edge_Idx']}_{r['Text_Idx']}", axis=1)
        s['unc_mean'] = s['variant'].map(unc_means).fillna(s['train_mean'])
        s = s.sort_values('unc_mean', ascending=True)

        colors = [TASK_COLORS.get(t, '#999999') for t in s['Task_Idx']]
        y = np.arange(len(s))
        ax.barh(y, s['train_mean'] * 100, color=colors, alpha=0.85, height=0.6, label='train')
        ax.barh(y, s['test_mean']  * 100, color=colors, alpha=0.40, height=0.6,
                left=0, edgecolor='none')

        ax.set_yticks(y)
        ax.set_yticklabels(s['variant'], fontsize=4.5)
        ax.set_xlabel('Score ×100', fontsize=8)
        ax.set_title(f'{DS_DISPLAY[ds]}\n({len(s)} variants)', fontsize=9, fontweight='bold')
        ax.axvline(0, color='#666', lw=0.5)
        ax.spines[['top','right']].set_visible(False)

    # legend
    patches = [mpatches.Patch(color=c, label=TASK_LABELS.get(t, t))
               for t, c in TASK_COLORS.items()]
    patches += [mpatches.Patch(color='#aaa', alpha=0.85, label='solid = train'),
                mpatches.Patch(color='#aaa', alpha=0.40, label='faded = test')]
    fig.legend(handles=patches, loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.04), fontsize=8, frameon=False)

    fig.tight_layout()
    out = OUT_DIR / 'fig_01_variant_barcharts.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out.name}')


# ── Figure 2 — Feature importance ─────────────────────────────────────────────

def fig_feature_importance(tr_trees: dict):
    keys   = [ds for ds in DATASETS if ds in tr_trees] + \
             (['combined'] if 'combined' in tr_trees else [])
    labels = [DS_DISPLAY.get(k, k.title()) for k in keys]
    axes_order = ['Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx']
    ax_colors  = {'Task_Idx': '#3b82d4', 'Node_Idx': '#7c5cd8',
                  'Edge_Idx': '#e67e22', 'Text_Idx': '#27ae60'}

    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 5), sharey=False)
    if n == 1: axes = [axes]
    fig.suptitle('Feature Importance by Axis — all 6 trees (train split)',
                 fontsize=11, fontweight='bold')

    for ax, key, lbl in zip(axes, keys, labels):
        imp = axis_importances(tr_trees[key])
        vals   = [imp.get(a, 0.0) for a in axes_order]
        colors = [ax_colors[a] for a in axes_order]
        xlbls  = [AXIS_DISPLAY.get(a, a) for a in axes_order]
        bars = ax.bar(xlbls, vals, color=colors, width=0.6, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(lbl, fontsize=9, fontweight='bold')
        ax.set_ylabel('Importance' if ax == axes[0] else '')
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)

    fig.tight_layout()
    out = OUT_DIR / 'fig_02_feature_importance.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out.name}')


# ── Figure 3 — 5×5 ρ table ────────────────────────────────────────────────────

def fig_rho_table():
    rho = pd.read_csv(OUT_DIR / 'cross_dataset_3x3_with_rho.csv', index_col=0)
    p   = pd.read_csv(OUT_DIR / 'cross_dataset_3x3_with_p.csv',   index_col=0)
    ds  = DATASETS

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axis('off')
    fig.suptitle('Cross-Dataset Spearman ρ — 5×5 Table\n'
                 '(diagonal = within-dataset Exp 1; * p<0.05; ** p<0.01; *** p<0.001)',
                 fontsize=11, fontweight='bold', y=0.97)

    cell_text, cell_colors = [], []
    for r in ds:
        row_t, row_c = [], []
        for c in ds:
            rv = float(rho.loc[r, c])
            pv = float(p.loc[r, c])
            if r == c:
                sig = '●'
            elif pv < 0.001: sig = '***'
            elif pv < 0.01:  sig = '** '
            elif pv < 0.05:  sig = '*  '
            else:             sig = '   '
            row_t.append(f'{rv:+.3f}{sig}')
            # colour by rho: blue positive, red negative
            if r == c:
                row_c.append('#e8f4f8')
            else:
                intensity = min(abs(rv), 1.0)
                if rv > 0:
                    row_c.append(plt.cm.Blues(0.15 + intensity * 0.65))
                else:
                    row_c.append(plt.cm.Reds(0.15 + intensity * 0.65))
        cell_text.append(row_t)
        cell_colors.append(row_c)

    col_labels = [DS_DISPLAY[d] for d in ds]
    row_labels = [DS_DISPLAY[d] for d in ds]

    tbl = ax.table(cellText=cell_text, cellColours=cell_colors,
                   rowLabels=row_labels, colLabels=col_labels,
                   cellLoc='center', loc='center',
                   bbox=[0.08, 0.0, 0.92, 0.88])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if r == 0 or c == -1:
            cell.set_text_props(fontweight='bold', fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / 'fig_03_rho_table.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out.name}')


# ── Figure 4 — 5×5 TED table ──────────────────────────────────────────────────

def fig_ted_table():
    ted = pd.read_csv(OUT_DIR / 'cross_dataset_3x3_with_ted.csv', index_col=0)
    p   = pd.read_csv(OUT_DIR / 'cross_dataset_3x3_with_p.csv',   index_col=0)
    ds  = DATASETS

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axis('off')
    fig.suptitle('Cross-Dataset Normalised Tree Edit Distance — 5×5 Table\n'
                 '(diagonal = within-dataset train vs test; * marks cells where ρ p<0.05)',
                 fontsize=11, fontweight='bold', y=0.97)

    cell_text, cell_colors = [], []
    for r in ds:
        row_t, row_c = [], []
        for c in ds:
            tv = ted.loc[r, c]
            pv = float(p.loc[r, c])
            sig = '*' if pv < 0.05 else ' '
            if pd.isna(tv) or tv == '':
                row_t.append('—'); row_c.append('#f7f8fa')
            else:
                tv = float(tv)
                if r == c:
                    row_t.append(f'{tv:.3f} ●')
                    row_c.append('#e8f4f8')
                else:
                    row_t.append(f'{tv:.3f}{sig}')
                    # higher TED = more orange
                    row_c.append(plt.cm.YlOrBr(0.1 + min(tv, 1.0) * 0.75))
        cell_text.append(row_t)
        cell_colors.append(row_c)

    col_labels = [DS_DISPLAY[d] for d in ds]
    row_labels = [DS_DISPLAY[d] for d in ds]

    tbl = ax.table(cellText=cell_text, cellColours=cell_colors,
                   rowLabels=row_labels, colLabels=col_labels,
                   cellLoc='center', loc='center',
                   bbox=[0.08, 0.0, 0.92, 0.88])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if r == 0 or c == -1:
            cell.set_text_props(fontweight='bold', fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / 'fig_04_ted_table.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out.name}')


# ── Figure 5 — ρ/TED scatter ──────────────────────────────────────────────────

def fig_rho_ted_scatter():
    rho = pd.read_csv(OUT_DIR / 'cross_dataset_3x3_with_rho.csv', index_col=0)
    ted = pd.read_csv(OUT_DIR / 'cross_dataset_3x3_with_ted.csv', index_col=0)
    p   = pd.read_csv(OUT_DIR / 'cross_dataset_3x3_with_p.csv',   index_col=0)

    pts = []
    for r in DATASETS:
        for c in DATASETS:
            if r == c: continue
            rv = rho.loc[r, c]; tv = ted.loc[r, c]; pv = float(p.loc[r, c])
            if pd.notna(rv) and pd.notna(tv):
                pts.append({'rho': float(rv), 'ted': float(tv),
                            'sig': pv < 0.05,
                            'label': f'{DS_DISPLAY[r][:3]}→{DS_DISPLAY[c][:3]}'})
    df = pd.DataFrame(pts)

    fig, ax = plt.subplots(figsize=(7, 5))
    sig_mask = df['sig'].values
    ax.scatter(df.loc[~sig_mask, 'ted'], df.loc[~sig_mask, 'rho'],
               color='#aaaaaa', s=55, alpha=0.7, zorder=3, label='p ≥ 0.05')
    ax.scatter(df.loc[sig_mask,  'ted'], df.loc[sig_mask,  'rho'],
               color='#3b82d4', s=80, alpha=0.9, zorder=4, label='p < 0.05')

    for _, row in df[sig_mask].iterrows():
        ax.annotate(row['label'], (row['ted'], row['rho']),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=7, color='#3b82d4')

    # regression line
    if len(df) >= 3:
        m, b = np.polyfit(df['ted'], df['rho'], 1)
        xs = np.linspace(df['ted'].min(), df['ted'].max(), 50)
        ax.plot(xs, m * xs + b, color='#e67e22', lw=1.5, ls='--', alpha=0.8,
                label=f'OLS slope={m:.2f}')
        rho_v, p_v = spearmanr(df['ted'], df['rho'])
        ax.text(0.05, 0.95, f'ρ(TED, ρ) = {rho_v:+.3f}  p={p_v:.2e}',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff8c5', edgecolor='#d4a72c'))

    ax.axhline(0, color='#666', lw=0.8, ls=':')
    ax.set_xlabel('Normalised Tree Edit Distance', fontsize=9)
    ax.set_ylabel('Spearman ρ (cross-dataset)', fontsize=9)
    ax.set_title('Joint ρ / TED Scatter — Off-Diagonal Cross-Dataset Pairs\n'
                 'Low TED does not reliably predict high ρ', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, frameon=True)
    ax.spines[['top','right']].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / 'fig_05_rho_ted_scatter.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out.name}')


# ── Figure 6 — Per-task ρ breakdown ───────────────────────────────────────────

def fig_per_task_rho(summ: dict, tr_trees: dict):
    tasks_all = ['M1', 'M3', 'M4', 'M5']
    records = []

    for ds in DATASETS:
        if ds not in tr_trees: continue
        s   = summ[ds].dropna(subset=['train_mean', 'test_mean'])
        pv  = predict_and_validate(tr_trees[ds], s, target_col='test_mean')
        pred, act = pv['predictions'], pv['actual']

        for tt in tasks_all:
            mask  = (s['Task_Idx'] == tt).values
            p_sub = pred[mask]; a_sub = act[mask]
            valid = ~np.isnan(a_sub)
            n     = int(valid.sum())
            if n >= 5:
                rho_v, p_v = spearmanr(p_sub[valid], a_sub[valid])
            else:
                rho_v, p_v = np.nan, np.nan
            records.append({'dataset': ds, 'task': tt,
                            'rho': rho_v, 'p': p_v, 'n': n})

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 5))
    ds_list   = [d for d in DATASETS if d in tr_trees]
    n_ds      = len(ds_list)
    n_tasks   = len(tasks_all)
    bar_w     = 0.15
    x         = np.arange(n_ds)

    for i, tt in enumerate(tasks_all):
        sub    = df[df['task'] == tt].set_index('dataset')
        vals   = [sub.loc[ds, 'rho'] if ds in sub.index else np.nan for ds in ds_list]
        ps     = [sub.loc[ds, 'p']   if ds in sub.index else np.nan for ds in ds_list]
        ns     = [int(sub.loc[ds, 'n']) if ds in sub.index else 0    for ds in ds_list]
        offset = (i - n_tasks/2 + 0.5) * bar_w

        bars = ax.bar(x + offset, vals, width=bar_w,
                      color=TASK_COLORS.get(tt, '#999'),
                      alpha=0.85, label=TASK_LABELS.get(tt, tt),
                      edgecolor='white', linewidth=0.4)

        for bar, v, pv, n in zip(bars, vals, ps, ns):
            if np.isnan(v): continue
            sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
            ypos = bar.get_height() + 0.02 if v >= 0 else bar.get_height() - 0.08
            if sig:
                ax.text(bar.get_x() + bar.get_width()/2, ypos, sig,
                        ha='center', fontsize=7, color='#333')
            ax.text(bar.get_x() + bar.get_width()/2,
                    -0.30 if v >= 0 else v - 0.08,
                    f'n={n}', ha='center', fontsize=5.5, color='#555', rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([DS_DISPLAY[d] for d in ds_list], fontsize=9)
    ax.axhline(0, color='#666', lw=0.8)
    ax.set_ylabel('Spearman ρ (tree train → test_mean)', fontsize=9)
    ax.set_title('Per-Task-Type ρ Breakdown — Strategy-1 Validation\n'
                 '(* p<0.05  ** p<0.01  *** p<0.001  n = variant count)',
                 fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, frameon=True)
    ax.set_ylim(-0.5, 1.15)
    ax.spines[['top','right']].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / 'fig_06_per_task_rho.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out.name}')


# ── Figure 7 — Band accuracy grouped bar + ρ overlay ─────────────────────────

def fig_band_accuracy(summ: dict, tr_trees: dict):
    records = []
    for ds in DATASETS:
        if ds not in tr_trees: continue
        s = summ[ds].dropna(subset=['train_mean', 'test_mean'])
        pv = predict_and_validate(tr_trees[ds], s, target_col='test_mean')
        records.append({'dataset': ds,
                        'band_acc': pv['band_accuracy'],
                        'rho':      pv['spearman_rho'],
                        'p':        pv['spearman_p'],
                        'n':        len(s)})
    df = pd.DataFrame(records)

    ds_list = df['dataset'].tolist()
    x = np.arange(len(ds_list))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(x - 0.18, df['band_acc'], width=0.32,
                   color='#7c5cd8', alpha=0.80, label='Band Accuracy (H/M/L)')
    ax2.bar(x + 0.18, df['rho'], width=0.32,
            color='#3b82d4', alpha=0.80, label='Spearman ρ')

    for bar, v in zip(bars, df['band_acc']):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                 f'{v:.2f}', ha='center', fontsize=8, color='#4a2d8c', fontweight='bold')

    for xi, (rho_v, pv, n) in enumerate(zip(df['rho'], df['p'], df['n'])):
        sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'ns'
        color = '#0969da' if pv < 0.05 else '#888'
        ax2.text(xi + 0.18, rho_v + 0.02, f'{rho_v:+.3f}{sig}',
                 ha='center', fontsize=7.5, color=color, fontweight='bold')
        ax2.text(xi + 0.18, -0.12, f'n={n}',
                 ha='center', fontsize=7, color='#555')

    ax1.axhline(1/3, color='#7c5cd8', ls=':', lw=1, alpha=0.5,
                label='Random baseline (0.33)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([DS_DISPLAY[d] for d in ds_list], fontsize=10)
    ax1.set_ylabel('Band Accuracy', fontsize=9, color='#4a2d8c')
    ax2.set_ylabel('Spearman ρ',    fontsize=9, color='#0969da')
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(-0.3, 1.3)
    ax1.set_title('Strategy-1 Validation — Band Accuracy & Spearman ρ per Dataset',
                  fontsize=10, fontweight='bold')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8, frameon=True)
    ax1.spines[['top']].set_visible(False)
    ax2.spines[['top']].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / 'fig_07_band_accuracy.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out.name}')


# ── Figure 8 — Strategy-1 summary table ───────────────────────────────────────

def fig_strategy1_table(summ: dict, tr_trees: dict, te_trees: dict):
    from decision_tree_analysis import _tree_edit_distance

    rows = []
    for ds in DATASETS:
        if ds not in tr_trees: continue
        s    = summ[ds].dropna(subset=['train_mean', 'test_mean'])
        tr   = tr_trees[ds]; te = te_trees[ds]
        cmp  = compare_trees(tr, te, s)
        pv   = predict_and_validate(tr, s, target_col='test_mean')

        n_tr = tr['tree'].tree_.node_count
        n_te = te['tree'].tree_.node_count
        ted_raw  = cmp['tree_edit_distance']
        ted_norm = ted_raw / (n_tr + n_te) if (n_tr + n_te) > 0 else np.nan

        sig_s1 = '***' if pv['spearman_p'] < 0.001 else \
                 '**'  if pv['spearman_p'] < 0.01  else \
                 '*'   if pv['spearman_p'] < 0.05  else 'ns'
        rows.append([
            DS_DISPLAY[ds],
            str(len(s)),
            f"{tr['r2']:.3f}",
            f"{te['r2']:.3f}",
            f"{cmp['spearman_rho']:+.3f}",
            f"{pv['spearman_rho']:+.3f} {sig_s1}",
            f"{pv['band_accuracy']:.3f}",
            f"{ted_norm:.3f}  (raw={ted_raw})",
        ])

    col_labels = ['Dataset', 'N variants',
                  'R²(train)', 'R²(test)',
                  'ρ train↔test', 'ρ Strategy-1 (sig)',
                  'Band Acc', 'Norm TED']

    fig, ax = plt.subplots(figsize=(14, 3.0 + 0.5 * len(rows)))
    ax.axis('off')
    fig.suptitle('Strategy-1 Validation Summary — All Datasets',
                 fontsize=11, fontweight='bold', y=0.98)

    # Colour ρ Strategy-1 column by significance
    cell_colors = []
    for row in rows:
        sig = row[5].split()[-1] if row[5].split() else 'ns'
        rho_color = ('#dafbe1' if sig not in ('ns',) else '#ffeef0')
        cell_colors.append(['#f7f8fa'] * 5 + [rho_color] + ['#f7f8fa'] * 2)

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellColours=cell_colors,
                   cellLoc='center', loc='center',
                   bbox=[0.0, 0.0, 1.0, 0.88])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(col_labels))))
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if r == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#e8f4f8')

    fig.tight_layout()
    out = OUT_DIR / 'fig_08_strategy1_summary.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out.name}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('REPORT FIGURES — output/run_final/ → output/analysis/')
    print('=' * 72)

    print('\nLoading data and fitting trees...')
    raw, summ = load_all()
    tr_trees, te_trees = fit_trees(summ)
    print(f'  Trees fitted: {sorted(tr_trees.keys())}')

    print('\nFig 1 — Variant bar charts')
    fig_variant_barcharts(summ)

    print('\nFig 2 — Feature importance')
    fig_feature_importance(tr_trees)

    print('\nFig 3 — 5×5 ρ table')
    fig_rho_table()

    print('\nFig 4 — 5×5 TED table')
    fig_ted_table()

    print('\nFig 5 — ρ/TED scatter')
    fig_rho_ted_scatter()

    print('\nFig 6 — Per-task ρ breakdown')
    fig_per_task_rho(summ, tr_trees)

    print('\nFig 7 — Band accuracy')
    fig_band_accuracy(summ, tr_trees)

    print('\nFig 8 — Strategy-1 summary table')
    fig_strategy1_table(summ, tr_trees, te_trees)

    print('\n' + '=' * 72)
    print(f'DONE  →  {OUT_DIR}')
    figs = sorted(OUT_DIR.glob('fig_0*.png'))
    for f in figs:
        print(f'  {f.name}')
    print('=' * 72)


if __name__ == '__main__':
    main()
