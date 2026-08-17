#!/usr/bin/env python3
"""
run_analysis.py — Three-experiment TAG construction analysis (v2).

Usage (from repo root):
    python3 code/run_analysis.py [--open]

Reads:  output/run_20260620/{dataset}/construction_performance_table_{dataset}.csv
Writes: output/analysis/decision_tree_{dataset}_combined.png
        output/analysis/cross_dataset_3x3_{with,clean}_rho.csv
        output/analysis/cross_dataset_3x3_{with,clean}_p.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

from decision_tree_analysis import (
    fit_tree,
    compare_trees,
    predict_and_validate,
    plot_tree_categorical,
    _tree_edit_distance,
    LABEL_MAP,
)

# ── Configuration — change here, zero code changes elsewhere ──────────────────
# Map each dataset name to the run directory that contains its CSV.
# _load_csv probes both layout patterns:
#   (a) {run_dir}/{dataset}/construction_performance_table_{dataset}.csv  (older runs)
#   (b) {run_dir}/construction_performance_table_{dataset}.csv            (flat layout)
DATASET_RUNS: dict[str, Path] = {
    'history':     Path('output/run_1000_final'),
    'amazon':      Path('output/run_1000_final'),
    'arxiv':       Path('output/run_1000_final'),
    'electronics': Path('output/run_1000_final'),
    'toys':        Path('output/run_1000_final'),
}

OUT_DIR   = Path('output/run_1000_final/analysis')
SCORE_COL = 'S_GNN_step1'

DATASETS = ['history', 'amazon', 'arxiv', 'electronics', 'toys']

# Variants where >ZERO_THRESHOLD fraction of train-pool scores are exactly 0.0
# are excluded from tree fitting.  All-NaN variants are always excluded.
ZERO_EXCLUSION_THRESHOLD = 0.95   # was 0.80 in prior run

# Per-dataset task exclusions applied when that dataset is the source (tree fit)
# OR the target (test set) in any experiment — exclusion is symmetric.
# Adding a 4th dataset with exclusions: add one entry here, nowhere else.
PER_DATASET_EXCLUDE_TASKS: dict[str, set] = {
    'arxiv': {'M2', 'M6'},   # graph-leakage: centrality ~91% predictable from degree
}

VARIANT_COLS = ['Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx']


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sep(title: str = '', width: int = 72):
    if title:
        pad = max(0, (width - len(title) - 4) // 2)
        right = max(0, width - pad - len(title) - 4)
        print(f'\n{"═"*pad}  {title}  {"═"*right}')
    else:
        print(f'\n{"─"*width}')


def _display(ds: str) -> str:
    return LABEL_MAP.get(ds, ds.title())


def _load_csv(dataset: str) -> pd.DataFrame:
    run_dir = DATASET_RUNS[dataset]
    fname   = f'construction_performance_table_{dataset}.csv'
    # Try per-dataset subdirectory layout first, then flat layout
    for path in [run_dir / dataset / fname, run_dir / fname]:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f'CSV not found for {dataset!r} — tried:\n'
        f'  {run_dir / dataset / fname}\n'
        f'  {run_dir / fname}'
    )


def _build_summary(
    df: pd.DataFrame,
    dataset: str,
    exclude_tasks: set = None,
    zero_threshold: float = ZERO_EXCLUSION_THRESHOLD,
    verbose: bool = True,
    label: str = '',
) -> pd.DataFrame:
    """
    Compute per-variant train_mean / test_mean from individual sample rows.

    Exclusions (logged):
      1. Task exclusion: rows where Task_Idx in exclude_tasks are dropped first.
      2. All-NaN variants: train_mean or test_mean is NaN (no valid scores).
      3. Zero-inflation: >zero_threshold fraction of train-pool scores are
         exactly 0.0.  Counted against total train rows (including NaN) so the
         threshold is not inflated by NaN rows.
    """
    score_col = SCORE_COL if SCORE_COL in df.columns else 'normalized_score'

    if exclude_tasks:
        df = df[~df['Task_Idx'].isin(exclude_tasks)].copy()

    excluded_allnan = []
    excluded_zero   = []
    rows = []

    for key, grp in df.groupby(VARIANT_COLS):
        vdict = dict(zip(VARIANT_COLS, key))

        train_rows  = grp[grp['run_split'] == 'train']
        test_rows   = grp[grp['run_split'] == 'test']
        train_valid = train_rows[score_col].dropna()
        test_valid  = test_rows[score_col].dropna()

        train_mean = float(train_valid.mean()) if len(train_valid) > 0 else float('nan')
        test_mean  = float(test_valid.mean())  if len(test_valid)  > 0 else float('nan')

        # Exclusion 2: all-NaN
        if np.isnan(train_mean) or np.isnan(test_mean):
            excluded_allnan.append(vdict)
            continue

        # Exclusion 3: zero-inflation (denominator = all train rows, incl. NaN)
        n_train_total = len(train_rows)
        n_train_zero  = int((train_valid == 0.0).sum())
        if n_train_total > 0 and n_train_zero / n_train_total > zero_threshold:
            excluded_zero.append(vdict)
            continue

        # Scale to 0–100 so plot_tree_categorical leaf labels display correctly
        train_mean *= 100.0
        test_mean  *= 100.0

        rows.append({
            **vdict,
            'task_type':     grp['task_type'].iloc[0] if 'task_type' in grp.columns else '',
            'train_mean':    train_mean,
            'test_mean':     test_mean,
            'n_train_valid': int(len(train_valid)),
            'n_test_valid':  int(len(test_valid)),
        })

    summary = pd.DataFrame(rows)

    if verbose:
        tag = f' [{label}]' if label else ''
        n_tasks = len(df['Task_Idx'].unique())
        print(f'  {_display(dataset)}{tag}: {n_tasks} task types | '
              f'{len(summary)} kept | '
              f'{len(excluded_allnan)} all-NaN | '
              f'{len(excluded_zero)} >={zero_threshold*100:.0f}% zero excluded')
        if excluded_allnan:
            for e in excluded_allnan:
                print(f'    ALL-NaN : {e}')
        if excluded_zero:
            for e in excluded_zero:
                print(f'    ZERO    : {e}')

    return summary


def _build_per_sample_summaries(
    df: pd.DataFrame,
    dataset: str,
    exclude_tasks: set = None,
    zero_threshold: float = ZERO_EXCLUSION_THRESHOLD,
) -> dict:
    """
    Build one summary DataFrame per sample index, applying the same exclusion
    logic as _build_summary but scoped to a single sample.

    Returns dict: {sample_idx: summary_df}  where summary_df has columns
        Task_Idx, Node_Idx, Edge_Idx, Text_Idx, task_type,
        train_mean, test_mean
    (only variants present in BOTH splits for that sample are kept)
    """
    score_col = SCORE_COL if SCORE_COL in df.columns else 'normalized_score'

    if exclude_tasks:
        df = df[~df['Task_Idx'].isin(exclude_tasks)].copy()

    # First determine which variants to exclude globally (zero-inflation on full pool)
    excluded_zero = set()
    for key, grp in df.groupby(VARIANT_COLS):
        train_rows  = grp[grp['run_split'] == 'train']
        train_valid = train_rows[score_col].dropna()
        n_total     = len(train_rows)
        n_zero      = int((train_valid == 0.0).sum())
        if n_total > 0 and n_zero / n_total > zero_threshold:
            excluded_zero.add(key)

    per_sample: dict = {}
    for sample_idx, sdf in df.groupby('sample_idx'):
        rows = []
        for key, grp in sdf.groupby(VARIANT_COLS):
            if key in excluded_zero:
                continue
            vdict = dict(zip(VARIANT_COLS, key))
            train_score = grp.loc[grp['run_split'] == 'train', score_col]
            test_score  = grp.loc[grp['run_split'] == 'test',  score_col]
            t_mean = float(train_score.dropna().mean()) if len(train_score.dropna()) > 0 else float('nan')
            e_mean = float(test_score.dropna().mean())  if len(test_score.dropna())  > 0 else float('nan')
            if np.isnan(t_mean) or np.isnan(e_mean):
                continue
            rows.append({
                **vdict,
                'task_type':  grp['task_type'].iloc[0] if 'task_type' in grp.columns else '',
                'train_mean': t_mean * 100.0,
                'test_mean':  e_mean * 100.0,
            })
        if rows:
            per_sample[int(sample_idx)] = pd.DataFrame(rows)

    return per_sample


def _fit_per_sample_trees(per_sample: dict, target_col: str = 'train_mean') -> dict:
    """Fit one decision tree per sample. Returns {sample_idx: tree_result}."""
    trees = {}
    for idx, summ in per_sample.items():
        valid = summ.dropna(subset=['train_mean', 'test_mean'])
        if len(valid) >= 5:
            trees[idx] = fit_tree(valid, target_col=target_col, max_depth=8)
    return trees


def _mean_normalized_ted(trees_a: dict, trees_b: dict) -> tuple:
    """
    Compute mean ± std of normalized TED across all (i, j) pairs from two
    dicts of per-sample trees.  For same-dataset reliability (Exp 1) pass
    the same sample indices; for cross-dataset pass all pairs.

    Returns (mean_norm_ted, std_norm_ted, n_pairs).
    """
    teds = []
    for idx_a, tr_a in trees_a.items():
        for idx_b, tr_b in trees_b.items():
            tree_a = tr_a['tree']
            tree_b = tr_b['tree']
            ted    = _tree_edit_distance(tree_a, tree_b)
            n_a    = tree_a.tree_.node_count
            n_b    = tree_b.tree_.node_count
            total  = n_a + n_b
            if total > 0:
                teds.append(ted / total)
    if not teds:
        return float('nan'), float('nan'), 0
    arr = np.array(teds)
    return float(arr.mean()), float(arr.std()), len(arr)


def _axis_importances(tree_res: dict) -> dict:
    """Aggregate OHE feature importances back to axis level."""
    imp = dict(zip(tree_res['ohe_names'], tree_res['tree'].feature_importances_))
    axis: dict = {}
    for k, v in imp.items():
        col, _ = tree_res['ohe_to_axis_val'][k]
        axis[col] = axis.get(col, 0.0) + v
    return axis


def _importance_table(tree_train_res: dict, tree_test_res: dict) -> str:
    imp_tr = _axis_importances(tree_train_res)
    imp_te = _axis_importances(tree_test_res)
    axes   = sorted(set(imp_tr) | set(imp_te),
                    key=lambda a: imp_tr.get(a, 0), reverse=True)
    lines  = [f'  {"Axis":<20}  {"train":>7}  {"test":>7}  {"Δ":>7}',
              '  ' + '─' * 44]
    for ax in axes:
        tr = imp_tr.get(ax, 0.0)
        te = imp_te.get(ax, 0.0)
        lines.append(f'  {ax:<20}  {tr:7.4f}  {te:7.4f}  {te-tr:+7.4f}')
    return '\n'.join(lines)


def _per_task_rows(pv_result: dict, summary: pd.DataFrame, min_n: int = 5) -> str:
    lines = []
    for tt in sorted(summary['Task_Idx'].unique()):
        mask  = (summary['Task_Idx'] == tt).values
        pred  = pv_result['predictions'][mask]
        act   = pv_result['actual'][mask]
        valid = ~np.isnan(act)
        n     = int(valid.sum())
        if n < min_n:
            lines.append(f'    {tt:<4}  n={n:2d}  (insufficient, need ≥{min_n})')
            continue
        rho, p = spearmanr(pred[valid], act[valid])
        lines.append(f'    {tt:<4}  n={n:3d}  ρ={rho:+.3f}  p={p:.2e}')
    return '\n'.join(lines)


def _print_3x3(rho_df: pd.DataFrame, p_df: pd.DataFrame,
               labels: list, display_fn, title: str):
    """Print a combined rho/p 3×3 table."""
    print(f'\n  {title}')
    col_w = 20
    header = f'  {"":14}' + ''.join(f'  {display_fn(c):>{col_w}}' for c in labels)
    print(header)
    print('  ' + '─' * (14 + (col_w + 2) * len(labels)))
    for r in labels:
        row = f'  {display_fn(r):<14}'
        for c in labels:
            rho = rho_df.loc[r, c]
            p   = p_df.loc[r, c]
            if isinstance(rho, float) and np.isnan(rho):
                cell = f'{"n/a":>{col_w}}'
            else:
                marker = '←' if r == c else ' '
                cell = f'{marker}ρ={rho:+.3f} p={p:.2e}'
                cell = f'{cell:>{col_w}}'
            row += f'  {cell}'
        print(row)


def _m4_inspection(summary: pd.DataFrame, tree_train_res: dict) -> str:
    """Report feature importances and top splits for M4 variants specifically."""
    m4 = summary[summary['Task_Idx'] == 'M4'].copy()
    lines = [f'  M4 subset: {len(m4)} variants']
    if len(m4) < 3:
        lines.append('  (too few M4 variants for sub-tree)')
        return '\n'.join(lines)

    # Axis importances from the full tree on M4 predictions
    from sklearn.tree import _tree as _sk_tree
    tree      = tree_train_res['tree']
    ohe       = tree_train_res['ohe']
    ohe_names = tree_train_res['ohe_names']
    feat_cols = tree_train_res['feature_cols']

    X_m4   = ohe.transform(m4[feat_cols])
    pred_m4 = tree.predict(X_m4)
    act_m4  = m4['test_mean'].values

    valid = ~np.isnan(act_m4)
    if valid.sum() >= 3:
        rho, p = spearmanr(pred_m4[valid], act_m4[valid])
        lines.append(f'  Spearman ρ (full tree preds on M4 test_mean): {rho:+.3f}  p={p:.2e}')

    # Fit a small dedicated tree just for M4 to expose its own splits
    if len(m4) >= 5:
        m4_tree_res = fit_tree(m4, target_col='train_mean', max_depth=4)
        imp = _axis_importances(m4_tree_res)
        lines.append(f'  M4-only tree (depth≤4, n={len(m4)}) feature importances:')
        for ax, v in sorted(imp.items(), key=lambda x: x[1], reverse=True):
            lines.append(f'    {ax:<20}  {v:.4f}')

        # Walk top splits of the M4-only tree
        t   = m4_tree_res['tree'].tree_
        names = m4_tree_res['ohe_names']
        lines.append('  Top splits (up to depth 2):')
        for node in range(t.node_count):
            if t.feature[node] == _sk_tree.TREE_UNDEFINED:
                continue
            depth = 0
            n = node
            while n != 0:
                parent = np.where(
                    (t.children_left == n) | (t.children_right == n))[0]
                if len(parent) == 0:
                    break
                n = parent[0]
                depth += 1
            if depth > 2:
                continue
            feat_name = names[t.feature[node]] if t.feature[node] >= 0 else '?'
            n_left  = int(t.n_node_samples[t.children_left[node]])
            n_right = int(t.n_node_samples[t.children_right[node]])
            lines.append(f'    depth={depth} node={node}  split={feat_name}  '
                         f'left n={n_left}  right n={n_right}  '
                         f'impurity_dec={t.impurity[node] - t.impurity[t.children_left[node]]:.4f}')
    return '\n'.join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--open', dest='open_fig', action='store_true')
    parser.add_argument('--run-dir', default=None,
                         help='Run directory containing construction_performance_table_*.csv '
                              '(flat layout). Defaults to the hardcoded DATASET_RUNS/OUT_DIR '
                              '(output/run_1000_final) when omitted.')
    args = parser.parse_args()

    global DATASET_RUNS, OUT_DIR
    if args.run_dir:
        run_dir = Path(args.run_dir)
        DATASET_RUNS = {ds: run_dir for ds in DATASETS}
        OUT_DIR = run_dir / 'analysis'

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── File confirmation ──────────────────────────────────────────────────────
    _sep('FILE CONFIRMATION')
    raw_dfs: dict[str, pd.DataFrame] = {}
    for ds in DATASETS:
        df = _load_csv(ds)
        raw_dfs[ds] = df
        sc = SCORE_COL if SCORE_COL in df.columns else 'normalized_score'
        n_variants = df.groupby(VARIANT_COLS).ngroups
        n_samples  = df.groupby(['run_split', 'sample_idx']).ngroups
        degen_pct  = 100 * df['degenerate'].mean() if 'degenerate' in df.columns else float('nan')
        nan_pct    = 100 * df[sc].isna().mean()
        task_cts   = df['Task_Idx'].value_counts().sort_index().to_dict()
        print(f'{_display(ds)}:  {len(df):,} rows  |  {n_variants} variants  |  '
              f'{n_samples} (split,sample)  |  score={sc}  |  '
              f'{degen_pct:.1f}% degen  |  {nan_pct:.1f}% NaN')
        print(f'  task counts: {task_cts}')

    # ── Data prep ─────────────────────────────────────────────────────────────
    _sep(f'DATA PREP  (zero threshold = {ZERO_EXCLUSION_THRESHOLD*100:.0f}%)')

    # Standard summaries — no extra task exclusion
    summaries: dict[str, pd.DataFrame] = {}
    for ds in DATASETS:
        summaries[ds] = _build_summary(raw_dfs[ds], ds)

    # Clean summaries — per-dataset task exclusions applied
    summaries_clean: dict[str, pd.DataFrame] = {}
    for ds in DATASETS:
        excl = PER_DATASET_EXCLUDE_TASKS.get(ds)
        if excl:
            print(f'\n  {_display(ds)} clean (excluding {excl}):')
            summaries_clean[ds] = _build_summary(
                raw_dfs[ds], ds, exclude_tasks=excl, label=f'excl {excl}')
        else:
            summaries_clean[ds] = summaries[ds]   # no exclusions for this dataset

    # ── Fit trees (standard and clean per dataset) ────────────────────────────
    # Trees keyed by (dataset, variant: 'std'|'clean')
    trees_train: dict[tuple, dict] = {}
    trees_test:  dict[tuple, dict] = {}

    def _fit_both(key: tuple, summ: pd.DataFrame, label: str):
        valid = summ.dropna(subset=['train_mean', 'test_mean'])
        if len(valid) < 5:
            print(f'  SKIP {label}: only {len(valid)} fully-valid variants')
            return
        tr = fit_tree(valid, target_col='train_mean', max_depth=8)
        te = fit_tree(valid, target_col='test_mean',  max_depth=8)
        trees_train[key] = tr
        trees_test[key]  = te

    for ds in DATASETS:
        _fit_both((ds, 'std'),   summaries[ds],       f'{_display(ds)} std')
        if ds in PER_DATASET_EXCLUDE_TASKS:
            _fit_both((ds, 'clean'), summaries_clean[ds], f'{_display(ds)} clean')

    # ── Experiment 1 — Reliability ────────────────────────────────────────────
    _sep('EXPERIMENT 1 — RELIABILITY (train tree vs test tree)')

    exp1: dict[tuple, dict] = {}

    # Per-sample summaries and trees for TED (train vs test, per sample)
    per_sample_summaries: dict[str, dict] = {}
    per_sample_train_trees: dict[str, dict] = {}
    per_sample_test_trees:  dict[str, dict] = {}
    for ds in DATASETS:
        excl = PER_DATASET_EXCLUDE_TASKS.get(ds)
        ps   = _build_per_sample_summaries(raw_dfs[ds], ds, exclude_tasks=excl)
        per_sample_summaries[ds]    = ps
        per_sample_train_trees[ds]  = _fit_per_sample_trees(ps, target_col='train_mean')
        per_sample_test_trees[ds]   = _fit_per_sample_trees(ps, target_col='test_mean')

    for ds in DATASETS:
        variants_to_run = [('std', summaries[ds])]
        if ds in PER_DATASET_EXCLUDE_TASKS:
            variants_to_run.append(('clean', summaries_clean[ds]))

        for variant, summ in variants_to_run:
            key  = (ds, variant)
            disp = _display(ds) + ('' if variant == 'std'
                                   else f' (excl {PER_DATASET_EXCLUDE_TASKS[ds]})')
            if key not in trees_train:
                print(f'\n  ── {disp} ── SKIP (tree not fitted)')
                continue

            valid = summ.dropna(subset=['train_mean', 'test_mean'])
            tr    = trees_train[key]
            te    = trees_test[key]
            cmp   = compare_trees(tr, te, valid)
            exp1[key] = cmp

            # Per-sample TED: pair each train-sample tree with its test-sample tree
            ps_tr  = per_sample_train_trees[ds]
            ps_te  = per_sample_test_trees[ds]
            # Align on same sample indices (i vs i)
            shared = sorted(set(ps_tr) & set(ps_te))
            aligned_tr = {i: ps_tr[i] for i in shared}
            aligned_te = {i: ps_te[i] for i in shared}
            ted_mean, ted_std, ted_n = _mean_normalized_ted(aligned_tr, aligned_te)

            print(f'\n  ── {disp} ──')
            print(f'  tree_train: R²={tr["r2"]:.4f}  '
                  f'depth={tr["tree"].get_depth()}  '
                  f'leaves={tr["tree"].get_n_leaves()}  '
                  f'n={len(valid)}')
            print(f'  tree_test:  R²={te["r2"]:.4f}  '
                  f'depth={te["tree"].get_depth()}  '
                  f'leaves={te["tree"].get_n_leaves()}')
            # Normalized TED between aggregate tree_train and tree_test
            agg_ted_raw  = cmp["tree_edit_distance"]
            n_tr_nodes   = tr['tree'].tree_.node_count
            n_te_nodes   = te['tree'].tree_.node_count
            total_nodes  = n_tr_nodes + n_te_nodes
            agg_norm_ted = agg_ted_raw / total_nodes if total_nodes > 0 else float('nan')

            print(f'  spearman ρ (train-pred vs test-pred): '
                  f'{cmp["spearman_rho"]:+.4f}  p={cmp["spearman_p"]:.2e}')
            print(f'  band_accuracy : {cmp["band_accuracy"]:.4f}')
            print(f'  TED (agg tree_train vs tree_test): '
                  f'raw={agg_ted_raw}  '
                  f'nodes=({n_tr_nodes}+{n_te_nodes}={total_nodes})  '
                  f'norm={agg_norm_ted:.4f}')
            print(f'  norm_TED (per-sample mean ± std, n={ted_n} pairs): '
                  f'{ted_mean:.4f} ± {ted_std:.4f}')
            print('\n  Feature importances (train | test | Δ):')
            print(_importance_table(tr, te))

    # ── Experiment 2 — Strategy 1 Validation ──────────────────────────────────
    _sep('EXPERIMENT 2 — STRATEGY 1 VALIDATION (tree_train → test_mean)')

    exp2: dict[tuple, dict] = {}

    for ds in DATASETS:
        variants_to_run = [('std', summaries[ds])]
        if ds in PER_DATASET_EXCLUDE_TASKS:
            variants_to_run.append(('clean', summaries_clean[ds]))

        for variant, summ in variants_to_run:
            key  = (ds, variant)
            disp = _display(ds) + ('' if variant == 'std'
                                   else f' (excl {PER_DATASET_EXCLUDE_TASKS[ds]})')
            if key not in trees_train:
                print(f'\n  ── {disp} ── SKIP')
                continue

            valid = summ.dropna(subset=['train_mean', 'test_mean'])
            pv    = predict_and_validate(trees_train[key], valid, target_col='test_mean')
            exp2[key] = pv

            print(f'\n  ── {disp} ──')
            print(f'  spearman ρ   : {pv["spearman_rho"]:+.4f}  p={pv["spearman_p"]:.2e}')
            print(f'  band_accuracy: {pv["band_accuracy"]:.4f}')
            print(f'  n_variants   : {len(valid)}')
            print('\n  Per Task_Idx (n ≥ 5 required, flagged otherwise):')
            print(_per_task_rows(pv, valid))

            # M4 inspection — run for any dataset that has M4 variants
            if variant == 'std' and 'M4' in valid['Task_Idx'].values:
                m4_sub = valid[valid['Task_Idx'] == 'M4']
                if len(m4_sub) >= 1:
                    print(f'\n  {_display(ds)} M4 inspection:')
                    print(_m4_inspection(valid, trees_train[key]))

    # ── Experiment 3 — Cross-dataset generalisation ───────────────────────────
    n = len(DATASETS)
    _sep(f'EXPERIMENT 3 — CROSS-DATASET GENERALISATION ({n}×{n} tables)')

    def _build_3x3(tree_key_fn, target_summ_fn, labels, ps_train_trees_fn=None):
        """
        Fill a 5×5 rho/p/TED table.

        TED is mean normalised TED across all (sample_i from row-dataset,
        sample_j from col-dataset) pairs.  Pass ps_train_trees_fn(ds) to
        retrieve the per-sample train-tree dict for dataset ds.
        Diagonal TED uses the paired train-sample tree vs test-sample tree
        (i.e., reliability: same index, different split).
        """
        rho_df = pd.DataFrame(index=labels, columns=labels, dtype=float)
        p_df   = pd.DataFrame(index=labels, columns=labels, dtype=float)
        ted_df = pd.DataFrame(index=labels, columns=labels, dtype=float)

        for r in labels:
            tr_key = tree_key_fn(r)
            if tr_key not in trees_train:
                continue
            tr = trees_train[tr_key]
            for c in labels:
                tgt = target_summ_fn(c).dropna(subset=['test_mean'])
                if len(tgt) < 5:
                    continue

                # ρ / p — may fail when feature spaces differ across datasets
                try:
                    pv = predict_and_validate(tr, tgt, target_col='test_mean')
                    rho_df.loc[r, c] = round(pv['spearman_rho'], 4)
                    p_df.loc[r,   c] = round(pv['spearman_p'],   4)
                except Exception as e:
                    print(f'  WARN {_display(r)}→{_display(c)}: {e}')

                # TED — per-sample, averaged across all (i, j) pairs
                if ps_train_trees_fn is not None:
                    ps_r = ps_train_trees_fn(r)   # per-sample train trees for row dataset
                    if r == c:
                        # Diagonal: train-sample i vs test-sample i (reliability)
                        ps_te = per_sample_test_trees[r]
                        shared = sorted(set(ps_r) & set(ps_te))
                        ps_r_aligned = {i: ps_r[i] for i in shared}
                        ps_te_aligned = {i: ps_te[i] for i in shared}
                        m, _, _ = _mean_normalized_ted(ps_r_aligned, ps_te_aligned)
                    else:
                        # Off-diagonal: all (i, j) pairs across the two datasets
                        ps_c = ps_train_trees_fn(c)
                        m, _, _ = _mean_normalized_ted(ps_r, ps_c)
                    if not np.isnan(m):
                        ted_df.loc[r, c] = round(m, 4)

        return rho_df, p_df, ted_df

    # Table A — with ArXiv M2/M6 included (standard summaries throughout)
    rho_a, p_a, ted_a = _build_3x3(
        tree_key_fn       = lambda ds: (ds, 'std'),
        target_summ_fn    = lambda ds: summaries[ds],
        labels            = DATASETS,
        ps_train_trees_fn = lambda ds: per_sample_train_trees[ds],
    )

    # Table B — ArXiv M2/M6 excluded symmetrically:
    #   when ArXiv is the row (tree source) → use clean tree
    #   when ArXiv is the col (test target) → use clean summary
    #   non-ArXiv rows/cols → unchanged
    def _clean_tree_key(ds):
        return (ds, 'clean') if ds in PER_DATASET_EXCLUDE_TASKS and (ds, 'clean') in trees_train \
               else (ds, 'std')

    def _clean_target(ds):
        return summaries_clean[ds] if ds in PER_DATASET_EXCLUDE_TASKS \
               else summaries[ds]

    rho_b, p_b, ted_b = _build_3x3(
        tree_key_fn       = _clean_tree_key,
        target_summ_fn    = _clean_target,
        labels            = DATASETS,
        ps_train_trees_fn = lambda ds: per_sample_train_trees[ds],
    )

    # Print both tables
    _print_3x3(rho_a, p_a, DATASETS, _display,
               f'Table A ({n}×{n}) — ArXiv M2/M6 INCLUDED (standard summaries)')
    _print_3x3(rho_b, p_b, DATASETS, _display,
               f'Table B ({n}×{n}) — ArXiv M2/M6 EXCLUDED symmetrically (tree source and test target)')

    # Print edit distance tables (normalized TED = edit_distance / total_nodes)
    # * marks cells where ρ is significant at p < 0.05
    col_w = 16
    for tbl_lbl, ted_df, p_df in [
        (f'Edit Distance Table A ({n}×{n}) — normalized TED (all off-diagonal; * = p<0.05)', ted_a, p_a),
        (f'Edit Distance Table B ({n}×{n}) — normalized TED (all off-diagonal; * = p<0.05)', ted_b, p_b),
    ]:
        print(f'\n  {tbl_lbl}')
        print(f'  {"":14}' + ''.join(f'  {_display(c):>{col_w}}' for c in DATASETS))
        print('  ' + '─' * (14 + (col_w + 2) * len(DATASETS)))
        for r in DATASETS:
            row_str = f'  {_display(r):<14}'
            for c in DATASETS:
                if r == c:
                    cell = f'{"—":>{col_w}}'
                else:
                    ted = ted_df.loc[r, c]
                    sig = not np.isnan(float(p_df.loc[r, c])) and float(p_df.loc[r, c]) < 0.05
                    if isinstance(ted, float) and not np.isnan(ted):
                        cell = f'{"*" if sig else " "}{ted:.4f}'
                        cell = f'{cell:>{col_w}}'
                    else:
                        cell = f'{"—":>{col_w}}'
                row_str += f'  {cell}'
            print(row_str)

    # Highlight specific cells of interest
    for r_ds, c_ds, label in [
        ('arxiv', 'history', 'ArXiv→History'),
        ('amazon', 'arxiv',  'Amazon→ArXiv'),
    ]:
        for tbl_name, rho_df, p_df, ted_df in [('A (incl)', rho_a, p_a, ted_a),
                                                 ('B (excl)', rho_b, p_b, ted_b)]:
                rho_v = rho_df.loc[r_ds, c_ds]
                p_v   = p_df.loc[r_ds, c_ds]
                ted_v = ted_df.loc[r_ds, c_ds]
                if not (isinstance(rho_v, float) and np.isnan(rho_v)):
                    ted_str = f'  normTED={ted_v:.4f}' if isinstance(ted_v, float) and not np.isnan(ted_v) else ''
                    print(f'  {label} [{tbl_name}]:  ρ={rho_v:+.4f}  p={p_v:.2e}{ted_str}')

    # Save CSVs
    rho_a.to_csv(OUT_DIR / 'cross_dataset_3x3_with_rho.csv')
    p_a.to_csv(  OUT_DIR / 'cross_dataset_3x3_with_p.csv')
    ted_a.to_csv(OUT_DIR / 'cross_dataset_3x3_with_ted.csv')
    rho_b.to_csv(OUT_DIR / 'cross_dataset_3x3_clean_rho.csv')
    p_b.to_csv(  OUT_DIR / 'cross_dataset_3x3_clean_p.csv')
    ted_b.to_csv(OUT_DIR / 'cross_dataset_3x3_clean_ted.csv')
    print(f'\n  Saved: cross_dataset_3x3_{{with,clean}}_{{rho,p,ted}}.csv → {OUT_DIR}')

    # ── Tree PNGs ──────────────────────────────────────────────────────────────
    _sep('TREE PNGs')

    for ds in DATASETS:
        variants_to_render = [('std', summaries[ds], '')]
        if ds in PER_DATASET_EXCLUDE_TASKS:
            excl_label = f'excl {PER_DATASET_EXCLUDE_TASKS[ds]}'
            variants_to_render.append(
                ('clean', summaries_clean[ds], '_clean'))

        for variant, summ, suffix in variants_to_render:
            key = (ds, variant)
            if key not in trees_train:
                continue

            valid   = summ.dropna(subset=['train_mean', 'test_mean'])
            tr      = trees_train[key]
            pv      = exp2.get(key)
            rho_str = f'{pv["spearman_rho"]:+.3f}' if pv else 'n/a'

            excl_str = ''
            if ds in PER_DATASET_EXCLUDE_TASKS and variant == 'clean':
                excl_str = f' — excl. {PER_DATASET_EXCLUDE_TASKS[ds]}'
            disp   = _display(ds) + excl_str
            out_png = OUT_DIR / f'decision_tree_{ds}{suffix}_combined.png'

            title = (
                f'TAG Construction Decision Tree — {disp} (all task types)\n'
                f'target = train_mean  '
                f'depth ≤ {tr["tree"].get_depth()}  '
                f'R²(train) = {tr["r2"]:.3f}  '
                f'ρ(train→test) = {rho_str}  '
                f'N = {len(valid)} variants'
            )

            plot_tree_categorical(
                tree_model      = tr['tree'],
                ohe_names       = tr['ohe_names'],
                ohe_to_axis_val = tr['ohe_to_axis_val'],
                title           = title,
                out_path        = out_png,
                open_fig        = args.open_fig,
            )
            print(f'  {out_png}')

    _sep('DONE')
    print(f'  Output: {OUT_DIR}')


if __name__ == '__main__':
    main()
