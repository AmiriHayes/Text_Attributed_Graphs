#!/usr/bin/env python3
"""
Checks whether M1 (node-classification) decision trees generalize to M3/M4
(edge-level) task types, via tree-structure edit distance and variant-ranking
Spearman correlation, on the existing run_post_audit construction
performance tables. No new training or API calls -- purely a re-analysis of
already-computed train_mean scores.

Reads:
  - output/run_post_audit/construction_performance_table_{dataset}.csv
    (history, arxiv, electronics, toys -- Amazon M3 skipped per instruction,
    too few variants)

Writes:
  - data/graphrag/m1_vs_m3m4_generalization.csv

Usage:
  python code/m1_vs_m3m4_generalization.py
"""
import sys
import importlib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ra = importlib.import_module('run_analysis')
from decision_tree_analysis import fit_tree, _tree_edit_distance, LABEL_MAP

ra.DATASET_RUNS = {ds: REPO_ROOT / 'output/run_post_audit'
                  for ds in ['history', 'arxiv', 'electronics', 'toys', 'amazon']}

DATASETS_TASKS = {
    'history':     ['M1', 'M3', 'M4'],
    'arxiv':       ['M1', 'M3'],       # arxiv has no M4 variants at all
    'electronics': ['M1', 'M3', 'M4'],
    'toys':        ['M1', 'M3', 'M4'],
    # amazon M3 explicitly skipped -- too few variants (per instruction)
}
N_TRUST_THRESHOLD = 5
FEATURES = ['Node_Idx', 'Edge_Idx', 'Text_Idx']  # Task_Idx excluded by design


def get_summary(dataset: str, task: str) -> pd.DataFrame:
    raw = ra._load_csv(dataset)
    sub = raw[raw['Task_Idx'] == task]
    if len(sub) == 0:
        return pd.DataFrame()
    return ra._build_summary(sub, dataset, verbose=False)


def first_split_axis(tree_res: dict) -> str:
    """Human-readable feature + category of the tree's root split."""
    t = tree_res['tree'].tree_
    if t.feature[0] == -2:
        return '(no split -- single leaf)'
    feat_name = tree_res['ohe_names'][t.feature[0]]
    axis_col, cat_val = tree_res['ohe_to_axis_val'][feat_name]
    return f'{axis_col}={cat_val}'


def main():
    rows = []
    all_trees = {}    # (dataset, task) -> tree_res or None
    all_summ = {}      # (dataset, task) -> summary df

    print('=' * 78)
    print('STEP 1 -- FIT TREES FOR M1, M3, M4 (features: N_Idx, E_Idx, T_Idx only)')
    print('=' * 78)

    for dataset, tasks in DATASETS_TASKS.items():
        for task in tasks:
            summ = get_summary(dataset, task)
            all_summ[(dataset, task)] = summ
            n = len(summ)
            trust_flag = '' if n >= N_TRUST_THRESHOLD else f'  ** UNTRUSTWORTHY (n<{N_TRUST_THRESHOLD}) **'
            print(f'\n{dataset} {task}: n={n} surviving variants{trust_flag}')

            if n < 3:
                print('  Cannot fit a tree (n<3) -- skipping tree fit entirely.')
                all_trees[(dataset, task)] = None
                continue

            valid = summ.dropna(subset=['train_mean'])
            tr = fit_tree(valid, target_col='train_mean', features=FEATURES, max_depth=8)
            all_trees[(dataset, task)] = tr

            imp = {}
            for ohe_name, v in zip(tr['ohe_names'], tr['tree'].feature_importances_):
                col, _ = tr['ohe_to_axis_val'][ohe_name]
                imp[col] = imp.get(col, 0.0) + v
            imp_str = ', '.join(f'{k}={v:.3f}' for k, v in sorted(imp.items(), key=lambda x: -x[1]))
            print(f'  feature importances: {imp_str}')
            print(f'  first split: {first_split_axis(tr)}')
            print(f'  R2(train)={tr["r2"]:.3f}  leaves={tr["tree"].get_n_leaves()}  depth={tr["tree"].get_depth()}')

    print('\n' + '=' * 78)
    print('STEP 2 -- TREE STRUCTURE COMPARISON (normalized TED)')
    print('=' * 78)

    ted_results = {}
    for dataset, tasks in DATASETS_TASKS.items():
        for other in ['M3', 'M4']:
            if other not in tasks:
                continue
            tr_m1 = all_trees.get((dataset, 'M1'))
            tr_other = all_trees.get((dataset, other))
            if tr_m1 is None or tr_other is None:
                ted_results[(dataset, other)] = None
                continue
            n_m1 = len(all_summ[(dataset, 'M1')])
            n_other = len(all_summ[(dataset, other)])
            raw_ted = _tree_edit_distance(tr_m1['tree'], tr_other['tree'])
            total_nodes = tr_m1['tree'].tree_.node_count + tr_other['tree'].tree_.node_count
            norm_ted = raw_ted / total_nodes if total_nodes > 0 else float('nan')
            untrust = n_m1 < N_TRUST_THRESHOLD or n_other < N_TRUST_THRESHOLD
            ted_results[(dataset, other)] = {'norm_ted': norm_ted, 'raw_ted': raw_ted,
                                             'n_m1': n_m1, 'n_other': n_other, 'untrust': untrust}
            flag = '  ** LOW-N, UNTRUSTWORTHY **' if untrust else ''
            print(f'{dataset} M1 vs {other}: norm_TED={norm_ted:.3f} (raw={raw_ted}, '
                  f'n_M1={n_m1}, n_{other}={n_other}){flag}')

    print('\n' + '=' * 78)
    print('STEP 3 -- VARIANT RANKING AGREEMENT (Spearman rho, shared variants only)')
    print('=' * 78)

    rank_results = {}
    for dataset, tasks in DATASETS_TASKS.items():
        for other in ['M3', 'M4']:
            if other not in tasks:
                continue
            s1 = all_summ[(dataset, 'M1')].copy()
            s2 = all_summ[(dataset, other)].copy()
            s1['key'] = s1['Node_Idx'] + '_' + s1['Edge_Idx'] + '_' + s1['Text_Idx']
            s2['key'] = s2['Node_Idx'] + '_' + s2['Edge_Idx'] + '_' + s2['Text_Idx']
            merged = s1[['key', 'train_mean']].merge(
                s2[['key', 'train_mean']], on='key', suffixes=('_m1', f'_{other.lower()}'))
            n_shared = len(merged)
            untrust = n_shared < N_TRUST_THRESHOLD
            if n_shared < 3:
                rank_results[(dataset, other)] = {'n': n_shared, 'rho': float('nan'), 'p': float('nan'), 'untrust': True}
                print(f'{dataset} M1 vs {other}: n_shared={n_shared} -- too few to compute rho at all')
                continue
            rho, p = spearmanr(merged['train_mean_m1'], merged[f'train_mean_{other.lower()}'])
            rank_results[(dataset, other)] = {'n': n_shared, 'rho': float(rho), 'p': float(p), 'untrust': untrust}
            flag = '  ** LOW-N, UNTRUSTWORTHY **' if untrust else ''
            print(f'{dataset} M1 vs {other}: n_shared={n_shared} rho={rho:+.3f} p={p:.4f}{flag}')

    # ── Summary table ──────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('SUMMARY TABLE')
    print('=' * 78)

    out_rows = []
    for dataset, tasks in DATASETS_TASKS.items():
        for other in ['M3', 'M4']:
            if other not in tasks:
                continue
            ted = ted_results.get((dataset, other))
            rank = rank_results.get((dataset, other))
            tr_m1 = all_trees.get((dataset, 'M1'))
            tr_other = all_trees.get((dataset, other))
            row = {
                'dataset': dataset, 'task_pair': f'M1_vs_{other}',
                'shared_n': rank['n'] if rank else None,
                'ranking_rho': round(rank['rho'], 3) if rank and not np.isnan(rank['rho']) else None,
                'ranking_p': round(rank['p'], 4) if rank and not np.isnan(rank['p']) else None,
                'tree_ted': round(ted['norm_ted'], 3) if ted else None,
                'm1_first_split': first_split_axis(tr_m1) if tr_m1 else None,
                'other_first_split': first_split_axis(tr_other) if tr_other else None,
                'untrustworthy_low_n': bool((rank and rank.get('untrust')) or (ted and ted.get('untrust'))),
            }
            out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    print(out_df.to_string(index=False))

    out_csv = REPO_ROOT / 'data/graphrag/m1_vs_m3m4_generalization.csv'
    out_df.to_csv(out_csv, index=False)
    print(f'\nsaved -> {out_csv}')

    # ── Verdict ──────────────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('VERDICT')
    print('=' * 78)

    trustworthy = out_df[~out_df['untrustworthy_low_n']]
    untrustworthy = out_df[out_df['untrustworthy_low_n']]
    print(f'{len(untrustworthy)}/{len(out_df)} comparisons flagged untrustworthy (n<{N_TRUST_THRESHOLD}):')
    for _, r in untrustworthy.iterrows():
        print(f'  {r["dataset"]} {r["task_pair"]}: shared_n={r["shared_n"]}')

    if len(trustworthy) == 0:
        print('\nNo comparison in this dataset has enough shared variants (n>=5) to trust.')
        print('Cannot make the rho>0.5-and-TED<0.3 / rho<0.3-or-TED>0.5 call on reliable data.')
        recommendation = ('Do not report an M1-to-M3/M4 generalization claim in the paper -- '
                          'every M3/M4 survivor count in run_post_audit is below 5 (as low as n=2 '
                          'for Toys M4), too small to support either a positive or negative claim.')
    else:
        good = trustworthy[(trustworthy['ranking_rho'] > 0.5) & (trustworthy['tree_ted'] < 0.3)]
        bad = trustworthy[(trustworthy['ranking_rho'] < 0.3) | (trustworthy['tree_ted'] > 0.5)]
        print(f'\nOf {len(trustworthy)} trustworthy comparison(s): {len(good)} support generalization, '
              f'{len(bad)} contradict it.')
        if len(good) > len(trustworthy) / 2:
            recommendation = 'M1 trees give useful guidance for M3/M4 task types, on the (few) trustworthy comparisons available.'
        elif len(bad) > len(trustworthy) / 2:
            recommendation = 'M1 and M3/M4 learn different construction rules -- do not extend M1-based recommendations to edge-level tasks.'
        else:
            recommendation = 'Evidence is mixed even among the trustworthy comparisons -- do not make a blanket generalization claim.'

    print(f'\nRECOMMENDATION FOR THE PAPER:\n  {recommendation}')


if __name__ == '__main__':
    main()
