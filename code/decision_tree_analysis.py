#!/usr/bin/env python3
"""
Decision tree analysis for TAG construction choices.

Features are one-hot encoded so every split is an exact equality check
("Is task M1? Yes / No") rather than a range over ordinal indices.
Tree is fit on 20 TRAIN-split variant means and evaluated on 20 TEST-split means.

Usage
-----
    python3 code/decision_tree_analysis.py --dataset history
    python3 code/decision_tree_analysis.py --dataset amazon
    python3 code/decision_tree_analysis.py --dataset arxiv
    python3 code/decision_tree_analysis.py --dataset combined
    python3 code/decision_tree_analysis.py --dataset all
    python3 code/decision_tree_analysis.py --dataset all --depth 4 --open
"""
import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

DATA_DIR   = Path('output')
TREE_DIR   = Path('output/decision_trees')
SCORE_COL  = 'normalized_score'
GROUP_BASE = ['Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx']

# ── label maps ────────────────────────────────────────────────────────────────

LABEL_MAP = {
    'M1': 'Node Categ.',       'M2': 'Node Scalar',
    'M3': 'Edge Categ.',       'M4': 'Edge Scalar',
    'M5': 'Global Categ.',     'M6': 'Global Scalar',
    'N7': 'Primary Entity',    'N8': 'Secondary Entity',   'N9': 'Aggregate Entity',
    'E10a': 'Categ. GT',       'E10b': 'Co-partic. (wt.)',
    'E10c': 'Binary Partic.',  'E11a': 'Semantic Sim.',
    'E11b': 'Structural Sim.', 'E11c': 'Functional Sim.',
    'T12a': 'Contextual',      'T12b': 'Standard',         'T12e': 'Baseline',
    'amazon': 'Amazon',        'arxiv': 'ArXiv',           'history': 'History',
}

# Full-phrase labels for the PNG node boxes
FULL_LABEL = {
    'M1': 'Node Level\nCategorical',    'M2': 'Node Level\nScalar',
    'M3': 'Edge Level\nCategorical',    'M4': 'Edge Level\nScalar',
    'M5': 'Global\nCategorical',        'M6': 'Global\nScalar',
    'N7': 'Primary Entity',             'N8': 'Secondary Entity',
    'N9': 'Aggregate Entity',
    'E10a': 'Categorical GT',           'E10b': 'Weighted\nCo-participation',
    'E10c': 'Binary Participation',     'E11a': 'Semantic Similarity',
    'E11b': 'Structural Similarity',    'E11c': 'Functional Similarity',
    'T12a': 'Contextual\n(Title + Body)', 'T12b': 'Standard\n(Title Only)',
    'T12e': 'Baseline\n(Zero Embeds)',
    'amazon': 'Amazon',  'arxiv': 'ArXiv',  'history': 'History',
}

AXIS_PREFIX = {
    'Task_Idx': 'TASK',   'Node_Idx': 'NODE',
    'Edge_Idx': 'EDGE',   'Text_Idx': 'TEXT',
    'dataset':  'DATASET',
}

FEATURE_DISPLAY = {
    'Task_Idx': 'Task Type (M)',   'Node_Idx': 'Node Type (N)',
    'Edge_Idx': 'Edge Type (E)',   'Text_Idx': 'Text Fidelity (T)',
    'dataset':  'Dataset',
}


# ── color helpers ─────────────────────────────────────────────────────────────

def _internal_color(val: float):
    t = max(0.0, min(1.0, val / 100.0))
    return plt.colormaps['YlOrBr'](0.15 + t * 0.72)

def _leaf_color(val: float):
    t = max(0.0, min(1.0, val / 100.0))
    g = 0.08 + t * 0.87
    return (g, g, g, 1.0)

def _leaf_text_color(val: float) -> str:
    return 'white' if val / 100.0 < 0.55 else '#1a1a1a'

def _internal_text_color(val: float) -> str:
    return 'white' if val / 100.0 > 0.72 else '#1a1a1a'


# ── encoding ──────────────────────────────────────────────────────────────────

def build_ohe(variant_df: pd.DataFrame, feature_cols: list):
    """
    One-hot encode feature_cols so every tree split is an exact equality check.
    Returns (X, ohe_feature_names, ohe_to_axis_val).

    ohe_to_axis_val maps each OHE column name → (original_col, category_value)
    e.g. 'Task_Idx_M1' → ('Task_Idx', 'M1')
    """
    try:
        ohe = OneHotEncoder(sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(sparse=False)

    X = ohe.fit_transform(variant_df[feature_cols])
    ohe_names = list(ohe.get_feature_names_out(feature_cols))

    # Build reverse lookup: ohe column name → (axis_col, cat_value)
    ohe_to_axis_val = {}
    for i, col in enumerate(feature_cols):
        for cat in ohe.categories_[i]:
            ohe_to_axis_val[f'{col}_{cat}'] = (col, str(cat))

    return X, ohe_names, ohe_to_axis_val


# ── layout ────────────────────────────────────────────────────────────────────

def _leaf_positions(tree_model, node: int = 0, counter: list = None) -> dict:
    if counter is None:
        counter = [0]
    t = tree_model.tree_
    if t.feature[node] == -2:
        pos = {node: float(counter[0])}
        counter[0] += 1
        return pos
    left  = _leaf_positions(tree_model, t.children_left[node],  counter)
    right = _leaf_positions(tree_model, t.children_right[node], counter)
    lx = left[t.children_left[node]]
    rx = right[t.children_right[node]]
    pos = {node: (lx + rx) / 2.0}
    pos.update(left)
    pos.update(right)
    return pos


# ── custom tree plotter ───────────────────────────────────────────────────────

def plot_tree_categorical(
    tree_model,
    ohe_names: list,
    ohe_to_axis_val: dict,
    title: str = '',
    out_path=None,
    open_fig: bool = False,
):
    """
    Draw the decision tree. Every internal node shows:
        AXIS: Full Description     ← the single category being tested
        N=42,  Val=74%
    Left branch = NO (feature == 0), Right branch = YES (feature == 1).
    Leaf nodes are grayscale-colored by score.
    """
    t        = tree_model.tree_
    n_leaves = tree_model.get_n_leaves()
    depth    = tree_model.get_depth()
    node_x   = _leaf_positions(tree_model)

    FIG_W = max(14, 3.0 * n_leaves)
    FIG_H = max(8,  2.8 * (depth + 1))
    fig, ax = plt.subplots(figsize=(min(FIG_W, 68), FIG_H))

    MARGIN = 0.7
    ax.set_xlim(-MARGIN, n_leaves - 1 + MARGIN)
    ax.set_ylim(-depth - MARGIN, 0.6)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def draw(node: int, d: int):
        x   = node_x[node]
        y   = float(-d)
        n   = int(t.n_node_samples[node])
        val = float(t.value[node][0][0])

        if t.feature[node] == -2:
            # ── LEAF ─────────────────────────────────────────────────
            ax.text(
                x, y, f'{val:.0f}%',
                ha='center', va='center',
                fontsize=15, fontweight='bold',
                color=_leaf_text_color(val),
                bbox=dict(
                    boxstyle='round,pad=0.55',
                    facecolor=_leaf_color(val),
                    edgecolor='#888888', linewidth=1.4,
                ),
                zorder=3,
            )
        else:
            # ── INTERNAL NODE ─────────────────────────────────────────
            feat_name          = ohe_names[t.feature[node]]
            axis_col, cat_val  = ohe_to_axis_val[feat_name]
            prefix             = AXIS_PREFIX.get(axis_col, axis_col.upper())
            desc               = FULL_LABEL.get(cat_val, cat_val)
            label              = f'{prefix}: {desc}\nN={n},  Val={val:.0f}%'

            ax.text(
                x, y, label,
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color=_internal_text_color(val),
                linespacing=1.55,
                bbox=dict(
                    boxstyle='round,pad=0.6',
                    facecolor=_internal_color(val),
                    edgecolor='#555555', linewidth=1.6,
                ),
                zorder=3,
            )

            # children: left = NO (feature==0), right = YES (feature==1)
            lc = t.children_left[node]   # NO
            rc = t.children_right[node]  # YES
            lx, ly = node_x[lc], float(-(d + 1))
            rx, ry = node_x[rc], float(-(d + 1))

            for cx, cy in [(lx, ly), (rx, ry)]:
                ax.annotate(
                    '', xy=(cx, cy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='#444444',
                                    lw=1.5, mutation_scale=14),
                    zorder=1,
                )

            def edge_label(tx, ty, text, ha):
                ax.text(tx, ty, text, ha=ha, va='center',
                        fontsize=9, fontstyle='italic', color='#333333',
                        bbox=dict(facecolor='white', edgecolor='none',
                                  alpha=0.85, pad=1.5),
                        zorder=4)

            edge_label(x + (lx - x) * 0.35 - 0.07,
                       y + (ly - y) * 0.35, 'No',  'right')
            edge_label(x + (rx - x) * 0.35 + 0.07,
                       y + (ry - y) * 0.35, 'Yes', 'left')

            draw(lc, d + 1)
            draw(rc, d + 1)

    draw(0, 0)

    # ── title + horizontal colour bars ────────────────────────────────────────
    # Leave headroom at the top for title + two bars; tree fills the rest.
    TOP = 0.88
    plt.subplots_adjust(top=TOP, bottom=0.02, left=0.02, right=0.98)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.985, va='top')

    v256 = np.linspace(0, 1, 256)

    # Internal node amber bar
    cb1 = fig.add_axes([0.10, TOP + 0.025, 0.35, 0.022])
    img1 = np.array([[plt.colormaps['YlOrBr'](0.15 + v * 0.72) for v in v256]])
    cb1.imshow(img1, aspect='auto')
    cb1.set_yticks([])
    cb1.set_xticks([0, 127, 255])
    cb1.set_xticklabels(['0%', '50%', '100%'], fontsize=8)
    cb1.set_title('Internal node mean score', fontsize=8.5, pad=3)

    # Leaf grayscale bar
    cb2 = fig.add_axes([0.55, TOP + 0.025, 0.35, 0.022])
    img2 = np.array([[(0.08 + v * 0.87,) * 3 + (1.0,) for v in v256]])
    cb2.imshow(img2, aspect='auto')
    cb2.set_yticks([])
    cb2.set_xticks([0, 127, 255])
    cb2.set_xticklabels(['0%', '50%', '100%'], fontsize=8)
    cb2.set_title('Leaf predicted score', fontsize=8.5, pad=3)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'\n  Saved → {out_path}')
    if open_fig:
        subprocess.run(['open', str(out_path)], check=False)
    plt.close(fig)


# ── terminal text decoder ─────────────────────────────────────────────────────

def decode_tree_text(tree_model, ohe_names: list, ohe_to_axis_val: dict) -> str:
    t = tree_model.tree_
    lines = []

    def recurse(node: int, depth: int):
        pad = '    ' * depth
        if t.feature[node] == -2:
            val = t.value[node][0][0]
            n   = int(t.n_node_samples[node])
            lines.append(f'{pad}→ score: {val:.1f}  (n={n})')
        else:
            feat_name         = ohe_names[t.feature[node]]
            axis_col, cat_val = ohe_to_axis_val[feat_name]
            disp  = FEATURE_DISPLAY.get(axis_col, axis_col)
            short = LABEL_MAP.get(cat_val, cat_val)
            n     = int(t.n_node_samples[node])
            lines.append(f'{pad}[{disp} == {short}?]  n={n}')
            lines.append(f'{pad}├── No')
            recurse(t.children_left[node],  depth + 1)
            lines.append(f'{pad}└── Yes')
            recurse(t.children_right[node], depth + 1)

    recurse(0, 0)
    return '\n'.join(lines)


# ── data helpers ──────────────────────────────────────────────────────────────

def load_dataset(name: str) -> pd.DataFrame:
    path = DATA_DIR / f'construction_performance_table_{name}.csv'
    df = pd.read_csv(path)
    df['dataset'] = name
    return df


def aggregate_variants(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    valid = df.dropna(subset=[SCORE_COL])

    def agg(split: str) -> pd.DataFrame:
        return (
            valid[valid['run_split'] == split]
            .groupby(group_cols)
            .agg(
                mean_score=(SCORE_COL, 'mean'),
                std_score=(SCORE_COL, 'std'),
                n_samples=(SCORE_COL, 'count'),
            )
            .reset_index()
        )

    train_v = agg('train')
    test_v  = agg('test')
    return train_v.merge(
        test_v[group_cols + ['mean_score', 'n_samples']],
        on=group_cols, suffixes=('_train', '_test'),
    )


# ── main routine per scenario ─────────────────────────────────────────────────

def run_tree(scenario: str, df: pd.DataFrame, max_depth: int, open_fig: bool) -> dict:
    is_combined = (scenario == 'combined')
    # For combined: aggregate by (dataset, M, N, E, T) so samples aren't
    # conflated across datasets, but train the tree on M/N/E/T only —
    # dataset is not a feature because the goal is general construction insight.
    group_cols   = (['dataset'] + GROUP_BASE) if is_combined else GROUP_BASE[:]
    feature_cols = GROUP_BASE[:]

    print(f'\n{"="*68}')
    print(f'  {scenario.upper()}')
    print(f'{"="*68}')

    variant_df = aggregate_variants(df, group_cols)
    n_var = len(variant_df)
    print(f'  {n_var} variants  |  '
          f'train score {variant_df["mean_score_train"].min():.1f}–'
          f'{variant_df["mean_score_train"].max():.1f}')

    X, ohe_names, ohe_to_axis_val = build_ohe(variant_df, feature_cols)
    y_train = variant_df['mean_score_train'].values
    y_test  = variant_df['mean_score_test'].values

    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=3, random_state=42)
    tree.fit(X, y_train)

    r2_train = tree.score(X, y_train)
    r2_test  = tree.score(X, y_test)
    print(f'  R²  train={r2_train:.3f}   test={r2_test:.3f}')

    # feature importances (aggregate back to original axes)
    imp_series = pd.Series(tree.feature_importances_, index=ohe_names)
    axis_imp   = {}
    for ohe_name, imp in imp_series.items():
        axis_col, _ = ohe_to_axis_val[ohe_name]
        axis_imp[axis_col] = axis_imp.get(axis_col, 0.0) + imp
    axis_imp = dict(sorted(axis_imp.items(), key=lambda x: x[1], reverse=True))

    print('\n  Feature importances (by axis):')
    for feat, imp in axis_imp.items():
        bar = '█' * int(imp * 32)
        print(f'    {FEATURE_DISPLAY.get(feat, feat):<22} {imp:.4f}  {bar}')

    print('\n  Decision tree (decoded):')
    txt = decode_tree_text(tree, ohe_names, ohe_to_axis_val)
    for line in txt.splitlines():
        print('  ' + line)

    print('\n  Mean score by axis  (train | test):')
    for col in GROUP_BASE:
        if col not in variant_df.columns:
            continue
        bd = (
            variant_df.groupby(col)[['mean_score_train', 'mean_score_test']]
            .mean().round(1)
            .sort_values('mean_score_train', ascending=False)
        )
        print(f'\n    {FEATURE_DISPLAY.get(col, col)}:')
        for idx_val, row in bd.iterrows():
            short = LABEL_MAP.get(idx_val, idx_val)
            print(f'      {idx_val}  {short:<26}  '
                  f'train={row["mean_score_train"]:5.1f}  '
                  f'test={row["mean_score_test"]:5.1f}')

    TREE_DIR.mkdir(parents=True, exist_ok=True)
    out_png = TREE_DIR / f'decision_tree_{scenario}.png'

    plot_tree_categorical(
        tree_model=tree,
        ohe_names=ohe_names,
        ohe_to_axis_val=ohe_to_axis_val,
        title=(
            f'TAG Construction Decision Tree — {scenario}\n'
            f'depth={max_depth}   train R²={r2_train:.3f}   test R²={r2_test:.3f}'
            f'   N={n_var} variants'
        ),
        out_path=out_png,
        open_fig=open_fig,
    )

    return {
        'scenario':   scenario,
        'n_variants': n_var,
        'r2_train':   round(r2_train, 3),
        'r2_test':    round(r2_test,  3),
        'axis_importances': {FEATURE_DISPLAY.get(k, k): round(v, 4)
                             for k, v in axis_imp.items()},
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train and visualize TAG construction decision trees.'
    )
    parser.add_argument(
        '--dataset',
        choices=['history', 'amazon', 'arxiv', 'combined', 'all'],
        default='all',
    )
    parser.add_argument('--depth', type=int, default=8,
                        help='Max tree depth (default: 8)')
    parser.add_argument('--open', dest='open_fig', action='store_true',
                        help='Open each PNG immediately after saving (macOS)')
    args = parser.parse_args()

    all_dfs: dict = {}
    for ds in ['history', 'amazon', 'arxiv']:
        p = DATA_DIR / f'construction_performance_table_{ds}.csv'
        if p.exists():
            all_dfs[ds] = load_dataset(ds)
        else:
            print(f'WARNING: {p} not found — {ds} will be skipped')

    scenarios = (
        ['history', 'amazon', 'arxiv', 'combined']
        if args.dataset == 'all'
        else [args.dataset]
    )

    results = []
    for scenario in scenarios:
        if scenario == 'combined':
            if len(all_dfs) < 2:
                print('Need ≥2 datasets for combined — skipping')
                continue
            df = pd.concat(all_dfs.values(), ignore_index=True)
        else:
            if scenario not in all_dfs:
                print(f'  SKIP {scenario} (CSV not found)')
                continue
            df = all_dfs[scenario]

        results.append(run_tree(scenario, df, max_depth=args.depth,
                                open_fig=args.open_fig))

    if len(results) > 1:
        print(f'\n{"="*68}')
        print('  SUMMARY')
        print(f'{"="*68}')
        summary = pd.DataFrame(results)[['scenario', 'n_variants', 'r2_train', 'r2_test']]
        print(summary.to_string(index=False))

    print(f'\nDone. PNGs saved in {TREE_DIR}/')


if __name__ == '__main__':
    main()
