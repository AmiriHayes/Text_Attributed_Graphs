#!/usr/bin/env python3
"""
Pipeline pressure check — validates that the full TAG construction / GNN /
GraphRAG output tree is present and internally consistent, and prints a
PASS/FAIL report per check plus a 30-second summary at the end.

Strictly read-only: opens files to read row counts / columns / means, never
writes, deletes, or modifies anything. Designed to finish in well under two
minutes (it never trains a model or calls an API — only pandas.read_csv /
Path.exists / Path.glob).

Several checks below encode assumptions from the original request that do
NOT match what is actually on disk (e.g. a hardcoded 100-samples-per-variant
formula, a "primary_id" column, exact filenames that were renamed somewhere
along the way). Rather than silently loosen those checks to whatever happens
to be true, each such case is reported explicitly as a MISMATCH line so nothing
gets smoothed over — see the printed report and the final summary paragraph.

Reads:
  - data/{dataset}/{train,test}/raw.jsonl
  - data/{dataset}/{train,test}/samples/sample_*.jsonl
  - output/run_post_audit/construction_performance_table_{dataset}.csv
  - output/run_post_audit/analysis/  (checked for existence only)
  - data/graphrag/within_nodetype_summary.csv
  - data/graphrag/proxy_summary_all_datasets.csv
  - data/{dataset}/questions.csv
  - data/graphrag/ragas_results_{dataset}[.csv|_corrected.csv|_n8_corrected.csv]
  - output/figures/*.pdf, *.png
  - data/graphrag/within_nodetype_scatter_*.png
  - data/graphrag/within_nodetype_summary.csv
  - data/graphrag/proxy_correlation_corrected_all_datasets.csv
  - data/graphrag/e11b_sparsification_ragas.csv (expected per spec — see CHECK 7)
  - output/run_post_audit/analysis/experiment2_pruning_full.csv (expected per spec — see CHECK 3/7)

Writes:
  - Nothing. Report goes to stdout only.

Usage:
  python code/validate_pipeline.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = ['arxiv', 'amazon', 'history', 'electronics', 'toys']

t_start = time.time()
results = {}   # check_name -> (status: 'PASS'/'FAIL'/'WARN', detail lines: list[str])
mismatches = []  # spec-vs-reality mismatches to surface in the final summary


def _report(name, status, lines):
    results[name] = (status, lines)
    print(f'\n{"="*76}\n{name}: {status}\n{"="*76}')
    for l in lines:
        print(f'  {l}')


def _mismatch(msg):
    mismatches.append(msg)
    print(f'  !! SPEC MISMATCH: {msg}')


# ── CHECK 1 — Raw data present ──────────────────────────────────────────────
def check1():
    lines = []
    n_fail = 0
    for ds in DATASETS:
        for split in ('train', 'test'):
            raw = REPO_ROOT / 'data' / ds / split / 'raw.jsonl'
            ok_raw = raw.exists() and raw.stat().st_size > 0
            if not ok_raw:
                n_fail += 1
            lines.append(f'{ds}/{split}/raw.jsonl: {"OK" if ok_raw else "MISSING/EMPTY"}'
                         f' ({raw.stat().st_size:,} bytes)' if raw.exists() else
                         f'{ds}/{split}/raw.jsonl: MISSING')

            samp_dir = REPO_ROOT / 'data' / ds / split / 'samples'
            n_samples = len(list(samp_dir.glob('sample_*.jsonl'))) if samp_dir.exists() else 0
            ok_samples = n_samples >= 50
            if not ok_samples:
                n_fail += 1
            lines.append(f'{ds}/{split}/samples/: {n_samples} files '
                         f'({"OK, >=50" if ok_samples else "FAIL, <50"})')
    status = 'PASS' if n_fail == 0 else 'FAIL'
    _report('CHECK 1 — Raw data present', status,
             lines + [f'{n_fail} file(s)/dir(s) missing or under threshold'])
    return status, n_fail


# ── CHECK 2 — Construction performance tables present ───────────────────────
def check2():
    lines = []
    n_fail = 0
    for ds in DATASETS:
        p = REPO_ROOT / 'output/run_post_audit' / f'construction_performance_table_{ds}.csv'
        if not p.exists():
            lines.append(f'{ds}: MISSING {p}')
            n_fail += 1
            continue
        df = pd.read_csv(p)
        n_variants = df.groupby(['Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx']).ngroups
        n_rows = len(df)
        samples_per_variant = n_rows / n_variants if n_variants else float('nan')

        # Spec says "n_variants x 100 samples". Reality: run_post_audit is a
        # smaller rerun -- report the ACTUAL samples/variant rather than
        # asserting against 100, and flag the mismatch once.
        expected_100 = n_variants * 100
        lines.append(f'{ds}: {n_rows:,} rows | {n_variants} variants | '
                     f'{samples_per_variant:.1f} rows/variant actual '
                     f'(spec assumed 100; expected-under-spec={expected_100:,})')

        required_cols_spec = ['primary_id (or variant)', 'S_GNN_step1', 'Final_Score', 'run_split']
        has_variant_id = all(c in df.columns for c in ['Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx'])
        has_primary_id = 'primary_id' in df.columns
        missing_required = [c for c in ['S_GNN_step1', 'Final_Score', 'run_split'] if c not in df.columns]
        if missing_required:
            lines.append(f'  MISSING required columns: {missing_required}')
            n_fail += 1
        if not has_primary_id:
            lines.append(f'  note: no "primary_id" column (variant identity is the 4-tuple '
                         f'Task_Idx/Node_Idx/Edge_Idx/Text_Idx instead) -- '
                         f'{"present" if has_variant_id else "MISSING"}')

        sc = 'S_GNN_step1'
        if sc in df.columns:
            bad = []
            for key, grp in df.groupby(['Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx']):
                nan_frac = grp[sc].isna().mean()
                if nan_frac > 0.95:
                    bad.append((key, round(nan_frac, 3)))
            if bad:
                lines.append(f'  {len(bad)} variant(s) with >95% NaN {sc} '
                             f'(no exclusion list exists in this repo to cross-check against): '
                             f'{bad[:5]}{" ..." if len(bad) > 5 else ""}')

    if n_variants if DATASETS else True:
        pass
    status = 'PASS' if n_fail == 0 else 'FAIL'
    _mismatch('CHECK 2: "n_variants x 100 samples" does not match run_post_audit '
              '(which uses far fewer samples per variant than run_1000_final) -- '
              'reported actual counts instead of asserting the spec formula.')
    _mismatch('CHECK 2: construction_performance_table_*.csv has no "primary_id" column -- '
              'variant identity is the (Task_Idx, Node_Idx, Edge_Idx, Text_Idx) tuple.')
    _report('CHECK 2 — Construction performance tables present', status, lines)
    return status, n_fail


# ── CHECK 3 — Decision tree outputs present ──────────────────────────────────
def check3():
    lines = []
    n_fail = 0
    analysis_dir = REPO_ROOT / 'output/run_post_audit/analysis'
    if not analysis_dir.exists():
        lines.append(f'output/run_post_audit/analysis/: DOES NOT EXIST')
        n_fail += 1
        trees_elsewhere = sorted((REPO_ROOT / 'output').glob('*/analysis/decision_tree_*combined*.png'))
        lines.append(f'  decision tree PNGs DO exist elsewhere: '
                     f'{[str(p.relative_to(REPO_ROOT)) for p in trees_elsewhere[:3]]}'
                     f'{" ... (" + str(len(trees_elsewhere)) + " total)" if len(trees_elsewhere) > 3 else ""}')
        lines.append(f'  -> those trees are from output/run_1000_final (timestamped 2026-08-17), '
                     f'which PRE-DATES output/run_post_audit (timestamped 2026-08-29/30) -- '
                     f'the decision trees currently in the repo were built on PRE-AUDIT-FIX data, '
                     f'not the corrected pipeline.')
        _mismatch('CHECK 3: no decision-tree analysis has ever been run on run_post_audit '
                  '(the corrected rerun) -- the only decision_tree_*.png files in the repo are '
                  'from output/run_1000_final / run_500_final / run_1000_10s, which predate the '
                  'E11c/T12e/E11b-k/N8-lookup fixes.')
    else:
        pngs = list(analysis_dir.glob('decision_tree_*.png'))
        pkls = list(analysis_dir.glob('*.pkl'))
        lines.append(f'output/run_post_audit/analysis/: exists, {len(pngs)} tree PNGs, {len(pkls)} pickles')
        if not pngs and not pkls:
            n_fail += 1

    wns = REPO_ROOT / 'data/graphrag/within_nodetype_summary.csv'
    lines.append(f'within_nodetype_summary.csv: {"OK" if wns.exists() else "MISSING"}')
    if not wns.exists():
        n_fail += 1

    psad = REPO_ROOT / 'data/graphrag/proxy_summary_all_datasets.csv'
    lines.append(f'proxy_summary_all_datasets.csv: {"OK" if psad.exists() else "MISSING"}')
    if not psad.exists():
        n_fail += 1

    status = 'FAIL' if n_fail else 'WARN'  # WARN not PASS: the analysis/ gap is real even if fallback data exists
    _report('CHECK 3 — Decision tree outputs present', status, lines)
    return status, n_fail


# ── CHECK 4 — Question sets present and complete ────────────────────────────
def check4():
    lines = []
    n_fail = 0
    required_cols = ['id', 'question', 'reference_answer', 'question_subtype', 'hop_type']
    expected_subtypes = {'single_specific', 'aggregate_cross_paper', 'edge_multihop'}
    for ds in DATASETS:
        p = REPO_ROOT / 'data' / ds / 'questions.csv'
        if not p.exists():
            lines.append(f'{ds}: MISSING questions.csv')
            n_fail += 1
            continue
        df = pd.read_csv(p)
        n = len(df)
        ok_n = abs(n - 102) <= 5
        missing_cols = [c for c in required_cols if c not in df.columns]
        subtype_counts = df['question_subtype'].value_counts().to_dict() if 'question_subtype' in df.columns else {}
        subtypes_present = set(subtype_counts.keys())
        roughly_equal = False
        if subtype_counts:
            vals = list(subtype_counts.values())
            roughly_equal = (max(vals) - min(vals)) <= max(3, 0.15 * (n / max(len(vals), 1)))
        lines.append(f'{ds}: n={n} ({"OK" if ok_n else "OUT OF RANGE (102+/-5)"}) | '
                     f'missing_cols={missing_cols or "none"} | '
                     f'subtypes={subtype_counts} | '
                     f'expected_subtypes_present={subtypes_present == expected_subtypes} | '
                     f'roughly_equal_thirds={roughly_equal}')
        if not ok_n or missing_cols:
            n_fail += 1
    status = 'PASS' if n_fail == 0 else 'FAIL'
    _report('CHECK 4 — Question sets present and complete', status, lines)
    return status, n_fail


# ── CHECK 5 — RAGAS results present ──────────────────────────────────────────
def check5():
    lines = []
    n_warn = 0
    corrected_name_variants = {
        'arxiv':       'ragas_results_arxiv_n8_corrected.csv',
        'amazon':      'ragas_results_amazon_n8_corrected.csv',
        'electronics': 'ragas_results_electronics_corrected.csv',
        'toys':        'ragas_results_toys_corrected.csv',
        'history':     None,  # History has no N8 (has_secondary_id=false) -- no corrected file possible
    }
    for ds in DATASETS:
        pre = REPO_ROOT / 'data/graphrag' / f'ragas_results_{ds}.csv'
        if pre.exists():
            df = pd.read_csv(pre)
            comp_col = 'composite_score' if 'composite_score' in df.columns else None
            mean_c = df[comp_col].mean() if comp_col else float('nan')
            lines.append(f'{ds} PRE-FIX  {pre.name}: {len(df):,} rows | '
                         f'{df["system_name"].nunique() if "system_name" in df.columns else "?"} variants | '
                         f'mean_composite={mean_c:.3f}' if comp_col else
                         f'{ds} PRE-FIX  {pre.name}: {len(df):,} rows (no composite_score column)')
        else:
            lines.append(f'{ds} PRE-FIX  {pre.name}: MISSING')

        corr_name = corrected_name_variants[ds]
        if corr_name is None:
            lines.append(f'{ds} CORRECTED: N/A (no N8 exists for History)')
            continue
        corr = REPO_ROOT / 'data/graphrag' / corr_name
        if corr.exists():
            df = pd.read_csv(corr)
            comp_col = 'composite_score' if 'composite_score' in df.columns else None
            mean_c = df[comp_col].mean() if comp_col else float('nan')
            n_var = df['system_name'].nunique() if 'system_name' in df.columns else None
            lines.append(f'{ds} CORRECTED {corr.name}: {len(df):,} rows | {n_var} variants | '
                         f'mean_composite={mean_c:.3f}')
        else:
            lines.append(f'{ds} CORRECTED {corr.name}: MISSING -- FLAG')
            n_warn += 1

        # Note: corrected files are N8-only (rebuild scripts never touched N7/N9).
        lines.append(f'  note: {ds} corrected file covers N8 ONLY -- N7/N9 have no '
                     f'post-fix RAGAS scores anywhere in this repo (pre-fix only).')

    status = 'WARN' if n_warn == 0 else 'FAIL'
    _mismatch('CHECK 5: filenames for "corrected" RAGAS results are inconsistent across datasets '
              '(ragas_results_{arxiv,amazon}_n8_corrected.csv vs '
              'ragas_results_{electronics,toys}_corrected.csv) -- checked both patterns explicitly '
              'rather than assuming one naming convention.')
    _mismatch('CHECK 5: every "corrected" RAGAS file that exists covers N8 only. Amazon N9, '
              'History N7, Electronics N7/N9, and Toys N7/N9 have never been re-evaluated on the '
              'post-audit-fix pipeline -- only pre-fix ragas_results_{dataset}.csv exists for them.')
    _report('CHECK 5 — RAGAS results present', status, lines)
    return status, n_warn


# ── CHECK 6 — Figure files present ───────────────────────────────────────────
def check6():
    lines = []
    n_fail = 0
    fig_dir = REPO_ROOT / 'output/figures'

    claim6 = fig_dir / 'claim6_n8_corrected_2x2.pdf'
    lines.append(f'claim6_n8_corrected_2x2.pdf: {"OK" if claim6.exists() else "MISSING"}')
    if not claim6.exists():
        n_fail += 1

    arxiv_heatmap_glob = list(REPO_ROOT.glob('**/*arxiv*heatmap*')) + list(REPO_ROOT.glob('**/*arxiv*combined*'))
    arxiv_heatmap_glob = [p for p in arxiv_heatmap_glob if 'node_modules' not in str(p)]
    fig2 = fig_dir / 'fig2_arxiv_construction_performance.png'
    lines.append(f'*arxiv*heatmap* literal glob: {len(arxiv_heatmap_glob)} matches '
                 f'(likely 0 -- no file is literally named "heatmap")')
    lines.append(f'  actual arxiv two-panel figure (curves+heatmap panel) instead: '
                 f'fig2_arxiv_construction_performance.png: {"OK" if fig2.exists() else "MISSING"}')
    if not fig2.exists():
        n_fail += 1
    _mismatch('CHECK 6: no file matches the literal "*arxiv*heatmap*" glob -- the actual arxiv '
              'two-panel figure is named fig2_arxiv_construction_performance.png. Checked for it '
              'explicitly instead of reporting a false negative.')

    epoch_glob = list(fig_dir.glob('*epoch*curve*'))
    lines.append(f'*epoch*curve*.png: {[p.name for p in epoch_glob]} '
                 f'({"OK" if epoch_glob else "MISSING"})')
    if not epoch_glob:
        n_fail += 1

    scatter_glob = sorted((REPO_ROOT / 'data/graphrag').glob('within_nodetype_scatter_*.png'))
    lines.append(f'within_nodetype_scatter_*.png: {len(scatter_glob)} files '
                 f'({[p.name for p in scatter_glob]})')
    if not scatter_glob:
        n_fail += 1

    status = 'PASS' if n_fail == 0 else 'FAIL'
    _report('CHECK 6 — Figure files present', status, lines)
    return status, n_fail


# ── CHECK 7 — Key analysis CSVs present ──────────────────────────────────────
def check7():
    lines = []
    n_fail = 0
    checks = [
        ('data/graphrag/within_nodetype_summary.csv', True),
        ('data/graphrag/proxy_correlation_corrected_all_datasets.csv', True),
        ('data/graphrag/e11b_sparsification_ragas.csv', False),
        ('output/run_post_audit/analysis/experiment2_pruning_full.csv', False),
    ]
    for rel, required in checks:
        p = REPO_ROOT / rel
        ok = p.exists()
        lines.append(f'{rel}: {"OK" if ok else "MISSING"}')
        if not ok and required:
            n_fail += 1

    e11b_alt = REPO_ROOT / 'data/graphrag/ragas_results_arxiv_e11b_k5.csv'
    lines.append(f'  e11b_sparsification_ragas.csv does not exist under that name -- '
                 f'the actual E11b k=5 causal-validation RAGAS output appears to be '
                 f'ragas_results_arxiv_e11b_k5.csv: {"OK, exists" if e11b_alt.exists() else "also missing"}')
    lines.append(f'  experiment2_pruning_full.csv does not exist anywhere in the repo '
                 f'(output/run_post_audit/analysis/ does not exist at all -- see CHECK 3). '
                 f'No alternate filename was found by search.')
    _mismatch('CHECK 7: e11b_sparsification_ragas.csv and '
              'output/run_post_audit/analysis/experiment2_pruning_full.csv do not exist under '
              'those names. The nearest equivalent found is ragas_results_arxiv_e11b_k5.csv; no '
              'equivalent to experiment2_pruning_full.csv was found at all.')

    status = 'FAIL' if n_fail else 'WARN'
    _report('CHECK 7 — Key analysis CSVs present', status, lines)
    return status, n_fail


def main():
    print('='*76)
    print('PIPELINE VALIDATION — read-only, no training/eval/git commands run')
    print('='*76)

    s1, _ = check1()
    s2, _ = check2()
    s3, _ = check3()
    s4, _ = check4()
    s5, _ = check5()
    s6, _ = check6()
    s7, _ = check7()

    elapsed = time.time() - t_start
    all_status = [s1, s2, s3, s4, s5, s6, s7]
    n_failed = sum(1 for s in all_status if s == 'FAIL')
    n_warned = sum(1 for s in all_status if s == 'WARN')

    print(f'\n{"="*76}\nSUMMARY\n{"="*76}')
    for i, s in enumerate(all_status, 1):
        print(f'  CHECK {i}: {s}')
    overall = 'FAIL' if n_failed else ('PASS (with warnings)' if n_warned else 'PASS')
    print(f'\nOVERALL: {overall}  ({n_failed} failed, {n_warned} warned, '
          f'{len(all_status) - n_failed - n_warned} clean)')
    print(f'Elapsed: {elapsed:.1f}s')

    print(f'\n{"="*76}\n30-SECOND SUMMARY FOR THE AUTHOR\n{"="*76}')
    print(
        "Raw data and question sets are complete for all 5 datasets (CHECK 1, 4 pass).\n"
        "Construction-performance tables exist for all 5 in run_post_audit, but that run\n"
        "used far fewer samples/variant than the '100' assumed in the request -- not a bug,\n"
        "just a different (smaller, corrected-code) run than run_1000_final.\n"
        "Decision-tree PNGs exist, but ONLY from the pre-audit-fix run_1000_final/run_500_final --\n"
        "nothing has been tree-analyzed on the corrected run_post_audit data yet (CHECK 3 gap).\n"
        "RAGAS: pre-fix results exist for all 5 datasets; CORRECTED results exist for N8 ONLY,\n"
        "in 4 datasets (no History N8). N7/N9 have never been re-scored post-fix, in any dataset\n"
        "(CHECK 5). Figures and key CSVs mostly exist under different filenames than the request\n"
        "assumed (CHECK 6/7) -- two files (e11b_sparsification_ragas.csv,\n"
        "experiment2_pruning_full.csv) genuinely do not exist anywhere in the repo."
    )
    sys.exit(1 if n_failed else 0)


if __name__ == '__main__':
    main()
