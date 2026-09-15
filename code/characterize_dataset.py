"""
Automatic dataset characterization — surfaces structural issues (N8/N9
viability, M1 label collapse, text-fidelity redundancy, edge sparsity,
E11b centrality-tie risk) at ingestion time, before any GNN training is
attempted.

Standalone function, no pipeline dependencies beyond yaml/pandas/numpy,
so it can run inside write_data on a brand-new dataset that doesn't have
a TAGConstructor-ready config yet.

Deliberate deviation from spec: `pool_path` accepts a directory of
`sample_*.jsonl` files (the real on-disk layout — see
data/{dataset}/train/samples/) rather than a single pre-pooled CSV, which
does not exist anywhere in this repo. Pooling (dedup on primary_id) is
done here. A single CSV/JSONL path also works if one exists.

Edge-sparsity checks (Check 5) are exact, fast candidate-pair counts for
the ground-truth edge types that are directly computable from the raw
dataframe (E10a categorical, E10c participation, E10d structural). E10b
(scalar) and E11a (semantic) require a similarity threshold / embeddings
that aren't available at this stage and are NOT approximated with a
fabricated number -- they're reported as unevaluated, which is itself
useful signal (this fast preview can't clear them, a real build must).
E11b's degeneracy risk is covered by Check 6 (its own dedicated check,
per the task spec) rather than duplicated here.

Usage:
    from characterize_dataset import characterize_dataset
    report = characterize_dataset('history',
                                   'data/history/train/samples',
                                   'data/configs/history_dataset.yaml')
"""
from __future__ import annotations

import json
import itertools
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

SINGLETON_WARN_THRESHOLD = 0.85
N8_MIN_REAL_HUBS = 50
LABEL_MAJORITY_WARN_THRESHOLD = 0.80
N9_RATIO_WARN_THRESHOLD = 0.90
N9_MIN_GROUPS = 50
TEXT_NULL_WARN_THRESHOLD = 0.10
TEXT_IDENTICAL_WARN_THRESHOLD = 0.95
EDGE_SAMPLE_N = 300
EDGE_SAMPLE_SEED = 42
NEAR_COMPLETE_WARN_THRESHOLD = 0.50
E11B_DEGREE_ZERO_WARN_THRESHOLD = 0.30


# ── pooling ──────────────────────────────────────────────────────────────

def _pool_samples(pool_path: str) -> pd.DataFrame:
    """Load and dedup (on primary_id) every sample_*.jsonl under pool_path,
    or a single CSV/JSONL file if pool_path points directly at one."""
    p = Path(pool_path)
    if p.is_file():
        df = pd.read_csv(p) if p.suffix == '.csv' else pd.read_json(p, lines=True)
    elif p.is_dir():
        files = sorted(p.glob('sample_*.jsonl'))
        if not files:
            # maybe pool_path IS the split dir (train/), not train/samples/
            files = sorted(p.glob('*.jsonl'))
        if not files:
            raise FileNotFoundError(f'No sample_*.jsonl or *.jsonl files found under {p}')
        frames = [pd.read_json(f, lines=True) for f in files]
        df = pd.concat(frames, ignore_index=True)
    else:
        raise FileNotFoundError(f'pool_path does not exist: {p}')

    before = len(df)
    if 'primary_id' in df.columns:
        df = df.drop_duplicates(subset='primary_id').reset_index(drop=True)
    after = len(df)
    df.attrs['n_files_pooled'] = before
    df.attrs['n_unique_after_dedup'] = after
    return df


def _load_one_sample(pool_path: str) -> Optional[pd.DataFrame]:
    """Load a single representative sample (not pooled) for scale-sensitive
    checks. Returns None if pool_path isn't a directory of sample_*.jsonl
    files (e.g. it's already a single pooled CSV/JSONL) -- callers fall
    back to the pooled frame in that case."""
    p = Path(pool_path)
    if not p.is_dir():
        return None
    files = sorted(p.glob('sample_*.jsonl'))
    if not files:
        return None
    return pd.read_json(files[0], lines=True)


# ── individual checks ────────────────────────────────────────────────────

def _check1_secondary_id(df: pd.DataFrame, config: dict, warnings: list, exclusions: list) -> dict:
    result = {'check': 1, 'name': 'Secondary ID density (N8 viability)'}
    if 'secondary_id' not in df.columns or df['secondary_id'].isna().all():
        result.update(singleton_fraction=None, real_hubs=0, strong_hubs=0,
                       n8_viable=False, status='NO_SECONDARY_ID')
        warnings.append('NO_SECONDARY_ID: secondary_id is null for all rows')
        exclusions.append('SKIP_N8')
        return result

    is_list = config.get('secondary_id_is_list', False)
    counts: Counter = Counter()
    for sec_id in df['secondary_id']:
        if sec_id is None or (not is_list and pd.isna(sec_id)):
            continue
        ids = sec_id if isinstance(sec_id, list) else [sec_id]
        counts.update(ids)

    if not counts:
        result.update(singleton_fraction=None, real_hubs=0, strong_hubs=0,
                       n8_viable=False, status='NO_SECONDARY_ID')
        warnings.append('NO_SECONDARY_ID: secondary_id present but all-null after parsing')
        exclusions.append('SKIP_N8')
        return result

    n_unique = len(counts)
    real_hubs = sum(1 for c in counts.values() if c >= 2)
    strong_hubs = sum(1 for c in counts.values() if c >= 3)
    singleton_frac = sum(1 for c in counts.values() if c == 1) / n_unique

    status = 'OK'
    if singleton_frac > SINGLETON_WARN_THRESHOLD:
        warnings.append(f'WARN Check1: secondary_id singleton fraction {singleton_frac:.1%} '
                         f'> {SINGLETON_WARN_THRESHOLD:.0%} (N8 nodes mostly identical to N7)')
        status = 'WARN'
    if real_hubs < N8_MIN_REAL_HUBS:
        warnings.append(f'SKIP_N8 Check1: only {real_hubs} secondary_ids appear in >=2 rows '
                         f'(< {N8_MIN_REAL_HUBS} floor)')
        exclusions.append('SKIP_N8')
        status = 'SKIP_N8'

    result.update(n_unique_secondary_id=n_unique, real_hubs=real_hubs, strong_hubs=strong_hubs,
                   singleton_fraction=round(singleton_frac, 4),
                   n8_viable=(status not in ('SKIP_N8', 'NO_SECONDARY_ID')), status=status)
    return result


def _check2_categorical_label(df: pd.DataFrame, config: dict, warnings: list, exclusions: list) -> dict:
    result = {'check': 2, 'name': 'Categorical label check (M1 viability)'}
    if 'categorical_label' not in df.columns or df['categorical_label'].isna().all():
        result.update(unique=0, majority_fraction=None, m1_viable=False, status='NO_LABEL')
        warnings.append('WARN Check2: categorical_label missing/all-null')
        exclusions.append('SKIP_M1')
        return result

    vc = df['categorical_label'].dropna().value_counts(normalize=True)
    n_unique = int(len(vc))
    majority_frac = float(vc.iloc[0]) if n_unique else float('nan')

    status = 'OK'
    if n_unique <= 2:
        warnings.append(f'WARN Check2: categorical_label has only {n_unique} unique value(s) '
                         f'(binary collapse)')
        status = 'WARN'
    if majority_frac > LABEL_MAJORITY_WARN_THRESHOLD:
        warnings.append(f'WARN Check2: majority class is {majority_frac:.1%} of rows '
                         f'(> {LABEL_MAJORITY_WARN_THRESHOLD:.0%})')
        status = 'WARN'
    if config.get('m1_meta_url'):
        warnings.append('NOTE Check2: m1_meta_url set -- raw categorical_label is NOT the '
                         'true M1 label space; label_factory.py rejoins against the meta '
                         'category file. This check ran on the pre-join label as a proxy.')

    result.update(unique=n_unique, majority_fraction=round(majority_frac, 4),
                   m1_viable=True, status=status)
    return result


def _check3_aggregate_id(df: pd.DataFrame, warnings: list, exclusions: list) -> dict:
    result = {'check': 3, 'name': 'Aggregate ID density (N9 viability)'}
    if 'aggregate_id' not in df.columns or df['aggregate_id'].isna().all():
        result.update(ratio=None, n9_viable=False, status='NO_AGGREGATE_ID')
        warnings.append('WARN Check3: aggregate_id missing/all-null')
        exclusions.append('SKIP_N9')
        return result

    n_unique_agg = df['aggregate_id'].nunique(dropna=True)
    n_unique_primary = df['primary_id'].nunique(dropna=True) if 'primary_id' in df.columns else len(df)
    ratio = n_unique_agg / n_unique_primary if n_unique_primary else float('nan')
    group_sizes = df.groupby('aggregate_id').size()
    groups_ge3 = int((group_sizes >= 3).sum())

    status = 'OK'
    if ratio > N9_RATIO_WARN_THRESHOLD:
        warnings.append(f'WARN Check3: aggregate_id/primary_id ratio {ratio:.3f} '
                         f'> {N9_RATIO_WARN_THRESHOLD} (near-degenerate N9)')
        status = 'WARN'

    # The absolute "50 well-populated groups" floor only makes sense when
    # there are enough distinct aggregate_id values for 50 to be a
    # meaningful bar. A dataset with few total candidate groups (e.g.
    # History's 12 fixed categories) should be judged on what FRACTION of
    # its groups are well-populated instead -- verified: the flat floor
    # incorrectly flagged History's N9 (11/12 groups populated, a known-
    # good axis already used throughout this project) as SKIP_N9.
    if n_unique_agg >= N9_MIN_GROUPS:
        if groups_ge3 < N9_MIN_GROUPS:
            warnings.append(f'SKIP_N9 Check3: only {groups_ge3} aggregate_ids group >=3 primary_ids '
                             f'(< {N9_MIN_GROUPS} floor, out of {n_unique_agg} candidates)')
            exclusions.append('SKIP_N9')
            status = 'SKIP_N9'
    else:
        frac_populated = groups_ge3 / n_unique_agg if n_unique_agg else 0.0
        if frac_populated < 0.5:
            warnings.append(f'SKIP_N9 Check3: only {groups_ge3}/{n_unique_agg} aggregate_ids '
                             f'({frac_populated:.0%}) group >=3 primary_ids, and total candidate '
                             f'groups ({n_unique_agg}) is below the {N9_MIN_GROUPS} floor where '
                             f'the absolute-count rule applies')
            exclusions.append('SKIP_N9')
            status = 'SKIP_N9'

    result.update(n_unique_aggregate_id=int(n_unique_agg), ratio=round(ratio, 4),
                   groups_ge3=groups_ge3, n9_viable=(status != 'SKIP_N9'), status=status)
    return result


def _check4_text_fidelity(df: pd.DataFrame, warnings: list) -> dict:
    result = {'check': 4, 'name': 'Text fidelity check'}
    out = {}
    for col in ['text_fidelity_a', 'text_fidelity_b']:
        if col not in df.columns:
            out[f'{col}_null_rate'] = 1.0
            out[f'{col}_mean_len'] = None
            out[f'{col}_std_len'] = None
            warnings.append(f'WARN Check4: {col} column missing')
            continue
        s = df[col]
        null_rate = s.isna().mean() + (s == '').mean()
        lens = s.dropna().astype(str).str.len()
        out[f'{col}_null_rate'] = round(float(null_rate), 4)
        out[f'{col}_mean_len'] = round(float(lens.mean()), 1) if len(lens) else None
        out[f'{col}_std_len'] = round(float(lens.std()), 1) if len(lens) else None
        if null_rate > TEXT_NULL_WARN_THRESHOLD:
            warnings.append(f'WARN Check4: {col} null rate {null_rate:.1%} '
                             f'> {TEXT_NULL_WARN_THRESHOLD:.0%}')

    identical_frac = None
    if 'text_fidelity_a' in df.columns and 'text_fidelity_b' in df.columns:
        both = df[['text_fidelity_a', 'text_fidelity_b']].dropna()
        if len(both):
            identical_frac = float((both['text_fidelity_a'] == both['text_fidelity_b']).mean())
            out['b_equals_a_fraction'] = round(identical_frac, 4)
            if identical_frac > TEXT_IDENTICAL_WARN_THRESHOLD:
                warnings.append(f'WARN Check4: text_fidelity_b == text_fidelity_a for '
                                 f'{identical_frac:.1%} of rows (indistinguishable T levels)')

    result.update(out, text_fidelity_b_distinct=(identical_frac is None or identical_frac <= TEXT_IDENTICAL_WARN_THRESHOLD))
    return result


def _check5_edge_sparsity(df: pd.DataFrame, config: dict, warnings: list) -> dict:
    result = {'check': 5, 'name': 'Edge sparsity preview'}
    sample = df.sample(min(EDGE_SAMPLE_N, len(df)), random_state=EDGE_SAMPLE_SEED).reset_index(drop=True)
    n = len(sample)
    max_pairs = n * (n - 1) // 2
    edge_reports = {}

    # E10a: categorical edges -- pairs sharing categorical_label
    if 'categorical_label' in sample.columns:
        sizes = sample['categorical_label'].dropna().value_counts()
        e10a_edges = int((sizes * (sizes - 1) // 2).sum())
        edge_reports['E10a'] = _density_row(e10a_edges, max_pairs, 'categorical_label match')
    else:
        edge_reports['E10a'] = {'status': 'UNEVALUATED', 'reason': 'categorical_label missing'}

    # E10c: participation edges -- pairs sharing secondary_id
    if config.get('has_secondary_id') and 'secondary_id' in sample.columns:
        is_list = config.get('secondary_id_is_list', False)
        counts: Counter = Counter()
        for sec_id in sample['secondary_id']:
            if sec_id is None:
                continue
            ids = sec_id if isinstance(sec_id, list) else [sec_id]
            counts.update(ids)
        e10c_edges = int(sum(c * (c - 1) // 2 for c in counts.values()))
        edge_reports['E10c'] = _density_row(e10c_edges, max_pairs, 'shared secondary_id')
    else:
        edge_reports['E10c'] = {'status': 'UNEVALUATED', 'reason': 'no secondary_id for this dataset'}

    # E10d: structural edges -- direct list column (History-style neighbour lists)
    if config.get('has_structural_edges') and 'structural_edges' in sample.columns:
        ids_in_sample = set(sample['primary_id']) if 'primary_id' in sample.columns else set()
        e10d_edges = 0
        seen_pairs = set()
        for pid, neighbours in zip(sample.get('primary_id', range(n)), sample['structural_edges']):
            if not isinstance(neighbours, list):
                continue
            for nb in neighbours:
                if nb in ids_in_sample and nb != pid:
                    pair = tuple(sorted((pid, nb)))
                    seen_pairs.add(pair)
        e10d_edges = len(seen_pairs)
        edge_reports['E10d'] = _density_row(e10d_edges, max_pairs, 'structural_edges list, both endpoints in sample')
    else:
        edge_reports['E10d'] = {'status': 'UNEVALUATED', 'reason': 'no structural_edges for this dataset'}

    # E10b (scalar) and E11a (semantic) need a similarity threshold / embeddings
    # not available at this fast-preview stage -- reported honestly as
    # unevaluated rather than approximated with a fabricated number.
    edge_reports['E10b'] = {'status': 'UNEVALUATED', 'reason': 'requires scalar-similarity threshold, not computed in fast preview'}
    edge_reports['E11a'] = {'status': 'UNEVALUATED', 'reason': 'requires embeddings, not computed in fast preview'}
    edge_reports['E11b'] = {'status': 'SEE_CHECK6', 'reason': 'centrality-tie degeneracy risk covered by Check 6'}

    for etype, rep in edge_reports.items():
        if rep.get('status') == 'WARN_ZERO':
            warnings.append(f'WARN Check5: {etype} produces 0 edges on the {n}-row sample')
        elif rep.get('status') == 'WARN_NEAR_COMPLETE':
            warnings.append(f'WARN Check5: {etype} density {rep["density"]:.1%} '
                             f'> {NEAR_COMPLETE_WARN_THRESHOLD:.0%} of all possible pairs (near-complete graph)')

    result['sample_size'] = n
    result['edges'] = edge_reports
    return result


def _density_row(edge_count: int, max_pairs: int, basis: str) -> dict:
    density = edge_count / max_pairs if max_pairs else 0.0
    status = 'OK'
    if edge_count == 0:
        status = 'WARN_ZERO'
    elif density > NEAR_COMPLETE_WARN_THRESHOLD:
        status = 'WARN_NEAR_COMPLETE'
    return {'edges': edge_count, 'density': round(density, 4), 'basis': basis, 'status': status}


def _check6_e11b_tie_risk(df: pd.DataFrame, config: dict, warnings: list) -> dict:
    result = {'check': 6, 'name': 'E11b centrality tie risk'}
    sample = df.sample(min(EDGE_SAMPLE_N, len(df)), random_state=EDGE_SAMPLE_SEED).reset_index(drop=True)
    n = len(sample)

    if not (config.get('has_secondary_id') and 'secondary_id' in sample.columns):
        result.update(status='UNEVALUATED', reason='no secondary_id (E10b co-participation basis) for this dataset',
                       degree_zero_fraction=None)
        return result

    is_list = config.get('secondary_id_is_list', False)
    # co-participation degree: for each row, its degree = (size of its
    # secondary_id group within the sample) - 1, summed across all its
    # secondary_ids if list-valued.
    groups: dict = {}
    for i, sec_id in enumerate(sample['secondary_id']):
        if sec_id is None:
            continue
        ids = sec_id if isinstance(sec_id, list) else [sec_id]
        for sid in ids:
            groups.setdefault(sid, []).append(i)

    degree = np.zeros(n, dtype=int)
    for members in groups.values():
        d = len(members) - 1
        if d > 0:
            for i in members:
                degree[i] += d

    zero_frac = float((degree == 0).mean())
    status = 'OK'
    if zero_frac > E11B_DEGREE_ZERO_WARN_THRESHOLD:
        warnings.append(f'WARN Check6: {zero_frac:.1%} of nodes have E10b-co-participation '
                         f'degree 0 (> {E11B_DEGREE_ZERO_WARN_THRESHOLD:.0%}) -- E11b k-NN-on-'
                         f'centrality risks degenerating under these tie conditions')
        status = 'WARN'

    result.update(sample_size=n, degree_zero_fraction=round(zero_frac, 4), status=status)
    return result


# ── main entry point ─────────────────────────────────────────────────────

def characterize_dataset(dataset_name: str, pool_path: str, config_path: str,
                          write_yaml: bool = True, verbose: bool = True) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    df = _pool_samples(pool_path)
    warnings: list = []
    exclusions: list = []

    # Check 1 (N8 hub density) is scale-sensitive and must NOT run on the
    # pooled frame: pooling 50 samples gives the same secondary_id far more
    # chances to reappear than it gets within any single training sample,
    # which understates singleton degeneracy badly (verified: pooled-scale
    # gave 62-71% singleton for Amazon/Electronics/Toys/ArXiv, all "OK";
    # the actual per-sample scale GNN training runs at gives 87-98%,
    # matching the original pre-publication audit almost exactly -- and
    # flips ArXiv from "passes" to the single worst dataset). Every other
    # check here is reasonably scale-insensitive (label balance, N9
    # density, text fidelity) and stays on the pooled frame for a more
    # stable estimate.
    single_sample_df = _load_one_sample(pool_path)
    c1 = _check1_secondary_id(single_sample_df if single_sample_df is not None else df,
                               config, warnings, exclusions)
    c2 = _check2_categorical_label(df, config, warnings, exclusions)
    c3 = _check3_aggregate_id(df, warnings, exclusions)
    c4 = _check4_text_fidelity(df, warnings)
    c5 = _check5_edge_sparsity(df, config, warnings)
    c6 = _check6_e11b_tie_risk(df, config, warnings)

    checks = [c1, c2, c3, c4, c5, c6]
    n_passed = sum(1 for c in checks if c.get('status') in ('OK',))

    notes = []
    if df.attrs.get('n_files_pooled') and df.attrs.get('n_unique_after_dedup'):
        notes.append(f"pooled {df.attrs['n_files_pooled']} rows across samples, "
                      f"{df.attrs['n_unique_after_dedup']} unique primary_id after dedup")

    report = {
        'dataset': dataset_name,
        'n_rows_pooled': len(df),
        'checks': checks,
        'checks_passed': f'{n_passed}/6',
        'warnings': warnings,
        'recommended_exclusions': sorted(set(exclusions)),
        'notes': notes,
    }

    if verbose:
        _print_report(report)

    if write_yaml:
        _write_characterization(config_path, c1, c2, c3, c4, c6, warnings, exclusions)

    return report


def _print_report(report: dict):
    print(f"\n{'='*70}\nDATASET: {report['dataset']}  (n={report['n_rows_pooled']} pooled rows)\n{'='*70}")
    for c in report['checks']:
        print(f"\n[CHECK {c['check']}] {c['name']}  -> {c.get('status', '?')}")
        for k, v in c.items():
            if k in ('check', 'name', 'status'):
                continue
            print(f"    {k}: {v}")
    print(f"\n{'-'*70}")
    print(f"CHECKS PASSED: {report['checks_passed']}")
    print(f"WARNINGS ({len(report['warnings'])}):")
    for w in report['warnings']:
        print(f"  - {w}")
    print(f"RECOMMENDED EXCLUSIONS: {report['recommended_exclusions'] or 'none'}")
    print(f"NOTES: {report['notes'] or 'none'}")
    print(f"{'='*70}\n")


def _write_characterization(config_path: str, c1, c2, c3, c4, c6, warnings, exclusions):
    """Append/replace ONLY the dataset_characterization block via text
    surgery, never a full yaml.safe_load + safe_dump round-trip.

    PyYAML's safe_dump does not preserve comments -- a full round-trip
    silently destroys every inline comment and the multi-line audit
    documentation already in these config files (verified: a first version
    of this function did exactly that, caught via `git diff` before it was
    left in place, and reverted). Every other key's bytes are left
    untouched here; only the dataset_characterization section is
    added or replaced as a plain text block.
    """
    block_dict = {
        'secondary_id_singleton_fraction': c1.get('singleton_fraction'),
        'secondary_id_real_hubs': c1.get('real_hubs', 0),
        'n8_viable': c1.get('n8_viable', False),
        'categorical_label_unique': c2.get('unique'),
        'categorical_label_majority_fraction': c2.get('majority_fraction'),
        'm1_viable': c2.get('m1_viable', False),
        'aggregate_id_ratio': c3.get('ratio'),
        'n9_viable': c3.get('n9_viable', False),
        'text_fidelity_b_distinct': c4.get('text_fidelity_b_distinct'),
        'e11b_degree_zero_fraction': c6.get('degree_zero_fraction'),
        'warnings': warnings,
        'recommended_exclusions': sorted(set(exclusions)),
        'characterized_on': str(date.today()),
    }
    block_yaml = yaml.safe_dump({'dataset_characterization': block_dict},
                                 sort_keys=False, default_flow_style=False)

    with open(config_path) as f:
        text = f.read()

    marker = 'dataset_characterization:'
    if marker in text:
        # replace the existing block: from the marker line to either the
        # next top-level (non-indented, non-comment) key or EOF
        lines = text.splitlines(keepends=True)
        start = next(i for i, l in enumerate(lines) if l.startswith(marker))
        end = len(lines)
        for i in range(start + 1, len(lines)):
            l = lines[i]
            if l.strip() and not l.startswith((' ', '\t', '#', '-')):
                end = i
                break
        new_lines = lines[:start] + [block_yaml] + lines[end:]
        text = ''.join(new_lines)
    else:
        if not text.endswith('\n'):
            text += '\n'
        text += block_yaml

    with open(config_path, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('pool_path')
    ap.add_argument('config_path')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    characterize_dataset(args.dataset, args.pool_path, args.config_path,
                          write_yaml=not args.no_write)
