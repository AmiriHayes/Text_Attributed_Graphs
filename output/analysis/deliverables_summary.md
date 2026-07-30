# TAG Construction Analysis — Deliverables Summary

**Session date:** 2026-07-09  
**Constraint:** No training runs. No modifications to existing CSVs or raw data.

---

## Deliverable 1 — Heatmap Comparisons

**Script:** `code/generate_analysis_heatmaps.py`  
**Output directory:** `output/analysis/`

Three heatmaps generated per dataset: **current** (clamped `Final_Score`), **original** (pre-scoring-v2 `normalized_score`), and **non-clamped** (on-the-fly `Δ/(1+|Δ|) × 100` for all tasks).

| Dataset | `_original.png` | `_unclamped.png` | `_current.png` |
|---|---|---|---|
| history | ✅ 96 variants × 20 samples | ✅ 96 × 10 | ✅ 96 × 10 |
| amazon | ✅ 192 variants × 20 samples | ✅ 192 × 10 | ✅ 192 × 10 |
| arxiv | ✅ 54 variants × 20 samples | ✅ 54 × 10 | ✅ 54 × 10 |
| electronics | — no pre-scoring-v2 CSV | ✅ 96 × 10 | ✅ 96 × 10 |
| toys | — no pre-scoring-v2 CSV | ✅ 96 × 10 | ✅ 96 × 10 |

### Scoring definitions

**Original** (`_original.png`)  
Source: root-level `output/construction_performance_table_{dataset}.csv` (pre-scoring-v2 run).  
Column: `normalized_score` — raw ML metric (accuracy or R²) × 100.  
ArXiv root CSV has a known tokenisation error; loaded with `on_bad_lines='skip'`.  
Electronics and toys never had an original-scoring run — skipped.

**Current** (`_current.png`)  
Source: `output/run_20260620/` (history/amazon/arxiv) and `output/run_20260707_091312/` (electronics/toys).  
Formula: `max(0, (S_GNN − S_MLP) / (1 − S_MLP)) × 100` — clamped to ≥ 0.

**Non-clamped** (`_unclamped.png`)  
Computed on-the-fly from `S_GNN_step1` / `S_MLP_step1` in the current-run CSVs.  
Formula: `Δ / (1 + |Δ|) × 100` applied uniformly to **all task types** (no M1/M3/M4/M5 vs M2/M6 branching).  
Signed — negative values indicate GNN performed worse than the MLP baseline.  
Diverging `RdBu_r` colourmap, centred at 0.  
**No CSV writes** — computation only.

### Layout conventions (shared across all three panels per dataset)

- Y-axis: variants sorted by mean `Final_Score` descending within each Task group (M1, M3, …)  
- X-axis: sample indices  
- Row order is fixed from the current-run CSV so all three heatmaps are directly comparable

---

## Deliverable 2 — Tree Edit Distance

**Modified file:** `code/run_analysis.py`  
**Output:** `output/analysis/` (updated CSVs + printed summary)

### Change made

`compare_trees()` in `decision_tree_analysis.py` already computes `tree_edit_distance` between any two fitted trees, but Experiment 1 only printed Spearman ρ — the TED value was computed but discarded. Six lines were added to the Experiment 1 print block to surface it:

```
TED (agg tree_train vs tree_test): raw=12  nodes=(15+11=26)  norm=0.4615
```

Normalization: `norm_TED = raw_TED / (n_train_nodes + n_test_nodes)` — values in [0, 1] where 0 = identical structure and 1 = maximally different.

### Experiment 1 results (aggregate tree_train vs tree_test)

| Dataset | ρ | p | raw TED | nodes | norm TED |
|---|---|---|---|---|---|
| History | +0.588 | 4.1e-04 | 12 | 15 + 11 = 26 | **0.462** |
| Amazon | +0.661 | 4.2e-14 | 34 | 51 + 47 = 98 | **0.347** |
| ArXiv | +0.934 | 3.4e-22 | 25 | 27 + 27 = 54 | **0.463** |
| Electronics | +0.590 | 1.3e-05 | 28 | 23 + 25 = 48 | **0.583** |
| Toys | +0.448 | 6.9e-03 | 17 | 19 + 21 = 40 | **0.425** |

Per-sample normalized TED (mean ± std across 100 paired train/test trees) is also reported alongside each row.

### Experiment 3 — 5×5 edit distance table (Table A, normalized TED)

`*` marks cells where ρ is significant at p < 0.05.

|  | History | Amazon | ArXiv | Electronics | Toys |
|---|---|---|---|---|---|
| **History** | — | \*0.650 | 0.596 | 0.586 | 0.430 |
| **Amazon** | 0.650 | — | 0.551 | 0.564 | \*0.641 |
| **ArXiv** | 0.596 | 0.551 | — | 0.627 | 0.615 |
| **Electronics** | \*0.586 | 0.564 | \*0.627 | — | 0.572 |
| **Toys** | \*0.430 | 0.641 | 0.615 | 0.572 | — |

Full CSVs saved:
- `output/analysis/cross_dataset_3x3_with_rho.csv` / `_p.csv` / `_ted.csv`
- `output/analysis/cross_dataset_3x3_clean_rho.csv` / `_p.csv` / `_ted.csv`

---

## Deliverable 3 — Extended Samples

**Script:** `code/extend_samples.py`  
**Outcome:** All samples 00–49 already existed; script confirmed and did not overwrite.

### Verification

```
✅ history/train:      All samples 00-49 present
✅ history/test:       All samples 00-49 present
✅ amazon/train:       All samples 00-49 present
✅ amazon/test:        All samples 00-49 present
✅ arxiv/train:        All samples 00-49 present
✅ arxiv/test:         All samples 00-49 present
✅ electronics/train:  All samples 00-49 present
✅ electronics/test:   All samples 00-49 present
✅ toys/train:         All samples 00-49 present
✅ toys/test:          All samples 00-49 present
```

### Generation convention

- `data/{dataset}/{split}/samples/sample_{i:02d}.jsonl` for `i` in `range(20, 50)`
- Each sample: 1,000 rows drawn from `{split}/raw.jsonl` with `random_state=i`
- Samples 00–19 untouched
- Raw JSONL files untouched

### Readiness for next training run

Samples 20–49 will be picked up automatically by the next `experiment_runner.py` run via the existing resume logic. No code changes needed — the runner already iterates over all files present in the `samples/` directory.

---

## Files changed / created this session

| File | Type | Change |
|---|---|---|
| `code/generate_analysis_heatmaps.py` | New | Heatmap generation script targeting `output/analysis/` |
| `code/run_analysis.py` | Modified | +6 lines: aggregate-tree norm TED added to Exp 1 print block |
| `output/analysis/heatmap_*.png` | New | 21 heatmaps (15 new + 6 regenerated current) |
| `output/analysis/cross_dataset_3x3_*.csv` | Updated | Fresh run with current data |
| `output/analysis/run_analysis_output.txt` | New | Full printed output from this run |

### Unchanged (constraint confirmed)

- All `data/` raw files and existing samples 00–19
- All `output/run_*/` construction performance CSVs
- `output/run_final/` (still empty — Electronics/Toys retrain not yet run)
