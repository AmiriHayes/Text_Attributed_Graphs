# Session Notes — TAG Research Handoff
*Last updated: 2026-06-18*

---

## Project Overview

Building a principled GraphRAG evaluation framework for ICLR. The core idea: systematically vary graph construction choices across an ontological framework Ω = M × C (C = N × E × T) and measure downstream GNN performance to identify which choices matter and why.

**Datasets**: arxiv (papers/authors/categories), amazon (products/users/families), history (books)  
**Total variants**: 180 arxiv, 204 amazon, 114 history  
**Samples per variant**: 40 (20 train + 20 test, each a 1000-row subsample)

---

## Ontology Reference

| Axis | Values | Meaning |
|---|---|---|
| M (task) | M1–M6 | node-categorical, node-scalar, edge-categorical, edge-scalar, global-categorical, global-scalar |
| N (node type) | N7, N8, N9 | primary, secondary, aggregate |
| E (edge type) | E10a–E11c | categorical GT, co-participation weighted, binary participation, semantic sim, structural sim, functional sim |
| T (text fidelity) | T12a, T12b, T12e | contextual (title+content), standard (title only), baseline (zeros) |

**C4 principle**: exclude (M, N, E) combos where the edge construction is definitionally equal to the label.

---

## Run Status (as of 2026-06-18)

| Dataset | Status | Output file |
|---|---|---|
| **history** | COMPLETE (4560 rows, 114 variants) | `output/construction_performance_table_history.csv` |
| **amazon** | COMPLETE (7680 rows, 204 variants) | `output/construction_performance_table_amazon.csv` |
| **arxiv** | IN PROGRESS (~variant 100/180 at session end) | `output/construction_performance_table_arxiv.csv` |

Arxiv restart command (resume-safe):
```bash
caffeinate -i /opt/anaconda3/envs/tag_graphs/bin/python3 code/experiment_runner.py --datasets arxiv --epochs 200 --early_stopping
```

---

## Key Findings So Far

### History (clean dataset, most signal)
- **M1** (book genre, 11 classes): T12a (75%) > T12b (74%) >> T12e (54%). E10b best edge type. 21-point gap proves text fidelity matters.
- **M3** (N7/E10b): 96% with T12a/T12b, drops to 57% with T12e. Text nearly indispensable.
- **M4** (N7/E10b): 87% with T12a/T12b, 50% (mean predictor) with T12e.
- **M2** (book price): BACKWARDS — T12e scores higher (~49%) than T12a/T12b (~36%). Real embeddings actively hurt price prediction (overfitting semantic content that doesn't correlate with price). N9 catastrophic (R2 avg −40 to −65).
- **M5**: Degenerate — label (avg_clustering > 0.3) is constant within every variant (std=0 across 40 samples). E10a/E11b/E11c always True=1, E10b/E10c/E11a always True=0.
- **M6**: All NaN — structural issue (see below).

### Amazon
- **M1**: REDESIGNED — 47-class sport subcategory label (46 named + Other ≤5%). Majority = Camping & Hiking (17.2%). Built from `parent_asin → level-3 breadcrumb` via `meta_Sports_and_Outdoors.jsonl`; cached in `code/derived/`. Training verified end-to-end (10-epoch test: T12a/T12b ~14–23%, T12e ~2–19% — expected direction with low epochs). Full 198-variant run pending (use 200 epochs + early stopping).
- **M3** (N7/E10b): 84% T12a, 84% T12b, 81% T12e — most meaningful Amazon signal.
- **M2**: All variants R2 ≈ −0.01 to −0.04 (barely beats mean predictor).
- **M4** (N8/E10b): 13–14% (very poor).
- **M5**: Degenerate — label always 0 for most edge types (clustering never > 0.3 for sparse Amazon graphs). E11b always 1 (k=50 creates dense clustering).
- **M6**: NaN throughout (structural issue, see below).

---

## Known Bugs / Degenerate Cases

### M5 100% / Degenerate Label
The global categorical label (`avg_clustering > 0.3`) is **constant within a variant** — determined entirely by the edge construction type, not by graph content. The GlobalTrainer trains on 20 identical-label graphs and trivially memorizes the constant. Not a code bug — a task definition issue. M5 results are uninformative for measuring construction quality. Consider redesigning M5 label for the paper.

### M6 NaN Scores
M6 uses per-row R2 (one prediction per graph). R2 is undefined for a single (predicted, true) pair — it's stored as NaN by design in `build_result_row_global`. This propagates to `normalized_score = NaN`, making M6 completely unrankable in the current schema.

**Fix needed**: compute aggregate R2 across all 40 (predicted, true) pairs for each variant post-hoc, rather than per-row. This is an analysis fix, not a re-run. Something like:
```python
variant_df = m6_df.groupby(['Task_Idx','Node_Idx','Edge_Idx','Text_Idx'])
r2_per_variant = variant_df.apply(lambda g: compute_r2(g['Predicted_Value'], g['True_Value']))
```

### Amazon M1 — FIXED (was binary, now 10-class subcategory)
`categorical_label` was binary sentiment (83.6% class 1 = degenerate). **Replaced** with 10-class sport subcategory prediction derived from the Amazon product family's level-3 category breadcrumb (`meta_Sports_and_Outdoors.jsonl`). Labels:
  0=Camping&Hiking, 1=Cycling, 2=Fishing, 3=Strength Training, 4=Water Sports,
  5=Leisure Sports&Game Room, 6=Golf, 7=Shooting, 8=Team Sports, 9=Boating&Sailing
Dataset filtered from 100k → ~55k reviews. Majority class ≈ 31% (Camping & Hiking). First results showing ~83% accuracy on 10-class — genuine signal.

**C4 consequence**: parent_asin → l3_cat is deterministic, so E10a and E11c are now C4 leaks for M1+N7 (excluded). M1 now has 14 MNE combos (was 16). Total Amazon variants: 198 (was 204).

**Files changed**: `code/regenerate_amazon.py` (new), `data/configs/amazon_variants.yaml` (C4 updated, totals updated), `data/amazon/{train,test}/raw.jsonl`, `review_index.csv`, `samples/*`, `embeddings/*.npy`. Old results archived as `output/construction_performance_table_amazon_OLD_binary.csv`.

### Dense Graph Variants (N8/E10a, N8/E11c for arxiv)
E10a and E11c for N8 connect all authors in the same arxiv category → near-complete subgraph per category → 673k+ edges → training intractable on CPU AND labels have near-zero variance (all centrality ≈ 1.0 → R2 = −∞).

**Current fix**: `MAX_EDGES = 200_000` guard in `experiment_runner.py` — fires before training for M1-M4, fires on probe graph for M5/M6. Dense variants get `degenerate=True`, `normalized_score=0.0`, skip training.

### M2 + T12e "Winning"
Zero embeddings (T12e) outperforming real text (T12a/T12b) for scalar tasks (especially history M2). Not a bug — real finding. Model with rich 768-dim embeddings overfits to semantic content that doesn't correlate with book price. Worth discussing in paper as evidence that T choice isn't monotone with performance across all tasks.

### E11b Top-K vs Threshold
Original E11b used `|centrality_i − centrality_j| ≤ 0.1` threshold. On sparse graphs where most nodes have near-zero centrality, this produced near-complete graphs (498k edges for history, 4.2M for arxiv N8). **Current fix**: top-k=50 per node (k was deliberately set to 50 after discussion — defensible as k-NN graph, standard in GNN literature). Research question open: does the top-k cap artificially constrain the search space? Argued that the threshold definition has a design flaw (not scale-invariant), and top-k is actually more principled.

---

## Code Changes Made This Session

| File | Change | Why |
|---|---|---|
| `edge_factory.py` | E11b switched from global threshold to top-k=50 | Threshold produced near-complete graphs for uniform centrality |
| `edge_factory.py` | E11a vectorized upper-triangle extraction | Was O(n²) Python loop |
| `trainer.py` | R2 normalized score: `clamp(R2, 0, 1)×100` → `(clamp(R2, -1, 1) + 1) / 2 × 100` | Clamping at 0 lost all ranking info for negative R2 values; R2=0 now maps to 50, R2=−1 maps to 0 |
| `trainer.py` | `_scalar_metrics`: guard `ss_tot < 1e-6` → return `R2=NaN` | Near-zero label variance was producing R2=−∞ (e.g., −94 billion for dense graphs) |
| `generic_data_manager.py` | `_emb_cache` dict: load each embedding file once, cache by (node_type, fidelity, split) | `np.load()` was called once per variant×sample; reduces disk reads from ~4500 to ~18 |
| `experiment_runner.py` | `MAX_EDGES = 200_000` guard in `_run_one` | Skip training for graphs too dense to train on CPU |
| `experiment_runner.py` | Probe graph density check for M5/M6 path | M5/M6 bypassed the guard; N8/E10a wasted 70 min training before outputting NaN |
| `experiment_runner.py` | `_dense_graph_row()` helper | Builds a properly-schemed degenerate row without running the model |

---

## Open Questions / Future Work

### For the Paper
1. **M5 redesign**: avg_clustering > 0.3 is too deterministic given the edge construction. Consider a composite label that's less directly determined by E choice (e.g., mixing clustering, diameter, density, component count).
2. **M6 post-hoc scoring**: compute per-variant aggregate R2 across all 40 predictions in analysis, don't rely on per-row normalized_score.
3. **Amazon M1**: decide whether to redefine as multi-class category prediction or keep as binary and note the degeneracy.
4. **E11b defense**: settle on whether top-k=50 or threshold-based is the canonical definition. The top-k argument: (a) k-NN graphs are standard in the literature, (b) threshold is not scale-invariant across datasets, (c) comparable density enables fair cross-dataset comparison.
5. **T12e + E11a is definitionally broken**: semantic similarity edges require non-zero embeddings. With T12e (zeros), cosine similarity is 0 everywhere → no edges → empty graph with no features. Consider adding `(*, *, E11a, T12e)` to the remove list in variant configs.
6. **M2 T12e > T12a finding**: potentially interesting for the paper — text richness doesn't always help, especially for scalar tasks where graph structure encodes label-relevant information better than semantic content.

### Infrastructure / Future Runs
- **Parallel training**: each variant's 40 samples are independent. `multiprocessing.Pool` across samples could give 4–8x wall-time speedup on a Mac with 8 cores. Requires lock on CSV append and careful PyTorch multiprocessing (spawn start method on macOS).
- **Variant-level dense cache**: after first sample of a (N, E) pair fires the MAX_EDGES guard, skip graph construction entirely for the remaining 39 samples. Currently each degenerate sample still pays ~20s for graph construction.
- **Hidden dim reduction**: 256 → 128 would give ~2-3x speedup. GNN on 1000-node graphs is likely overparameterized at 256.
- **E10b for history is O(n²)**: `_build_history_neighbour_cooccurrence` iterates all pairs to count shared neighbours. For 1000 nodes this is 500k iterations in Python. Could be vectorized.

---

## Infrastructure Notes

- **Conda env**: `tag_graphs`
- **Python**: `/opt/anaconda3/envs/tag_graphs/bin/python3`
- **Working directory**: `/Users/amirihayes/Documents/GitHub/Text_Attributed_Graphs`
- **Resume logic**: keyed on `(M, N, E, T, split, sample_idx)` — ctrl+C and re-run the same caffeinate command to resume from last completed row
- **Configs**: `data/configs/{dataset}_dataset.yaml` and `data/configs/{dataset}_variants.yaml`
- **Output**: `output/construction_performance_table_{dataset}.csv`, one row per (variant, split, sample_idx)
- **Log**: `output/experiment_runner.log`

---

## Things Deliberately NOT Done (Decisions Made)

- Did not implement weighted/class-balanced loss for M1 Amazon (task redefinition preferred over loss fix)
- Did not implement M5 weighted composite statistic (discussed, decided to run first and reassess)
- Did not reduce epochs from 200 (user decided to keep for this run)
- Did not parallelize sample processing (deferred — bigger refactor)
