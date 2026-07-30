# Definitive M-Task Status Table
**Generated**: 2026-07-08  
**Datasets**: History, ArXiv, Amazon, Electronics, Toys  
**Tasks**: M1–M6

---

## Status Legend

| Symbol | Meaning |
|---|---|
| ✓ | Fully implemented and runnable |
| ✓* | Implemented with modification (see notes) |
| ✗ | Excluded — data not viable for this task |
| ~ | Derived post-hoc (no additional training runs) |

---

## Status Table

| Task | History | ArXiv | Amazon | Electronics | Toys |
|---|---|---|---|---|---|
| **M1** node categorical | ✓ (12 book subject categories; standard CS-TAG node classification) | ✓ (CS subdomain, e.g. cs.AI/cs.CV; standard CS-TAG benchmark) | ✓ (L3 product subcategory via meta streaming join) | ✓ (same mechanism as Amazon) | ✓ (same mechanism as Amazon) |
| **M2** node scalar | ✗ (local z-score train signal not generalizable: train mean FS=0.05 collapses to test mean FS=0.003; train/test ρ=−0.23; no suitable scalar label found) | ✗ (abstract length orthogonal to BERT embeddings — all-zero Final_Score across all 16 variants; no suitable scalar label without re-downloading source data) | ✗ (helpful_vote: 73% zeros; MSE collapses to mode) | ✗ (same as Amazon) | ✗ (same as Amazon) |
| **M3** edge categorical | ✓ (same vs cross book-subject-category per edge) | ✓ (same vs cross CS subdomain per edge) | ✓ (same vs cross L3 subcategory per edge) | ✓ | ✓ |
| **M4** edge scalar | ✓ (absolute price diff per edge) | ✓ (absolute abstract length diff) | ✗ (|helpful_vote_i − helpful_vote_j|: N8 edge zero-rate 65%, N7 44%; train range 0.049 with 4/6 variants at 0; ρ=0.90 is T12e-confound artifact not helpfulness signal) | ✗ (N7 edge zero 50%, N8 70%; ρ=+0.26 p=0.62, not significant) | ✓* (N7 E10b T12a/T12b only: 1,627 edges/sample vs 208 for Amazon, train≈0.28 test≈0.25; N8 and T12e variants degenerate) |
| **M5** global categorical | ~ (derived from M1; aggregate TVD gap) | ~ | ~ | ~ | ~ |
| **M6** global scalar | ✗ (requires M2; M2 excluded for all datasets) | ✗ | ✗ | ✗ | ✗ |

---

## Variant Counts (N×E×T combos per task)

| Task | History | ArXiv | Amazon | Electronics | Toys |
|---|---|---|---|---|---|
| M1 | 4 | 8 | 14 | 14 | 14 |
| M2 | 0 | 0 | 0 | 0 | 0 |
| M3 | 2 | 2 | 2 | 2 | 2 |
| M4 | 2 | 2 | 2 | 2 | 2 |
| M5 | 10 | 16 | 16 | 16 | 16 |
| M6 | 0 | 0 | 0 | 0 | 0 |

---

## Notes

### M2 — History (✗)
**Label attempts (in order)**:
1. Raw price — near-zero Moran's I (max I=+0.098, p=0.61) under all 6 constructions tested. Gate failed.
2. Within-category z-score (global) — same issue; category membership is too coarse a grouping for price.
3. Local z-score `target_i = (price_i − mean_N(i)) / (std_N(i) + ε)` — Moran's I gate passed (4/5 constructions, p < 0.001, mechanically induced). Connectivity diagnostic confirmed ρ(FS, n_edges)=−0.10 (gate pass). **Scoped rerun (6 variants × 20 samples) showed signal does not generalize**: E10c train mean FS=0.05 collapses to test mean FS=0.003; train/test Spearman ρ=−0.23. Gate failed (requires mean > 0.01 AND ρ > 0.3).

**Root cause (final)**: The local z-score target is per-construction and per-graph. It encodes "how does book i's price compare to its specific neighbors in this specific sample graph?" — a question that changes with every new sample. The GNN can memorize this for training graphs with the right structural edges, but the learned representation does not transfer to held-out test samples with different neighborhood structures. There is no globally learnable pattern — only sample-specific fits.

**Consequence**: M2 is excluded for all 5 datasets. M6 (derived) is also excluded for all datasets (requires M2 training rows). `scoring_formula: algebraic_lift` column remains correct for any future M2 variants.

### M2 — ArXiv (✗)
**Attempted target**: `len(text_fidelity_a) − len(text_fidelity_b)` = abstract character count
(config key: `m2_source: derived_from_text`).

**Failure mode**: All-zero Final_Score across all 16 variants (all N×E×T combinations).
BERT sentence embeddings are semantic encoders — they are orthogonal to character count.
Both GNN and MLP achieve R²≈0 → S_GNN = S_MLP = 0 → Δ = 0 → Final_Score = 0.
The character-count signal is not recoverable from the available embedding features.

**Alternatives ruled out**:
- Citation count / year: not in the stored schema without re-downloading source data
- Graph centrality (degree, PageRank): direct C4 violation — the label would encode
  the graph construction being evaluated
- Category-level price analog: ArXiv has no scalar attribute comparable to book price

**Resolution**: ArXiv M2 excluded. No M6 derived rows.

### M2 — Amazon/Electronics/Toys (✗)
**Excluded scalar**: `helpful_vote` (product review upvotes).  
**Root cause**: 73.3% zero-inflation. MSE loss collapses to the mode regardless of
model or graph construction. Converting to binary (is_helpful > 0) degenerates to an
M1-style task. Relative comparison (|vote_i − vote_j|) retains more signal than
absolute prediction from a zero-dominated distribution — M4 retained.  
**No M6**: Derived task requires M2 training rows; excluded downstream.

### M4 — Amazon (✗) and Electronics (✗)
Despite M2 exclusion, M4 was initially retained on the grounds that pairwise
differences reduce zero-inflation relative to absolute prediction. This is true
at N7 (44–50% edge-zero-rate vs 73–78% node-zero-rate), but:

1. **Leaf-artifact ρ=1.000**: The `predict_and_validate` M4 result for Amazon
   reported ρ=1.000 from a dedicated 5-variant tree. This is a Spearman artifact:
   4/6 variants collapsed to train_mean=0.0, so the tree predicts a constant,
   and Spearman of two constant vectors is undefined/1.0 by convention.
2. **T12e confound**: The actual train→test ρ=+0.90 for Amazon is driven by
   T12e (Final_Score=0, zero embeddings produce no signal) vs T12a/T12b (≈0.05).
   This separation reflects embedding type, not learned helpfulness structure.
3. **N8 still fully degenerate**: N8 edge zero-rate ~65–70% — mean-pooling
   per-user does not rescue the zero-inflation.
4. **Electronics ρ=+0.26 p=0.62**: Not significant at any level.

**M4 excluded for Amazon and Electronics.**

### M4 — Toys (✓*)
Toys has substantially more same-user review pairs per 1k-node sample (1,627 vs
Amazon's 208), producing denser E10b graphs and a more informative edge-label
distribution. N7 E10b T12a/T12b variants show train≈0.28, test≈0.25 with
Toys ρ=+0.68 (p=0.14, n=6, not significant but directionally consistent).
Retained with caveat: N8 and T12e variants remain degenerate; result depends on
N7 E10b T12a/T12b being kept (the only 2 non-degenerate variants).

### M5 — Derived (all 5 datasets)
M5 is computed post-hoc from M1 training rows without additional training:  
- **M5**: TVD between predicted and true label distribution over M1 test nodes  
This adds global-level rows to the result CSV at zero extra compute cost.

### M6 — Excluded (all 5 datasets)
M6 derives from M2 training rows. Since M2 is excluded for all 5 datasets (finalized
2026-07-08), M6 is also excluded everywhere. Variant count = 0 for all datasets.

---

## Paper-Ready Statement (M2)

> **Absolute scalar attribute prediction (M2) is excluded for all five datasets.**
> For the three Amazon-family datasets (Amazon, Electronics, Toys), the primary
> scalar attribute (helpful_vote) is 73% zero-inflated, causing regression models
> to collapse to the mode regardless of graph construction. For ArXiv, abstract
> character count is orthogonal to BERT sentence embeddings, producing all-zero
> Final_Score across all 16 variants. For History (book prices), raw price has
> near-zero Moran's I under all tested graph constructions (max I=+0.098, p=0.61).
> A local z-score formulation `target_i = (price_i − mean_N(i)) / (std_N(i) + ε)`
> passes the Moran's I gate with mechanically negative autocorrelation and a
> provable GNN/MLP asymmetry (MLP cannot aggregate neighborhood means at inference
> time), but a scoped validation (6 variants × 20 samples) shows the training
> signal does not generalize: E10c test mean Final_Score = 0.003 (< 0.01 threshold)
> and train/test Spearman ρ = −0.23 (< 0.3 threshold). The local z-score encodes
> sample-specific graph structure rather than a globally learnable book property.
> M6 (derived global scalar gap) is excluded downstream. Future work could explore
> external scalar attributes (publication date, page count) not present in the
> current CS-TAG schema.
