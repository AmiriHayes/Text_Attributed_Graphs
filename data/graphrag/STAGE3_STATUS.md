# Stage 3 (QA generation + RAGAS gating) — status and next moves

Context for planner: this summarizes everything decided/found across the ArXiv
first-pass work before scaling to Amazon/Electronics/History/Toys. Original
proposal assumed 3 datasets and 100 Q/dataset (50 single-hop + 50 multi-hop);
we're now on 5 datasets and have deviated from the proposal in specific,
deliberate ways documented below.

## What's built and validated (ArXiv only)

- `data/graphrag/format_rows.py` — row → clean text (title+abstract via
  `text_fidelity_a`), filters <100 chars.
- `data/graphrag/generate_qa.py` — `generate_qa()` (single-hop),
  `generate_aggregate_qa()` (cross-paper), `validate_qa()`,
  `classify_question_type()`, `sample_cross_category_groups()`.
- `data/graphrag/run_question_generation.py` — full generation loop, saves
  `data/arxiv/questions.csv`.
- `data/graphrag/run_full_ragas.py` — builds a plain `VectorStoreIndex`
  (LlamaIndex, no graph) over the source pool, scores all questions with
  RAGAS, saves `data/arxiv/ragas_baseline.csv`.
- `data/graphrag/evaluate_rag.py` — shared index-building + RAGAS-scoring
  helpers used by the above.
- LLM: OpenAI `gpt-4o-mini` via `.env` (`OPENAI_API_KEY`) — not
  Anthropic/Colab as originally proposed. Cheap (~$0.15 for 50
  questions generated + fully RAGAS-scored).
- `ragas==0.2.15` pinned in `requirements.txt` — latest (0.4.x) fails to
  import in this environment (`langchain_community.chat_models.vertexai`
  no longer exists upstream). Also had to protect a pre-existing
  `langgraph`/`langchain-openai` install on this machine from a
  transitive-dependency downgrade while fixing this — resolved, noted in
  case it recurs on another machine.
- Output format is **CSV, not JSONL** — deliberate deviation from the
  original spec, for quick visual validation. Keep this going forward.
- Code lives in `data/graphrag/`, not `code/graphrag/` — treated as
  one-off/exploratory in the spirit of `data/write_data.ipynb`, not
  general reusable pipeline code.

## Current ArXiv results (50 Q: 35 single-hop + 15 aggregate)

```
single-hop (35):  faithfulness 0.909±0.171  answer_relevance 0.790±0.313  context_relevance 0.929±0.247  → STRONG
aggregate  (15):  faithfulness 0.720±0.291  answer_relevance 0.816±0.081  context_relevance 0.533±0.481  → STRONG
```
Aggregate context_relevance is bimodal: 6 questions at 0.0, 2 at 0.5, 7 at 1.0
— not a soft degradation, a real split. delta vs single-hop on
context_relevance is -0.395, reproduced twice (10-Q gate and full 50).

## Key findings (grounded, not opinions)

1. **Word-count floor was over-rejecting.** `validate_qa()`'s question-length
   floor was `>15 words`; the harder v2 prompt (Change 1: forbid trivial
   "what does this paper propose" questions) naturally produces terser,
   more technical phrasing, causing ~62% rejection at 15-word floor. Fixed
   to `>10`; rejection rate dropped to 0% in the final 50-question run.
2. **JSON/LaTeX corruption bug, found and fixed.** Source abstracts contain
   correctly-escaped LaTeX (`\\nu`, `\\frac`). The LLM's own JSON responses
   sometimes emit single-backslash LaTeX, and `\n`/`\f`/`\t`/`\r` are
   reserved JSON escapes — `json.loads()` silently corrupted text
   (`\nu=\frac{1}{-1}` → garbled, and in one case the model also
   misrepresented the actual physics claim while corrupting the notation).
   Fixed with a prompt instruction (paraphrase LaTeX in plain English) +
   validator-level control-char regex safety net. Confirmed clean on a
   targeted 10-row test filtered to backslash-heavy rows (2,363/9,178 pool
   rows contain backslashes).
3. **Aggregate ("cross-paper") mechanism is literal random-sample-then-
   reject-until-diverse, not deliberately curated.** `sample_cross_category_
   groups()` draws 5-10 rows uniformly at random from the whole pool,
   checks post-hoc for ≥3 distinct `aggregate_id` (top-level category)
   values, redraws up to 20x if not met. It works empirically (the
   context_relevance delta above) but category-label diversity is an
   imperfect proxy for real topical/semantic diversity — one group labeled
   `['astro-ph','cond-mat','math']` was thematically all
   galactic-structure/orbital-dynamics content and scored context_relevance
   1.0 anyway (retrieval found it easy despite passing the label check).
4. **Reference answers are 100% LLM-generated, not ground truth, and
   currently unused.** `generate_qa()`/`generate_aggregate_qa()` produce
   both question and answer in one call — there's no independent answer
   key. None of the three RAGAS metrics in use (faithfulness,
   answer_relevance, context_relevance) actually consume
   `reference_answer` — it's saved for human inspection only. Open
   decision: add `answer_correctness`/`context_recall` (both need a
   reference) since we're already generating one.
5. **Aggregate is NOT true multi-hop — real gap vs. the paper's thesis.**
   No graph edge is ever traversed; it's random rows filtered by label
   diversity. GraphRAG's actual pitch is retrieval via structural/edge
   proximity, not just "harder because topically scattered." If Part 3's
   eval set never requires using an edge to answer, a positive correlation
   wouldn't actually validate that the E axis (which Part 1/2 spent all
   the effort characterizing — E10a/b/c, E11a/b/c) specifically drives
   GraphRAG quality. **Decision: add a third question type, edge-based
   multi-hop**, via genuine N7→N8→N7 (or N7→N9→N7) two-hop traversal —
   same mechanism we'd sketched earlier (shared `secondary_id`, e.g.
   co-authorship) before the planner-bot spec redirected toward
   category-based aggregate. Not yet implemented or gate-tested.
6. **Config selection (Part 3) has a real methodology flaw, confirmed
   empirically on ArXiv.** Naive "top-K by raw train_mean" score
   clusters almost entirely on `Task_Idx`/`Node_Idx` (these dominate
   decision-tree feature importance across datasets — e.g. amazon:
   Node_Idx 0.575, Task_Idx 0.422, Edge+Text axes combined <0.01).
   Checked directly: ArXiv's "top 10 by score" are **all M1, all N8**,
   differing only in `E10b` vs `E10c` and `T12a/b/e`. Picking "5 good"
   this way gives one construction recipe with text-fidelity jitter, not
   5 independent tests of different edge-construction mechanisms — which
   undermines the entire point of Part 3 (testing whether E-axis choice
   predicts GraphRAG quality). **Decision: selection needs to be
   diversity-constrained** (e.g. require distinct `Edge_Idx`, or distinct
   `(Node_Idx, Edge_Idx)`, within each performance band), not naive top-K.
   Not yet implemented.
7. **Community-based vs. text-based GraphRAG system types are not equally
   diagnostic.** Text-based injects the constructed edges directly into
   retrieval — a direct, attributable reflection of the E-axis choice.
   Community-based runs Louvain clustering on top of the graph first, an
   abstraction layer that could smooth over genuinely different
   edge-construction choices into similar community partitions, weakening
   attribution to the specific E-axis value. Keep both (community-based is
   a real, common GraphRAG pattern), but know text-based is the sharper
   instrument for the headline claim.
8. **Negation-style questions are a single-hop failure mode.** 3/35 v2
   single-hop questions scored `answer_relevance = 0.000` exactly — all
   were "what is NOT discussed in this paper" style questions, which the
   downstream RAG answerer hedges on ("cannot be determined from the
   available context"), and RAGAS zeroes non-committal answers by design.
   Traces back to Change 1's "limitations/failure modes" framing
   occasionally producing pure negations. Not yet fixed in the prompt.
9. **Cost math for Part 3 scaling** (formula: systems =
   `n_datasets × (n_configs × 2_automatic_types + 1_KG_baseline)`;
   evaluations = `systems × N_questions_per_dataset`; ~$0.002/evaluation
   extrapolated from the $0.15-for-50 anchor — not an exact audit, no
   billing-API access with this key):

   | configs | N/dataset | evaluations | ≈ cost |
   |---|---|---|---|
   | 15 (5/5/5) | 100 (orig. proposal) | 15,500 | $31 |
   | 15 (5/5/5) | 150 | 23,250 | $46.50 |
   | 9 (3/3/3) | 250 | 23,750 | $47.50 |
   | 15 (5/5/5) | 500 | 77,500 | $155 |

   Given finding #6 above, **raw config *count* isn't the right lever to
   reason about cost/power tradeoffs on until selection is fixed** — a
   diversity-constrained n=9 could carry more real information than a
   naive top-K n=15.
10. **Sample pool should expand to `sample_00`-`sample_49`** (all 50
    files, pooled+deduped) — confirmed 31,943 unique rows vs. 9,178 from
    10 files, out of arxiv's 50,000-row corpus. Reduces repeat-paper risk
    as N grows, "completionism" also just correct given these already
    exist on disk.

## Visuals/figures inventory (separate thread, not blocking Stage 3)

- Exist and reproducible: `decision_tree_{dataset}_combined.png` (both
  run_500_final and run_1000_final), `ablation_heatmap_{dataset}.png`,
  cross-dataset rho/TED/p CSVs (both run sizes).
- Exist, reproducible with a fix: `fig_01`-`fig_08` report figures
  (variant barcharts, rho/TED tables, rho-vs-TED scatter, per-task rho,
  band accuracy, strategy1 summary) — generated once, deleted from working
  tree in a later cleanup commit, generator (`code/generate_report_
  figures.py`) still points at a stale `output/analysis/` path instead of
  `output/run_1000_final`.
- Exist, **not reproducible**: `checks/` (7 diagnostic PNGs) and
  `adversarial_checks.pdf` in `run_1000_final/analysis/` — source script
  was never committed, added as raw binaries in a past commit. Same
  situation as the missing combined/general decision tree.
- Missing entirely: Paper Figure #3 ("training curve variance") — no code
  generates this anywhere. Paper Figure #4 (combined-tree, good/bad paths
  highlighted) and the true 3×4 generalization matrices — blocked on
  `D^(gen)_T` (cross-dataset combined tree) not existing.
- 500-vs-1000 comparison: **nothing persisted** — only a one-off chat
  widget this session, not saved as a file/script output.

## Open decisions (need explicit answers before scaling past ArXiv)

1. Diversity-constrained config selection rule — exact tie-breaking logic,
   and re-derive what "N good/mid/bad configs" actually looks like once
   duplicates are excluded (per finding #6).
2. Whether Part 3 config selection stays "global" (needs `D^(gen)_T`,
   which doesn't exist) or switches to per-dataset selection (deviates
   from the original proposal's wording, but avoids depending on a tree
   that doesn't exist yet and may be more principled anyway given dataset
   heterogeneity, e.g. history has no N8).
3. Final N (questions/dataset) and the single/aggregate/edge split ratio
   — leaning equal thirds by default, not locked. Depends partly on (1).
4. Whether to add `answer_correctness`/`context_recall` RAGAS metrics now
   that `reference_answer` is confirmed generated-but-unused.
5. Fate of the `checks/`/`adversarial_checks.pdf` diagnostics and
   `D^(gen)_T` — rebuild, or accept the gap.
6. Whether a persisted 500-vs-1000 comparison artifact is needed for the
   paper (currently ephemeral only).

## Concrete next moves, roughly in order

1. Implement diversity-constrained config selection (extend
   `code/run_analysis.py` or a new script); re-run against ArXiv's actual
   data to see what real "5 good / 5 mid / 5 bad" (or whatever count)
   looks like once duplicates are excluded. Resolves open items 1+3.
2. Design + implement + gate-test edge-based multi-hop (mirror the
   single-hop/aggregate build→gate→iterate pattern already used):
   dataset-agnostic hop mechanism (`secondary_id` where dense enough,
   `aggregate_id` fallback otherwise — same pattern as aggregate's
   grouping-key selection), own prompt, reuse `validate_qa()`, gate on
   ~10 before trusting it.
3. Fix the negation-question failure mode in the single-hop prompt
   (finding #8).
4. Once (1)-(3) are settled: lock final N, config count/composition, and
   split ratio; update `run_question_generation.py`/`run_full_ragas.py`;
   switch source pool to `sample_00`-`sample_49`.
5. Re-run the full pipeline for ArXiv once more as the template/reference
   pass (3 question types, corrected config selection upstream, expanded
   pool).
6. Port the validated recipe to Amazon, Electronics, History, Toys —
   specifically stress-test the dataset-agnostic fallback logic on
   History (no `secondary_id` at all) and Amazon-family (sparse
   `secondary_id`).
7. Separately, not blocking: decide on `D^(gen)_T`, regenerate
   `fig_01`-`fig_08`, decide fate of `checks/`, build a persisted
   500-vs-1000 figure if the paper needs one.
