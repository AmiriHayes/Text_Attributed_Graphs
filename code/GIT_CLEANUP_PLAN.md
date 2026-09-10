# Git Cleanup Plan (proposal only — no git commands executed)

## 0. `run_clean_20260730` does not exist

The request assumes `output/run_clean_20260730` is a real pre-audit run
directory to exclude. **It is not on disk anywhere in this repo** (checked
`find . -iname "*run_clean*" -o -iname "*20260730*"` — zero hits). The
real pre/post-audit split in this repo is:

| Directory | Size | What it actually is |
|---|---|---|
| `output/run_1000_final/` | 129M | **Pre-audit-fix** run (timestamped 2026-08-17, before the E11c/T12e/E11b-k/N8-lookup fixes). This is the run all the decision-tree PNGs and cross-dataset ρ/TED tables currently in the repo were built from. |
| `output/run_500_final/` | 42M | Also pre-audit-fix, smaller sample count; appears to be an earlier/alternate version of the same run. |
| `output/run_post_audit/` | 3.0M | **Post-audit-fix** (corrected) run, timestamped 2026-08-29/30. Has the 5 `construction_performance_table_*.csv` files but **no decision-tree analysis has been run on it yet** — no `analysis/` subfolder exists (confirmed via `validate_pipeline.py` CHECK 3). |
| `output/run_1000_10s/` | 9.5M | A 10-sample variant run; unclear provenance, has its own `decision_tree_*_raw_gnn.png` set with a different naming convention than the other two. |

**This changes the recommendation.** Option C ("delete the pre-audit run
after verifying post_audit is complete") is not safe to execute right now —
`run_post_audit` is not complete: it has no tree-consistency analysis, and
every currently-published decision-tree figure/table in the repo is sourced
from the pre-audit `run_1000_final`. Deleting it today would delete the only
copy of data the paper's current tables and figures were built from, with no
replacement yet computed on the corrected pipeline.

**Recommendation: Option B, with a concrete follow-up.** Keep
`run_1000_final` (canonical for GNN-config numbers already in the draft, per
this session's earlier "100 random subsets" verification) and
`run_post_audit` (canonical for the audit-fixed pipeline, once its own
analysis is run) side by side, each with a short `README.md`:
- `output/run_1000_final/README.md`: "Pre-audit-fix run. Canonical source for
  [tables X, Y currently in the draft] as of [date]. Superseded by
  run_post_audit once its decision-tree analysis is complete — see
  AUDIT_pre_publication.md."
- `output/run_post_audit/README.md`: "Post-audit-fix (corrected) run — see
  AUDIT_pre_publication.md for the 4 fixes applied. Decision-tree /
  cross-dataset analysis not yet run on this data (as of [date]); until it
  is, tables in the draft citing tree-consistency numbers still reflect
  run_1000_final."

Once `run_analysis.py --run-dir output/run_post_audit` is actually run and
its outputs verified against the pre-audit numbers, Option C becomes safe —
not before.

The small `run_20260321_*` / `run_20260407_*` directories (4–24K each) look
like leftover debug runs; recommend deleting these regardless of the A/B/C
decision above — they carry no analysis and are already tiny, but there's
no reason to publish them in an anonymous repo. (Not executed — flagging
for the author's own `rm -rf`.)

---

## 1. Author-identifying information

**`README.md:3`** — contains a real name and institution:
```
...@ NJIT Fall '25, Advisor: Kristina Wicke.
```
This is the one clear author-deanonymizing string found in the repo. No
other tracked `.py`, `.yaml`, or `.md` file contains a name, email, or
institution — searched for the session's own git user/email
(`amkaha2018@gmail.com`, "Amiri Hayes") and found zero matches in tracked
source files. **Action for anonymous submission: redact or remove this line
from `README.md`** (or maintain a separate anonymized README for the
submission branch/repo).

---

## 2. API keys / credentials

**Clean — nothing to flag beyond confirming the existing setup is correct.**
- `.env` exists locally but is `.gitignore`d (line present) **and was never
  committed** — `git log --all --full-history -- .env` returns no history.
- `kaggle.json` is likewise gitignored and never committed.
- The only files matching `api_key`/`API_KEY` patterns
  (`data/graphrag/generate_qa.py`) reference `os.environ.get('OPENAI_API_KEY')`
  — reading from environment/`.env`, never a hardcoded key. Confirmed by
  reading the matched lines directly, not just the grep hit.

No credential files need to be added to `.gitignore` — they already are, and
the ignore has held since day one for this repo's history.

---

## 3. Large binary files — and a real problem beyond size

**The size threshold (>10MB) finding overlaps with a more important one: the
`.gitignore` already declares intent to exclude the embedding `.npy` files,
but they are tracked anyway** — because they were committed before the
ignore rule was added, and adding a pattern to `.gitignore` does not retroactively
untrack already-committed files. This is very likely why `.git/` is **859MB**
despite the ignore file's own comment ("Pre-computed SBERT embeddings —
regeneratable from raw text, several files >100MB").

Confirmed via `git rev-list --objects --all` + `git cat-file --batch-check`
(largest blobs across all of git history, not just the working tree):

| File | Size | Status |
|---|---|---|
| `data/toys/test/embeddings/review_embeddings_contextual.npy` | 85.1MB | tracked, despite `.gitignore` covering `data/*/test/embeddings/` |
| `data/toys/test/embeddings/review_embeddings.npy` | 85.1MB | tracked, same pattern |
| `data/subsets/embeddings_00/author_embeddings.npy` | 82.5MB | tracked; `data/subsets/` not covered by any current ignore rule |
| `data/electronics/test/embeddings/review_embeddings_contextual.npy` | 80.3MB | tracked despite ignore |
| `data/electronics/test/embeddings/review_embeddings.npy` | 80.3MB | tracked despite ignore |
| `output/run_1000_final/run.log` | 77.5MB | tracked despite `.gitignore`'s `*.log` rule |
| `data/author_embeddings.npy`, `code/data/author_embeddings.npy` | 72.1MB each | tracked, two copies, neither covered by an ignore rule |
| `data/toys/test/embeddings/family_embeddings.npy` | 40.9MB | tracked despite ignore |
| `data/electronics/test/embeddings/family_embeddings.npy` | 34.4MB | tracked despite ignore |
| `data/{arxiv,amazon,history}/{train,test}/raw.jsonl` and matching `.csv` | 19–57MB each | tracked, **not** covered by any ignore rule — likely intentional (needed to reproduce experiments), separate issue from the ones above |

**Action items (not executed — all require the author's own git commands):**
1. For files the `.gitignore` already intends to exclude but are still
   tracked (all the `embeddings/*.npy` files, `run.log`): `git rm --cached`
   them from the current tip, then rewrite history with `git filter-repo`
   (or BFG) to actually shrink `.git/` — a plain `git rm --cached` alone
   only stops future tracking, it does not remove the 859MB already baked
   into history.
2. `data/subsets/embeddings_00/author_embeddings.npy` and
   `data/author_embeddings.npy` / `code/data/author_embeddings.npy` are not
   covered by any existing `.gitignore` pattern — decide whether these are
   regeneratable (like the other embeddings) and add a rule, or keep them
   intentionally.
3. The raw `.jsonl`/`.csv` data files are large but appear to be
   intentionally tracked (they're the actual dataset, not a byproduct) —
   flagged for awareness, not necessarily an action item. If the anonymous
   review copy needs to be small, these would be the next thing to consider
   trimming, separately from the accidental-tracking issue above.

Doing (1) is what would actually bring `.git/` down from 859MB — the current
working-tree `output/*` sizes (129M + 42M + 9.5M + ...) are dwarfed by what's
sitting in git history.
