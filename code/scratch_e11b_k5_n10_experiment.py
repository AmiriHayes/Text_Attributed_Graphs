#!/usr/bin/env python3
"""
Full n=10 causal test of E11b's k parameter (k=5 vs k=50) on ArXiv N7 M1
T12a: reuses the 3 pilot samples from scratch_e11b_k5_experiment.py (hardcoded
as PILOT_SCORES/PILOT_EDGES below) and trains 7 additional samples (3-9), then
runs a Welch's t-test on the full n=10 -- see code/AUDIT_pre_publication.md
FIX 3 for the full result writeup.

Same hyperparameters as experiment_runner.py's default M1 GNN run:
hidden_dim=256, lr=0.001, epochs=100, patience=30.

Reads:
  - data/arxiv/train/samples/sample_{03..09}.jsonl (via GenericDataManager;
    samples 0-2 reused from the hardcoded pilot results, not re-read)

Writes:
  - Nothing — prints results to stdout only.

Usage:
  python code/scratch_e11b_k5_n10_experiment.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats
from generic_data_manager import GenericDataManager
from tag_constructor import TAGConstructor
from edge_factory import EdgeFactory
from models import ModelFactory
from trainer import GNNTrainer, compute_step1_score

HIDDEN_DIM = 256
LR = 0.001
EPOCHS = 100
PATIENCE = 30
VARIANT = {'M': 'M1', 'N': 'N7', 'E': 'E11b', 'T': 'T12a'}

# from the n=3 pilot -- reused, not rerun
PILOT_SCORES = {
    50: [0.5987, 0.5618, 0.5396],
    5:  [0.5813, 0.4841, 0.5093],
}
PILOT_EDGES = {
    50: [47405, 47444, 47420],
    5:  [4955, 4964, 4955],
}


def run_samples(k, sample_indices, dm, tc):
    original = EdgeFactory._build_structural_similarity

    def patched(node_list, base_graph, k=k, chunk_size=1000):
        return original(node_list, base_graph, k=k, chunk_size=chunk_size)
    EdgeFactory._build_structural_similarity = staticmethod(patched)

    scores, edges = [], []
    try:
        for i in sample_indices:
            df = dm.load_data('train', i)
            t0 = time.time()
            data = tc.construct(VARIANT, df, 'train')
            n_edges = data.edge_index.shape[1] // 2
            edges.append(n_edges)

            model = ModelFactory.create(
                task_type='categorical', in_dim=data.x.shape[1],
                hidden_dim=HIDDEN_DIM, out_dim=data.y.shape[-1], dropout=0.5,
            )
            trainer = GNNTrainer(
                model=model, device=None, lr=LR, weight_decay=5e-4,
                epochs=EPOCHS, patience=PATIENCE, task_type='categorical',
            )
            metrics = trainer.train(data=data, num_classes=data.y.shape[-1],
                                     verbose=False, early_stopping=True)
            s_gnn, _ = compute_step1_score(metrics, 'categorical')
            scores.append(s_gnn)
            print(f'    k={k} sample_{i:02d}: n_edges={n_edges}  S_GNN_step1={s_gnn:.4f}  ({time.time()-t0:.1f}s)')
    finally:
        EdgeFactory._build_structural_similarity = original

    return scores, edges


if __name__ == '__main__':
    dm = GenericDataManager('arxiv')
    tc = TAGConstructor(dm)
    new_samples = list(range(3, 10))  # 7 additional, samples 0-2 already done

    print('=== E11b k=50 vs k=5, ArXiv N7 M1 T12a, n=10 ===\n')

    print(f'--- k=50: reusing pilot samples 0-2, running {new_samples} ---')
    new_scores_50, new_edges_50 = run_samples(50, new_samples, dm, tc)
    scores_50 = PILOT_SCORES[50] + new_scores_50
    edges_50 = PILOT_EDGES[50] + new_edges_50

    print(f'\n--- k=5: reusing pilot samples 0-2, running {new_samples} ---')
    new_scores_5, new_edges_5 = run_samples(5, new_samples, dm, tc)
    scores_5 = PILOT_SCORES[5] + new_scores_5
    edges_5 = PILOT_EDGES[5] + new_edges_5

    scores_50, scores_5 = np.array(scores_50), np.array(scores_5)

    print(f'\n=== RESULTS (n=10 each) ===')
    print(f'k=50: mean={scores_50.mean()*100:.2f}  std={scores_50.std(ddof=1)*100:.2f}  '
          f'mean_edges={np.mean(edges_50):.0f}')
    print(f'  per-sample (x100): {[f"{s*100:.2f}" for s in scores_50]}')
    print(f'k=5 : mean={scores_5.mean()*100:.2f}  std={scores_5.std(ddof=1)*100:.2f}  '
          f'mean_edges={np.mean(edges_5):.0f}')
    print(f'  per-sample (x100): {[f"{s*100:.2f}" for s in scores_5]}')

    t_stat, p_val = stats.ttest_ind(scores_5, scores_50, equal_var=False)
    pooled_std = np.sqrt((scores_50.var(ddof=1) + scores_5.var(ddof=1)) / 2)
    cohens_d = (scores_5.mean() - scores_50.mean()) / pooled_std

    print(f'\n=== WELCH T-TEST (k=5 vs k=50) ===')
    print(f't-statistic = {t_stat:.4f}')
    print(f'p-value     = {p_val:.4f}')
    print(f"Cohen's d   = {cohens_d:.4f}")
    print(f'delta (k5-k50) x100 = {(scores_5.mean()-scores_50.mean())*100:+.2f}')
    sig = 'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'
    direction = 'increases' if scores_5.mean() > scores_50.mean() else ('decreases' if scores_5.mean() < scores_50.mean() else 'has no effect on')
    print(f'\n=== KEY QUESTION ANSWER ===')
    print(f'Sparsification (k=5 vs k=50) {direction} raw_gnn -- {sig} at alpha=0.05 (p={p_val:.4f})')
