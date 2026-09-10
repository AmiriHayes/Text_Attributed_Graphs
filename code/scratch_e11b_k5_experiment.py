#!/usr/bin/env python3
"""
Pilot causal test of E11b's k parameter (k=5 vs the old default k=50) on
ArXiv N7 M1 T12a: trains 3 train-samples at each k and compares raw_gnn
scores, to check the direction/magnitude of the sparsification fix before
committing to the full n=10 run (see scratch_e11b_k5_n10_experiment.py) --
see code/AUDIT_pre_publication.md FIX 3 for the full writeup.

Same hyperparameters as experiment_runner.py's default M1 GNN run:
hidden_dim=256, lr=0.001, epochs=100, patience=30.

Reads:
  - data/arxiv/train/samples/sample_{00,01,02}.jsonl (via GenericDataManager)

Writes:
  - Nothing — prints results to stdout only.

Usage:
  python code/scratch_e11b_k5_experiment.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from generic_data_manager import GenericDataManager
from tag_constructor import TAGConstructor
from label_factory import LabelFactory
import edge_factory as edge_factory_module
from edge_factory import EdgeFactory
from models import ModelFactory
from trainer import GNNTrainer, compute_step1_score

N_SAMPLES = 3
HIDDEN_DIM = 256
LR = 0.001
EPOCHS = 100
PATIENCE = 30
VARIANT = {'M': 'M1', 'N': 'N7', 'E': 'E11b', 'T': 'T12a'}


def run_for_k(k: int, dm, tc):
    """Monkey-patch _build_structural_similarity's default k, build+train
    on N_SAMPLES train samples, restore, return list of train scores."""
    original = EdgeFactory._build_structural_similarity

    def patched(node_list, base_graph, k=k, chunk_size=1000):
        return original(node_list, base_graph, k=k, chunk_size=chunk_size)
    EdgeFactory._build_structural_similarity = staticmethod(patched)

    scores = []
    edge_counts = []
    try:
        for i in range(N_SAMPLES):
            df = dm.load_data('train', i)
            t0 = time.time()
            data = tc.construct(VARIANT, df, 'train')
            n_edges = data.edge_index.shape[1] // 2
            edge_counts.append(n_edges)

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

    return scores, edge_counts


if __name__ == '__main__':
    dm = GenericDataManager('arxiv')
    tc = TAGConstructor(dm)

    print(f'=== E11b k=50 (current default) vs k=5, ArXiv N7 M1, {N_SAMPLES} train samples ===\n')

    print('--- k=50 (baseline, should roughly match historical train_mean~54.7 for T12a) ---')
    scores_50, edges_50 = run_for_k(50, dm, tc)

    print('\n--- k=5 (sparsified) ---')
    scores_5, edges_5 = run_for_k(5, dm, tc)

    print(f'\n=== RESULTS ===')
    print(f'k=50: mean_edges={np.mean(edges_50):.0f}  mean_S_GNN_step1={np.mean(scores_50):.4f} (x100={np.mean(scores_50)*100:.2f})  scores={[f"{s:.3f}" for s in scores_50]}')
    print(f'k=5 : mean_edges={np.mean(edges_5):.0f}  mean_S_GNN_step1={np.mean(scores_5):.4f} (x100={np.mean(scores_5)*100:.2f})  scores={[f"{s:.3f}" for s in scores_5]}')
    print(f'\nHistorical k=50 train_mean (full 10-sample, paper-reported): 54.67 (T12a)')
    print(f'Delta (k5 - k50) x100: {(np.mean(scores_5) - np.mean(scores_50))*100:+.2f}')
