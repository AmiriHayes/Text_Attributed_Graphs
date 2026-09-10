#!/usr/bin/env python3
"""
Trains 6 ArXiv M1 variants (3 top-quartile, 3 bottom-quartile by train_mean,
excluding the two degenerate T12e ablations), 3 samples each, with per-epoch
test-mask accuracy logging enabled -- for Figure 2's epoch-curve panel.

Reads:
  - data/arxiv/train/samples/sample_{00,01,02}.jsonl (via GenericDataManager)

Writes:
  - output/epoch_figure_arxiv/epoch_logs_arxiv_{N}_{E}_{T}_sample{s}.csv

Usage:
  python code/run_epoch_figure_arxiv.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generic_data_manager import GenericDataManager
from tag_constructor import TAGConstructor
from models import ModelFactory
from trainer import GNNTrainer, compute_step1_score

HIDDEN_DIM = 256
LR = 0.001
EPOCHS = 100
PATIENCE = 30
OUT_DIR = Path(__file__).resolve().parent.parent / 'output/epoch_figure_arxiv'
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    # (N, E, T, tier)
    ('N8', 'E10b', 'T12b', 'top'),
    ('N8', 'E11a', 'T12a', 'top'),
    ('N8', 'E11b', 'T12b', 'top'),
    ('N7', 'E10b', 'T12b', 'bottom'),
    ('N7', 'E10c', 'T12b', 'bottom'),
    ('N7', 'E11b', 'T12b', 'bottom'),
]
SAMPLES = [0, 1, 2]

if __name__ == '__main__':
    dm = GenericDataManager('arxiv')
    tc = TAGConstructor(dm)

    for N, E, T, tier in VARIANTS:
        variant = {'M': 'M1', 'N': N, 'E': E, 'T': T}
        exp_id = f'{N}_{E}_{T}'
        for sample_idx in SAMPLES:
            t0 = time.time()
            df = dm.load_data('train', sample_idx)
            data = tc.construct(variant, df, 'train')

            model = ModelFactory.create(
                task_type='categorical', in_dim=data.x.shape[1],
                hidden_dim=HIDDEN_DIM, out_dim=data.y.shape[-1], dropout=0.5,
            )
            trainer = GNNTrainer(
                model=model, device=None, lr=LR, weight_decay=5e-4,
                epochs=EPOCHS, patience=PATIENCE, task_type='categorical',
            )
            log_path = OUT_DIR / f'epoch_logs_arxiv_{exp_id}_sample{sample_idx}.csv'
            metrics = trainer.train(
                data=data, num_classes=data.y.shape[-1], verbose=False,
                early_stopping=True, epoch_log_path=str(log_path),
                experiment_id=f'{exp_id}_s{sample_idx}',
            )
            s_gnn, _ = compute_step1_score(metrics, 'categorical')
            print(f'[{tier}] {exp_id} sample_{sample_idx}: S_GNN_step1={s_gnn:.4f}  '
                  f'test_top1={metrics.get("top1", float("nan")):.4f}  '
                  f'({time.time()-t0:.1f}s)  -> {log_path.name}')

    print('\nDONE')
