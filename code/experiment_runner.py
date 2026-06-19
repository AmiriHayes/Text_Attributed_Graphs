"""
Experiment Runner — TAG Research
Loops over datasets, variants, splits, and samples; trains a GNN per graph;
collects result rows; saves construction_performance_table_{dataset}.csv.

Usage (from the code/ directory or after adding code/ to sys.path):
    from experiment_runner import ExperimentRunner
    runner = ExperimentRunner(base_path='data', output_path='output')
    runner.run()

Or run directly:
    python experiment_runner.py
"""

import sys
import os
import math
import traceback
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# NOTE on sample counts:
# The original spec mentions "25 from train split + 15 from test split = 40".
# In practice each split contains exactly 20 samples (sample_00 .. sample_19).
# We therefore use all 20 from train (indices 0-19) and all 20 from test
# (indices 0-19), for 40 total samples per variant — consistent with the intent
# of 40 samples, just split 20/20 instead of 25/15.
# ---------------------------------------------------------------------------

N_TRAIN_SAMPLES = 20   # indices 0-19
N_TEST_SAMPLES  = 20   # indices 0-19

DATASETS = ['arxiv', 'amazon', 'history']

# Map M-code to trainer task_type string
M_TO_TASK_TYPE = {
    'M1': 'categorical',
    'M2': 'scalar',
    'M3': 'edge_categorical',
    'M4': 'edge_scalar',
    'M5': 'global_categorical',   # uses GlobalTrainer (global_trainer.py)
    'M6': 'global_scalar',        # uses GlobalTrainer (global_trainer.py)
}

# Default model / trainer hyper-parameters
DEFAULT_HIDDEN_DIM = 256
DEFAULT_DROPOUT    = 0.5
DEFAULT_LR         = 0.001
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_EPOCHS     = 200
DEFAULT_PATIENCE   = 50

# Graphs with more than this many undirected edges are skipped: training on
# near-complete graphs is both intractable on CPU and produces degenerate
# labels (e.g. all-identical centrality → zero-variance → R2 = -∞).
MAX_EDGES = 200_000


def _setup_logging(output_path: Path) -> logging.Logger:
    output_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('experiment_runner')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh = logging.FileHandler(output_path / 'experiment_runner.log')
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

def _dense_graph_row(variant: dict, run_split: str, sample_idx: int,
                     n_edges: int, task_type: str) -> dict:
    """Return a pre-filled degenerate row for graphs that exceed MAX_EDGES."""
    nan = float('nan')
    M = variant['M']
    primary_metric = 'accuracy' if M in ('M1', 'M3') else 'r2'
    return {
        'Task_Idx': M, 'Node_Idx': variant['N'],
        'Edge_Idx': variant['E'], 'Text_Idx': variant['T'],
        'task_type': task_type,
        'KL': nan, 'Top1': nan, 'Top3': nan, 'Cosine': nan,
        'Accuracy': nan, 'F1': nan,
        'MAE': nan, 'MSE': nan, 'R2': nan,
        'Primary_Metric': primary_metric, 'Primary_Value': nan,
        'Performance_Band': nan, 'normalized_score': 0.0,
        'number_of_nodes': nan, 'avg_text_length': nan,
        'text_vocab_entropy': nan, 'label_balance_entropy': nan,
        'output_dimension': nan,
        'run_split': run_split, 'degenerate': True,
        'Predicted_Value': nan, 'True_Value': nan,
        'sample_idx': sample_idx,
    }


def _load_existing_keys(csv_path: Path) -> set:
    """Return set of (Task_Idx, Node_Idx, Edge_Idx, Text_Idx, run_split, sample_idx) already done."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=[
            'Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx', 'run_split', 'sample_idx'
        ])
        return set(zip(df['Task_Idx'], df['Node_Idx'], df['Edge_Idx'],
                       df['Text_Idx'], df['run_split'], df['sample_idx']))
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Outer loop: datasets → variants → splits → samples → train → collect row.
    """

    def __init__(
        self,
        base_path: str = 'data',
        output_path: str = 'output',
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        epochs: int = DEFAULT_EPOCHS,
        patience: int = DEFAULT_PATIENCE,
        early_stopping: bool = False,
        verbose: bool = False,
        device: Optional[str] = None,
        datasets: Optional[List[str]] = None,
    ):
        self.base_path   = Path(base_path)
        self.output_path = Path(output_path)
        self.hidden_dim  = hidden_dim
        self.dropout     = dropout
        self.lr          = lr
        self.weight_decay = weight_decay
        self.epochs      = epochs
        self.patience    = patience
        self.early_stopping = early_stopping
        self.verbose     = verbose
        self.datasets    = datasets or DATASETS

        import torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.logger = _setup_logging(self.output_path)

    def run(self):
        """Run all experiments."""
        for dataset in self.datasets:
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Dataset: {dataset}")
            self.logger.info(f"{'='*60}")
            try:
                self._run_dataset(dataset)
            except Exception as e:
                self.logger.error(f"Fatal error for dataset '{dataset}': {e}")
                self.logger.error(traceback.format_exc())

    def _run_dataset(self, dataset: str):
        from generic_data_manager import GenericDataManager
        from tag_constructor import TAGConstructor
        from variant_registry import VariantRegistry
        from models import ModelFactory
        from trainer import GNNTrainer, build_result_row
        from global_trainer import GlobalTrainer, build_result_row_global

        dm = GenericDataManager(dataset, base_path=str(self.base_path))
        builder = TAGConstructor(dm)
        registry = VariantRegistry(dataset, config_path=str(self.base_path / 'configs'))

        csv_path = self.output_path / f'construction_performance_table_{dataset}.csv'
        existing_keys = _load_existing_keys(csv_path)
        self.logger.info(f"Resuming from {len(existing_keys)} existing rows in {csv_path}")

        all_variants = list(registry.enumerate_variants())
        n_variants   = len(all_variants)
        rows_written  = 0

        for v_idx, variant in enumerate(all_variants):
            M, N, E, T = variant['M'], variant['N'], variant['E'], variant['T']
            variant_str = f"{M}/{N}/{E}/{T}"
            task_type   = M_TO_TASK_TYPE[M]

            self.logger.info(f"[{v_idx+1}/{n_variants}] {dataset} | {variant_str} | task={task_type}")

            # ----------------------------------------------------------------
            # M5 / M6 — global graph-level training path
            # ----------------------------------------------------------------
            if M in ('M5', 'M6'):
                out_dim = 2 if M == 'M5' else 1

                try:
                    # Probe first graph for edge density before committing to
                    # building and training on all 40. Dense graphs (e.g.
                    # N8/E10a category-membership cliques) take hours and
                    # produce degenerate labels anyway.
                    probe = builder.construct(variant, dm.load_data('train', 0), 'train')
                    probe_edges = probe.edge_index.shape[1] // 2
                    if probe_edges > MAX_EDGES:
                        self.logger.warning(
                            f"  SKIP {variant_str}: probe graph has {probe_edges:,} edges "
                            f"(>{MAX_EDGES:,}) — writing degenerate rows"
                        )
                        all_keys = (
                            [('train', i) for i in range(N_TRAIN_SAMPLES)] +
                            [('test',  i) for i in range(N_TEST_SAMPLES)]
                        )
                        for split_name, sample_idx in all_keys:
                            resume_key = (M, N, E, T, split_name, sample_idx)
                            if resume_key in existing_keys:
                                continue
                            row = _dense_graph_row(variant, split_name, sample_idx,
                                                   probe_edges, task_type)
                            out_df = pd.DataFrame([row])
                            write_header = not csv_path.exists()
                            out_df.to_csv(csv_path, mode='a', index=False, header=write_header)
                            existing_keys.add(resume_key)
                            rows_written += 1
                        continue

                    # Load all train-pool and test-pool graphs
                    train_graphs = [probe] + [
                        builder.construct(variant, dm.load_data('train', i), 'train')
                        for i in range(1, N_TRAIN_SAMPLES)
                    ]
                    test_graphs = [
                        builder.construct(variant, dm.load_data('test', i), 'test')
                        for i in range(N_TEST_SAMPLES)
                    ]

                    in_dim = train_graphs[0].x.shape[1]
                    model  = ModelFactory.create(
                        task_type=task_type,
                        in_dim=in_dim,
                        hidden_dim=self.hidden_dim,
                        out_dim=out_dim,
                        dropout=self.dropout,
                    )
                    global_trainer = GlobalTrainer(
                        model=model,
                        task_type=task_type,
                        device=self.device,
                        lr=self.lr,
                        weight_decay=self.weight_decay,
                        epochs=self.epochs,
                        patience=self.patience,
                    )
                    global_trainer.train(train_graphs)
                    self.logger.info(f"  Trained on {len(train_graphs)} train graphs")

                except Exception as e:
                    self.logger.error(f"  ERROR training {variant_str}: {e}")
                    self.logger.debug(traceback.format_exc())
                    continue

                # Evaluate on all 40 graphs individually
                for split_name, graphs in [('train', train_graphs), ('test', test_graphs)]:
                    n_samples = len(graphs)
                    for sample_idx in range(n_samples):
                        resume_key = (M, N, E, T, split_name, sample_idx)
                        if resume_key in existing_keys:
                            continue
                        try:
                            graph      = graphs[sample_idx]
                            df         = dm.load_data(split_name, sample_idx)
                            graph_result = global_trainer.evaluate_single(graph)
                            row = build_result_row_global(
                                variant=variant,
                                graph_result=graph_result,
                                graph=graph,
                                df=df,
                                trainer=global_trainer,
                                run_split=split_name,
                                sample_idx=sample_idx,
                                output_dimension=out_dim,
                            )
                            row['sample_idx'] = sample_idx

                            out_df = pd.DataFrame([row])
                            write_header = not csv_path.exists()
                            out_df.to_csv(csv_path, mode='a', index=False, header=write_header)
                            existing_keys.add(resume_key)
                            rows_written += 1

                            pv = row['Primary_Value']
                            pv_str = f"{pv:.4f}" if not (isinstance(pv, float) and math.isnan(pv)) else 'nan'
                            self.logger.info(
                                f"  {split_name} sample_{sample_idx:02d} | "
                                f"primary={row['Primary_Metric']}={pv_str} | "
                                f"degen={row['degenerate']}"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"  ERROR {variant_str} {split_name} sample_{sample_idx:02d}: {e}"
                            )
                            self.logger.debug(traceback.format_exc())

                continue   # skip the M1-M4 loop below

            # ----------------------------------------------------------------
            # M1-M4 — existing node/edge training path
            # ----------------------------------------------------------------
            for split in ['train', 'test']:
                n_samples = N_TRAIN_SAMPLES if split == 'train' else N_TEST_SAMPLES

                for sample_idx in range(n_samples):
                    # Resume check
                    resume_key = (M, N, E, T, split, sample_idx)
                    if resume_key in existing_keys:
                        continue

                    try:
                        row = self._run_one(
                            dm=dm,
                            builder=builder,
                            variant=variant,
                            task_type=task_type,
                            split=split,
                            sample_idx=sample_idx,
                            ModelFactory=ModelFactory,
                            GNNTrainer=GNNTrainer,
                            build_result_row=build_result_row,
                        )
                        if row is None:
                            continue

                        row['sample_idx'] = sample_idx

                        # Append to CSV
                        out_df = pd.DataFrame([row])
                        write_header = not csv_path.exists()
                        out_df.to_csv(csv_path, mode='a', index=False, header=write_header)
                        existing_keys.add(resume_key)
                        rows_written += 1

                        self.logger.info(
                            f"  {split} sample_{sample_idx:02d} | "
                            f"primary={row['Primary_Metric']}={row['Primary_Value']:.4f} | "
                            f"norm={row['normalized_score']:.1f} | "
                            f"degen={row['degenerate']}"
                        )

                    except Exception as e:
                        self.logger.error(
                            f"  ERROR {variant_str} {split} sample_{sample_idx:02d}: {e}"
                        )
                        self.logger.debug(traceback.format_exc())
                        continue

        self.logger.info(f"Dataset '{dataset}' complete. Rows written this run: {rows_written}")
        self.logger.info(f"Output: {csv_path}")

    def _run_one(
        self, dm, builder, variant, task_type, split, sample_idx,
        ModelFactory, GNNTrainer, build_result_row,
    ) -> Optional[Dict[str, Any]]:
        """
        Train on a single (variant, split, sample_idx) combination and return a result row dict.
        """
        M, N, E, T = variant['M'], variant['N'], variant['E'], variant['T']

        # Load data
        df = dm.load_data(split, sample_idx)

        # Construct PyG Data object
        data = builder.construct(variant, df, split)

        # Guard: skip variants whose graphs are too dense to train on CPU.
        # Near-complete graphs are also scientifically degenerate for scalar
        # tasks (all-identical centrality → zero label variance → R2 = -∞).
        n_edges = data.edge_index.shape[1] // 2
        if n_edges > MAX_EDGES:
            return _dense_graph_row(variant, split, sample_idx,
                                    n_edges, task_type)

        # Determine output_dimension dynamically from data.y
        if M == 'M1':
            # One-hot categorical — data.y.shape = (N_nodes, n_classes)
            output_dimension = data.y.shape[-1]
        elif M == 'M2':
            output_dimension = 1
        elif M == 'M3':
            output_dimension = 2
        elif M == 'M4':
            output_dimension = 1
        else:
            raise ValueError(f"Unexpected M: {M}")

        # Input feature dimension
        in_dim = data.x.shape[1]

        # Create model
        model = ModelFactory.create(
            task_type=task_type,
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=output_dimension,
            dropout=self.dropout,
        )

        # Create trainer
        trainer = GNNTrainer(
            model=model,
            device=self.device,
            lr=self.lr,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            patience=self.patience,
            task_type=task_type,
        )

        # num_classes for top-k metrics (only meaningful for M1)
        num_classes = output_dimension

        # Train
        test_metrics = trainer.train(
            data=data,
            num_classes=num_classes,
            verbose=self.verbose,
            early_stopping=self.early_stopping,
        )

        # Build result row
        row = build_result_row(
            variant=variant,
            test_metrics=test_metrics,
            data=data,
            df=df,
            trainer=trainer,
            run_split=split,
            sample_idx=sample_idx,
            output_dimension=output_dimension,
        )

        # M1-M4 rows don't have per-sample predictions; fill with NaN
        row['Predicted_Value'] = float('nan')
        row['True_Value']      = float('nan')

        return row


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run TAG experiments')
    parser.add_argument('--base_path',   default='data',   help='Path to data/ directory')
    parser.add_argument('--output_path', default='output', help='Path to output/ directory')
    parser.add_argument('--hidden_dim',  type=int,   default=DEFAULT_HIDDEN_DIM)
    parser.add_argument('--epochs',      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument('--lr',          type=float, default=DEFAULT_LR)
    parser.add_argument('--datasets',    nargs='+',  default=DATASETS)
    parser.add_argument('--device',      default=None)
    parser.add_argument('--verbose',     action='store_true')
    parser.add_argument('--early_stopping', action='store_true')
    args = parser.parse_args()

    runner = ExperimentRunner(
        base_path=args.base_path,
        output_path=args.output_path,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        verbose=args.verbose,
        early_stopping=args.early_stopping,
        datasets=args.datasets,
    )
    runner.run()


if __name__ == '__main__':
    # When running from the repo root, add code/ to sys.path
    sys.path.insert(0, str(Path(__file__).parent))
    main()
