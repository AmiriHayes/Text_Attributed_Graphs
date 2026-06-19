"""
Global Trainer — TAG Research
Trains graph-level GNN models (M5 global_categorical, M6 global_scalar) using
torch_geometric DataLoader to batch multiple Data objects.

Unlike GNNTrainer (which operates on a single graph with node/edge masks),
GlobalTrainer operates across a *pool* of graphs, one label per graph.

  M5 — binary graph classification  → CrossEntropyLoss, accuracy metric
  M6 — graph-level regression       → MSELoss, MAE/MSE metric (R² aggregated externally)
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool   # imported for re-export / doc clarity
from typing import Dict, List, Optional, Any

from trainer import _text_stats, RESULT_COLUMNS


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------

def build_loader(graphs: list, batch_size: int) -> DataLoader:
    """
    Wrap a list of PyG Data objects into a DataLoader.

    Args:
        graphs:     List of torch_geometric.data.Data objects
        batch_size: Mini-batch size

    Returns:
        DataLoader that yields batched Data objects with a .batch tensor
    """
    return DataLoader(graphs, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# GlobalTrainer
# ---------------------------------------------------------------------------

class GlobalTrainer:
    """
    Trains SAGEGlobalPredictor models on a pool of graph-level Data objects.

    Training is "full-batch over train-pool": all train graphs are loaded in
    a single forward pass per epoch (batch_size = len(train_graphs)).

    Early stopping is based on *train* loss (there is no separate val split
    for M5/M6 — the 20 test-pool graphs are held out for final evaluation).
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: str,
        device: str,
        lr: float = 0.001,
        weight_decay: float = 5e-4,
        epochs: int = 200,
        patience: int = 50,
    ):
        """
        Args:
            model:        SAGEGlobalPredictor instance (already instantiated)
            task_type:    'global_categorical' (M5) or 'global_scalar' (M6)
            device:       'cpu' or 'cuda'
            lr:           Adam learning rate
            weight_decay: L2 regularisation
            epochs:       Maximum training epochs
            patience:     Early-stopping patience (epochs without train-loss improvement)
        """
        assert task_type in ('global_categorical', 'global_scalar'), (
            f"GlobalTrainer only handles 'global_categorical'/'global_scalar', got '{task_type}'"
        )
        self.model       = model.to(device)
        self.task_type   = task_type
        self.device      = device
        self.epochs      = epochs
        self.patience    = patience

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        if task_type == 'global_categorical':
            # M5: binary graph classification — raw logits, CrossEntropyLoss
            self.criterion = nn.CrossEntropyLoss()
        else:
            # M6: graph-level regression — raw scalar, MSELoss
            self.criterion = nn.MSELoss()

        self.training_history: List[Dict] = []   # one entry per epoch: {'epoch', 'loss'}
        self.best_loss  = float('inf')
        self.best_state = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, train_graphs: list) -> None:
        """
        Train model on a pool of graph-level Data objects.

        Full-batch over train-pool: all graphs in one forward pass per epoch.
        Early stopping on train loss (no val split available for M5/M6).

        Args:
            train_graphs: List of torch_geometric.data.Data objects,
                          each with data.y.shape == [1]
        """
        if len(train_graphs) == 0:
            return

        # One batch = entire train pool
        loader = build_loader(train_graphs, batch_size=len(train_graphs))

        no_improve = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.batch)

                if self.task_type == 'global_categorical':
                    # out: (B, 2),  batch.y: (B,) long
                    loss = self.criterion(out, batch.y.long())
                else:
                    # out: (B, 1),  batch.y: (B,) float → reshape to (B, 1)
                    tgt = batch.y.float().view(-1, 1)
                    loss = self.criterion(out, tgt)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.training_history.append({'epoch': epoch, 'loss': epoch_loss})

            # Early stopping on train loss
            if epoch_loss < self.best_loss:
                self.best_loss  = epoch_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                break

        # Restore best
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_single(self, graph) -> Dict[str, Any]:
        """
        Run a forward pass on ONE graph and return per-graph metrics.

        Args:
            graph: A single torch_geometric.data.Data object

        Returns:
            For M5 (global_categorical):
                {'pred_class': int, 'true_class': int, 'correct': int}
                where correct ∈ {0, 1}
            For M6 (global_scalar):
                {'predicted_value': float, 'true_value': float,
                 'mae': float, 'mse': float}
        """
        self.model.eval()

        # Wrap single graph in a DataLoader (batch_size=1) to get batch tensor
        loader = build_loader([graph], batch_size=1)
        batch  = next(iter(loader))
        batch  = batch.to(self.device)

        out = self.model(batch.x, batch.edge_index, batch.batch)

        if self.task_type == 'global_categorical':
            pred_class = int(out.argmax(dim=-1).item())
            true_class = int(batch.y.item())
            correct    = int(pred_class == true_class)
            return {'pred_class': pred_class, 'true_class': true_class, 'correct': correct}

        else:
            # M6
            predicted_value = float(out.item())
            true_value      = float(batch.y.item())
            mae = abs(predicted_value - true_value)
            mse = (predicted_value - true_value) ** 2
            return {
                'predicted_value': predicted_value,
                'true_value':      true_value,
                'mae':             mae,
                'mse':             mse,
            }

    # ------------------------------------------------------------------
    # Degenerate detection
    # ------------------------------------------------------------------

    def is_degenerate(self, window: int = 20, threshold: float = 0.01) -> bool:
        """
        Return True if total loss improvement over the first `window` epochs < threshold.
        Mirrors GNNTrainer.is_degenerate().
        """
        if len(self.training_history) < window:
            return False
        first_loss = self.training_history[0]['loss']
        last_loss  = self.training_history[window - 1]['loss']
        return abs(first_loss - last_loss) < threshold


# ---------------------------------------------------------------------------
# Result row builder
# ---------------------------------------------------------------------------

def build_result_row_global(
    variant:          Dict,
    graph_result:     Dict[str, Any],
    graph,            # PyG Data object (single graph)
    df,               # pandas DataFrame (raw sample, for text stats)
    trainer:          'GlobalTrainer',
    run_split:        str,
    sample_idx:       int,
    output_dimension: int,
) -> Dict[str, Any]:
    """
    Produce a result row matching the 25-column RESULT_COLUMNS schema,
    PLUS two extra columns: ``Predicted_Value`` and ``True_Value``.

    Args:
        variant:          {'M': ..., 'N': ..., 'E': ..., 'T': ...}
        graph_result:     dict returned by GlobalTrainer.evaluate_single()
        graph:            PyG Data object for this sample
        df:               Raw DataFrame for this sample (used for text stats)
        trainer:          Fitted GlobalTrainer
        run_split:        'train' or 'test'
        sample_idx:       Integer sample index (for reference)
        output_dimension: 2 for M5, 1 for M6

    Returns:
        Dict with all RESULT_COLUMNS keys plus 'Predicted_Value' and 'True_Value'
    """
    M, N, E, T = variant['M'], variant['N'], variant['E'], variant['T']
    task_type   = trainer.task_type
    nan         = float('nan')

    # --- Text stats (reuse trainer.py helper) ---
    avg_text_length, text_vocab_entropy = _text_stats(df, T)

    # --- Label balance entropy — undefined for single-label-per-graph ---
    label_balance_ent = nan

    # --- Degenerate flag ---
    degenerate = trainer.is_degenerate(window=20, threshold=0.01)

    if task_type == 'global_categorical':
        # M5: per-graph accuracy (0 or 1)
        accuracy         = float(graph_result['correct'])
        predicted_value  = float(graph_result['pred_class'])
        true_value       = float(graph_result['true_class'])
        primary_metric   = 'accuracy'
        primary_value    = accuracy
        normalized_score = accuracy * 100.0

        row = {
            'Task_Idx':   M, 'Node_Idx': N, 'Edge_Idx': E, 'Text_Idx': T,
            'task_type':  task_type,
            'KL':         nan, 'Top1': nan, 'Top3': nan, 'Cosine': nan,
            'Accuracy':   accuracy,
            'F1':         nan,
            'MAE':        nan, 'MSE': nan, 'R2': nan,
            'Primary_Metric':        primary_metric,
            'Primary_Value':         primary_value,
            'Performance_Band':      None,
            'normalized_score':      normalized_score,
            'number_of_nodes':       graph.num_nodes,
            'avg_text_length':       avg_text_length,
            'text_vocab_entropy':    text_vocab_entropy,
            'label_balance_entropy': label_balance_ent,
            'output_dimension':      output_dimension,
            'run_split':             run_split,
            'degenerate':            degenerate,
            # Extra columns
            'Predicted_Value':       predicted_value,
            'True_Value':            true_value,
        }

    else:
        # M6: per-graph MAE / MSE; R2 is aggregate (NaN here, computed externally)
        mae = graph_result['mae']
        mse = graph_result['mse']
        predicted_value = graph_result['predicted_value']
        true_value      = graph_result['true_value']

        row = {
            'Task_Idx':   M, 'Node_Idx': N, 'Edge_Idx': E, 'Text_Idx': T,
            'task_type':  task_type,
            'KL':         nan, 'Top1': nan, 'Top3': nan, 'Cosine': nan,
            'Accuracy':   nan,
            'F1':         nan,
            'MAE':        mae,
            'MSE':        mse,
            'R2':         nan,   # aggregate — computed across all graphs in Part 2
            'Primary_Metric':        'r2',
            'Primary_Value':         nan,   # not available per-row
            'Performance_Band':      None,
            'normalized_score':      nan,   # R2 not available per-row
            'number_of_nodes':       graph.num_nodes,
            'avg_text_length':       avg_text_length,
            'text_vocab_entropy':    text_vocab_entropy,
            'label_balance_entropy': label_balance_ent,
            'output_dimension':      output_dimension,
            'run_split':             run_split,
            'degenerate':            degenerate,
            # Extra columns
            'Predicted_Value':       predicted_value,
            'True_Value':            true_value,
        }

    return row
