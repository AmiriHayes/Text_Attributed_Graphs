"""
Execution Engine — TAG Research (code/ version)
Trains GNN models on TAG Data objects and produces structured result rows.

Mirrors the legacy root trainer.py structure; adds:
  - 'edge_scalar'  task type (M4)
  - Fixed M3 masking: labels == -1 excluded via boolean mask, NOT ignore_index
  - result_row() adapter that emits the full schema for construction_performance_table
"""

import csv
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import KLDivLoss
from pathlib import Path
from sklearn.metrics import top_k_accuracy_score, f1_score
from typing import Dict, List, Optional, Any
from collections import defaultdict
from tqdm import tqdm


# ---------------------------------------------------------------------------
# GNNTrainer
# ---------------------------------------------------------------------------

class GNNTrainer:
    """
    Trains GNN models on PyG Data objects.

    Supported task_type values:
      'categorical'      — M1: KLDivLoss, kl/top1/top3/cosine, node masks
      'scalar'           — M2: MSELoss, mae/mse/r2, node masks
      'edge_categorical' — M3: CrossEntropyLoss (with -1 mask), accuracy/f1, edge masks
      'edge_scalar'      — M4: MSELoss, mae/mse/r2, edge masks
    """

    def __init__(self, model, device: str = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'),
                 lr: float = 0.001, weight_decay: float = 5e-4,
                 epochs: int = 200, patience: int = 50,
                 task_type: str = 'categorical'):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.task_type = task_type

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        if task_type in ('scalar', 'edge_scalar'):
            self.criterion = torch.nn.MSELoss()
        elif task_type == 'edge_categorical':
            # We do NOT pass ignore_index=-1 here because -1 is not a valid
            # class index for CrossEntropyLoss (it would silently mis-route
            # gradients). Instead we filter with a boolean mask before loss.
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            # categorical (M1)
            self.criterion = KLDivLoss(reduction='batchmean')

        self.training_history: List[Dict] = []
        self.best_val_loss = float('inf')
        self.best_state = None

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def train_one_epoch(self, data) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        mask = data.train_mask

        if self.task_type == 'edge_categorical':
            # target_edge_index[:,mask] selects masked edges for message-passing score
            target_ei = data.target_edge_index[:, mask]
            out = self.model(data.x, data.edge_index, target_edge_index=target_ei)
            labels = data.y[mask]

            # Exclude edges where label == -1 (unknown — see LabelFactory M3)
            valid = labels >= 0
            if valid.sum() == 0:
                return 0.0
            loss = self.criterion(out[valid], labels[valid])

        elif self.task_type == 'edge_scalar':
            target_ei = data.target_edge_index[:, mask]
            out = self.model(data.x, data.edge_index, target_edge_index=target_ei)
            labels = data.y[mask].float().unsqueeze(-1) if data.y[mask].dim() == 1 else data.y[mask].float()
            loss = self.criterion(out, labels)

        elif self.task_type == 'scalar':
            out = self.model(data.x, data.edge_index)
            tgt = data.y[mask].float()
            if tgt.dim() == 1:
                tgt = tgt.unsqueeze(-1)
            loss = self.criterion(out[mask], tgt)

        else:
            # categorical (M1)
            out = self.model(data.x, data.edge_index)
            loss = self.criterion(out[mask], data.y[mask])

        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, data, mask, num_classes: int) -> Dict[str, float]:
        self.model.eval()
        mask_cpu = mask.cpu().numpy()

        if mask_cpu.sum() == 0:
            return self._empty_metrics()

        if self.task_type == 'edge_categorical':
            target_ei = data.target_edge_index[:, mask]
            out = self.model(data.x, data.edge_index, target_edge_index=target_ei)
            labels = data.y[mask].cpu()
            # Exclude -1 labels
            valid = labels >= 0
            if valid.sum() == 0:
                return {'accuracy': float('nan'), 'f1': float('nan')}
            pred_cls = out[valid].argmax(dim=-1).cpu().numpy()
            true_cls = labels[valid].numpy()
            acc = float((pred_cls == true_cls).mean())
            f1 = float(f1_score(true_cls, pred_cls, average='binary', zero_division=0))
            # Soft probs + one-hot labels for pseudo_R2 (Step 1 scoring)
            probs = torch.softmax(out[valid], dim=-1).cpu().numpy()
            one_hot = np.eye(2)[true_cls]
            return {'accuracy': acc, 'f1': f1, '_raw_pred': probs, '_raw_targ': one_hot}

        if self.task_type == 'edge_scalar':
            target_ei = data.target_edge_index[:, mask]
            out = self.model(data.x, data.edge_index, target_edge_index=target_ei)
            pred = out.cpu().numpy().flatten()
            targ = data.y[mask].cpu().numpy().flatten()
            metrics = self._scalar_metrics(pred, targ)
            metrics['_raw_pred'] = pred
            metrics['_raw_targ'] = targ
            return metrics

        out = self.model(data.x, data.edge_index)

        if self.task_type == 'scalar':
            pred = out[mask].cpu().numpy().flatten()
            targ = data.y[mask].cpu().numpy().flatten()
            metrics = self._scalar_metrics(pred, targ)
            # Raw arrays for M6 aggregate-gap derivation (ignored by build_result_row)
            metrics['_raw_pred'] = pred
            metrics['_raw_targ'] = targ
            return metrics

        # categorical (M1)
        probs = out.exp().cpu().numpy()
        targets = data.y.cpu().numpy()
        pred = probs[mask_cpu]
        targ = targets[mask_cpu]
        kl = float(np.sum(targ * (np.log(np.maximum(targ, 1e-12)) -
                                   np.log(np.maximum(pred, 1e-12)))) / max(len(pred), 1))
        y_true_idx = targ.argmax(axis=1)
        y_pred_idx = pred.argmax(axis=1)
        top1 = float((y_pred_idx == y_true_idx).mean())
        try:
            top3 = float(top_k_accuracy_score(
                y_true_idx, pred, k=min(3, num_classes), labels=np.arange(num_classes)
            ))
        except Exception:
            top3 = float('nan')
        dot = (pred * targ).sum(axis=1)
        norm_p = np.linalg.norm(pred, axis=1)
        norm_t = np.linalg.norm(targ, axis=1)
        cosines = dot / (norm_p * norm_t + 1e-12)
        # Raw arrays for M5 aggregate-gap derivation (ignored by build_result_row)
        return {'kl': kl, 'top1': top1, 'top3': top3, 'cosine': float(np.nanmean(cosines)),
                '_raw_pred': pred, '_raw_targ': targ}

    def _scalar_metrics(self, pred: np.ndarray, targ: np.ndarray) -> Dict[str, float]:
        mae = float(np.mean(np.abs(pred - targ)))
        mse = float(np.mean((pred - targ) ** 2))
        ss_res = np.sum((targ - pred) ** 2)
        ss_tot = np.sum((targ - targ.mean()) ** 2)
        # Near-zero ss_tot means labels have no variance (e.g. all centrality ≈ 1
        # in a dense graph). R2 is undefined in that case — return NaN rather than
        # a meaningless extreme value that poisons the normalized_score.
        if ss_tot < 1e-6:
            r2 = float('nan')
        else:
            r2 = float(1 - ss_res / ss_tot)
        return {'mae': mae, 'mse': mse, 'r2': r2}

    def _empty_metrics(self) -> Dict[str, float]:
        if self.task_type == 'scalar':
            return {'mae': float('nan'), 'mse': float('nan'), 'r2': float('nan')}
        if self.task_type in ('edge_categorical',):
            return {'accuracy': float('nan'), 'f1': float('nan')}
        if self.task_type == 'edge_scalar':
            return {'mae': float('nan'), 'mse': float('nan'), 'r2': float('nan')}
        return {'kl': float('nan'), 'top1': float('nan'),
                'top3': float('nan'), 'cosine': float('nan')}

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, data, num_classes: int, verbose: bool = True,
              early_stopping: bool = False,
              epoch_log_path: Optional[str] = None,
              experiment_id: Optional[str] = None) -> Dict[str, float]:
        """
        Full training loop.

        Args:
            data: PyG Data object
            num_classes: Number of output classes (used for top-k metrics)
            verbose: Print progress every 10 epochs
            early_stopping: Enable early stopping
            epoch_log_path: If set, stream per-epoch metrics to this CSV
            experiment_id: String written to epoch log

        Returns:
            Test metrics dict
        """
        data = data.to(self.device)
        no_improve_count = 0

        epoch_log_fields = {
            'categorical': ['experiment_id', 'epoch', 'loss',
                            'train_kl', 'train_top1', 'train_cosine',
                            'val_kl', 'val_top1', 'val_cosine'],
            'scalar':      ['experiment_id', 'epoch', 'loss',
                            'train_mae', 'train_r2', 'val_mae', 'val_r2'],
            'edge_categorical': ['experiment_id', 'epoch', 'loss',
                                 'train_acc', 'train_f1', 'val_acc', 'val_f1'],
            'edge_scalar': ['experiment_id', 'epoch', 'loss',
                            'train_mae', 'train_r2', 'val_mae', 'val_r2'],
        }.get(self.task_type, ['experiment_id', 'epoch', 'loss'])

        for epoch in range(1, self.epochs + 1):
            loss = self.train_one_epoch(data)
            train_metrics = self.evaluate(data, data.train_mask, num_classes)
            val_metrics = self.evaluate(data, data.val_mask, num_classes)

            self.training_history.append({
                'epoch': epoch, 'loss': loss,
                'train_metrics': train_metrics, 'val_metrics': val_metrics,
            })

            # Stream epoch to CSV
            if epoch_log_path is not None:
                log_path = Path(epoch_log_path)
                write_header = not log_path.exists()
                log_path.parent.mkdir(exist_ok=True, parents=True)
                epoch_row = self._epoch_row(experiment_id, epoch, loss,
                                            train_metrics, val_metrics)
                with open(log_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=epoch_log_fields)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(epoch_row)

            # Track best state (lower = better for loss-like metrics)
            val_track = self._val_track(val_metrics)
            if val_track < self.best_val_loss:
                self.best_val_loss = val_track
                self.best_state = self.model.state_dict().copy()
                # Store deep copy
                import copy
                self.best_state = copy.deepcopy(self.model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Verbose logging
            if verbose and (epoch % 10 == 0 or epoch == 1):
                self._print_epoch(epoch, loss, train_metrics, val_metrics)

            if early_stopping and no_improve_count >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        # Restore best state
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        test_metrics = self.evaluate(data, data.test_mask, num_classes)

        if verbose:
            print(f"\n{'='*50}")
            print("Test results:", test_metrics)
            print(f"{'='*50}\n")

        return test_metrics

    def _val_track(self, val_metrics: Dict) -> float:
        """Return a value that should decrease as validation improves."""
        if self.task_type == 'edge_categorical':
            # Higher accuracy = better → negate
            return -val_metrics.get('accuracy', 0.0)
        if self.task_type in ('scalar', 'edge_scalar'):
            return val_metrics.get('mse', float('inf'))
        # categorical
        return val_metrics.get('kl', float('inf'))

    def _epoch_row(self, experiment_id, epoch, loss, train_m, val_m) -> Dict:
        base = {'experiment_id': experiment_id or '', 'epoch': epoch,
                'loss': round(loss, 6)}
        if self.task_type in ('scalar', 'edge_scalar'):
            base.update({
                'train_mae': round(float(train_m.get('mae', float('nan'))), 6),
                'train_r2':  round(float(train_m.get('r2',  float('nan'))), 6),
                'val_mae':   round(float(val_m.get('mae',   float('nan'))), 6),
                'val_r2':    round(float(val_m.get('r2',    float('nan'))), 6),
            })
        elif self.task_type == 'edge_categorical':
            base.update({
                'train_acc': round(float(train_m.get('accuracy', float('nan'))), 6),
                'train_f1':  round(float(train_m.get('f1',       float('nan'))), 6),
                'val_acc':   round(float(val_m.get('accuracy',   float('nan'))), 6),
                'val_f1':    round(float(val_m.get('f1',         float('nan'))), 6),
            })
        else:
            base.update({
                'train_kl':     round(float(train_m.get('kl',     float('nan'))), 6),
                'train_top1':   round(float(train_m.get('top1',   float('nan'))), 6),
                'train_cosine': round(float(train_m.get('cosine', float('nan'))), 6),
                'val_kl':       round(float(val_m.get('kl',       float('nan'))), 6),
                'val_top1':     round(float(val_m.get('top1',     float('nan'))), 6),
                'val_cosine':   round(float(val_m.get('cosine',   float('nan'))), 6),
            })
        return base

    def _print_epoch(self, epoch, loss, train_m, val_m):
        if self.task_type in ('scalar', 'edge_scalar'):
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                  f"Train MAE {train_m.get('mae', float('nan')):.4f} | "
                  f"Val MAE {val_m.get('mae', float('nan')):.4f} | "
                  f"Val R2 {val_m.get('r2', float('nan')):.3f}")
        elif self.task_type == 'edge_categorical':
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                  f"Train Acc {train_m.get('accuracy', float('nan')):.3f} | "
                  f"Val Acc {val_m.get('accuracy', float('nan')):.3f} | "
                  f"Val F1 {val_m.get('f1', float('nan')):.3f}")
        else:
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                  f"Train KL {train_m.get('kl', float('nan')):.4f} | "
                  f"Val KL {val_m.get('kl', float('nan')):.4f} | "
                  f"Val top1 {val_m.get('top1', float('nan')):.3f} | "
                  f"Val cos {val_m.get('cosine', float('nan')):.3f}")

    def is_degenerate(self, window: int = 20, threshold: float = 0.01) -> bool:
        """
        Return True if total loss improvement over the first `window` epochs < threshold.
        Used to flag degenerate runs in result rows.
        """
        if len(self.training_history) < window:
            return False
        first_loss = self.training_history[0]['loss']
        last_loss = self.training_history[window - 1]['loss']
        return abs(first_loss - last_loss) < threshold


# ---------------------------------------------------------------------------
# Result row adapter
# ---------------------------------------------------------------------------

RESULT_COLUMNS = [
    'Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx',
    'task_type',
    'KL', 'Top1', 'Top3', 'Cosine',
    'Accuracy', 'F1',
    'MAE', 'MSE', 'R2',
    'Primary_Metric', 'Primary_Value',
    'Performance_Band',
    'S_GNN_step1', 'S_MLP_step1', 'Final_Score',
    # Three scoring views saved simultaneously for clean ablation comparison:
    #   raw_gnn_score   — S_GNN_step1 directly (no MLP comparison, no clamp)
    #   unclamped_score — (S_GNN-S_MLP)/(1-S_MLP) or Δ/(1+|Δ|), signed, no max(0)
    #   Final_Score     — clamped/clipped version already above
    'raw_gnn_score',
    'unclamped_score',
    'scoring_formula',
    'normalized_score',
    'number_of_nodes',
    'avg_text_length',
    'text_vocab_entropy',
    'label_balance_entropy',
    'output_dimension',
    'run_split',
    'degenerate',
    'nan_reason',
]


def _shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy (nats) of a discrete distribution given raw counts."""
    counts = counts[counts > 0]
    if len(counts) == 0:
        return float('nan')
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _text_stats(df, T: str):
    """
    Return (avg_text_length, text_vocab_entropy) for the text fidelity column
    indicated by T.
      T12a → text_fidelity_a
      T12b → text_fidelity_b
      T12e → (0.0, 0.0)  [empty string by definition]
    """
    if T == 'T12e':
        return 0.0, 0.0

    col = 'text_fidelity_a' if T == 'T12a' else 'text_fidelity_b'
    if col not in df.columns:
        return float('nan'), float('nan')

    texts = df[col].dropna().astype(str)
    if len(texts) == 0:
        return 0.0, 0.0

    word_counts = texts.str.split().str.len()
    avg_len = float(word_counts.mean())

    # Word-token frequency for entropy
    from collections import Counter
    word_freq: Counter = Counter()
    for t in texts:
        word_freq.update(t.lower().split())
    if not word_freq:
        return avg_len, 0.0

    counts = np.array(list(word_freq.values()), dtype=np.float64)
    entropy = _shannon_entropy(counts)
    return avg_len, entropy


def _label_balance_entropy(data, task_type: str) -> float:
    """
    Shannon entropy of the label distribution.
    Categorical (M1, M3): entropy over class counts.
    Scalar (M2, M4): entropy of 20-bin histogram of values.
    """
    y = data.y.cpu().numpy()
    if task_type == 'categorical':
        # y is (N, C) one-hot; sum over nodes per class
        if y.ndim == 2:
            class_counts = y.sum(axis=0)
        else:
            class_counts, _ = np.histogram(y, bins=20)
        return _shannon_entropy(class_counts.astype(np.float64))

    if task_type == 'edge_categorical':
        # y is (E,) integer labels; exclude -1
        labels = y[y >= 0]
        if len(labels) == 0:
            return float('nan')
        counts = np.bincount(labels.astype(int))
        return _shannon_entropy(counts.astype(np.float64))

    # scalar / edge_scalar
    vals = y.flatten()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float('nan')
    counts, _ = np.histogram(vals, bins=20)
    return _shannon_entropy(counts.astype(np.float64))


def build_result_row(
    variant: Dict,
    test_metrics: Dict[str, float],
    data,           # PyG Data object
    df,             # pandas DataFrame (sample)
    trainer: 'GNNTrainer',
    run_split: str,
    sample_idx: int,
    output_dimension: int,
    s_gnn_step1: float = float('nan'),
    s_mlp_step1: float = float('nan'),
    final_score: float = float('nan'),
    r2_for_row: float = float('nan'),
) -> Dict[str, Any]:
    """
    Wrap training output into the full construction_performance_table schema.

    Args:
        variant: {'M': ..., 'N': ..., 'E': ..., 'T': ...}
        test_metrics: dict returned by trainer.train()
        data: PyG Data object produced by TAGConstructor.construct()
        df: DataFrame for the sample (used for text stats)
        trainer: fitted GNNTrainer instance (for degenerate detection)
        run_split: 'train' or 'test'
        sample_idx: integer index of the sample (for reference)
        output_dimension: int (data.y.shape[-1] for node tasks, etc.)

    Returns:
        Ordered dict matching RESULT_COLUMNS
    """
    M, N, E, T = variant['M'], variant['N'], variant['E'], variant['T']
    task_type = trainer.task_type

    nan = float('nan')

    # --- Per-task metric extraction ---
    kl     = test_metrics.get('kl',       nan)
    top1   = test_metrics.get('top1',     nan)
    top3   = test_metrics.get('top3',     nan)
    cosine = test_metrics.get('cosine',   nan)
    acc    = test_metrics.get('accuracy', nan)
    f1     = test_metrics.get('f1',       nan)
    mae    = test_metrics.get('mae',      nan)
    mse    = test_metrics.get('mse',      nan)
    r2     = test_metrics.get('r2',       nan)

    # --- R2 column: use pseudo_R2 for categorical tasks if provided ---
    # For M1/M3 categorical tasks, r2_for_row is the pseudo_R2 (McFadden-style,
    # computed via compute_step1_score in _run_one). For M2/M4 scalar tasks
    # r2_for_row == r2. Stored here so the R2 column is always populated.
    if task_type in ('categorical', 'edge_categorical') and not math.isnan(r2_for_row):
        r2 = r2_for_row

    # --- Primary metric and normalized score (two-step scoring framework) ---
    # DEPRECATED — old single-metric normalization:
    #   categorical: normalized_score = top1 * 100
    #   edge_categorical: normalized_score = acc * 100
    #   scalar/edge_scalar: normalized_score = (clamp(R2,-1,1)+1)/2*100
    #       (the "(-1 floor, 0=mean-prediction → 50)" linear scheme)
    # Replaced by two-step scoring: Step 1 = theoretical normalization
    # (pseudo_R2 / max(0,R2) / TVD-relative / Gap-relative), Step 2 = graph-lift
    # above feature-only MLP baseline. Final_Score drives normalized_score and
    # Primary_Value. See compute_step1_score / compute_final_score in this file.
    primary_metric = 'final_score'
    primary_value  = final_score
    normalized_score = final_score * 100 if not math.isnan(final_score) else nan

    # --- Text stats ---
    avg_text_length, text_vocab_entropy = _text_stats(df, T)

    # --- Label balance entropy ---
    label_balance_ent = _label_balance_entropy(data, task_type)

    # --- Degenerate flag ---
    degenerate = trainer.is_degenerate(window=20, threshold=0.01)
    # ss_tot < 1e-6 guard in _scalar_metrics returns r2=NaN when label variance
    # collapses — training-loss check alone misses this, so propagate here.
    if task_type in ('scalar', 'edge_scalar') and math.isnan(r2):
        degenerate = True
    # Any NaN final_score is a degenerate outcome regardless of task type.
    if math.isnan(final_score):
        degenerate = True

    # unclamped_score: signed delta with no max(0,...) applied.
    # M2 uses algebraic sigmoid Δ/(1+|Δ|) to avoid denominator explosion.
    # All others use (S_GNN-S_MLP)/(1-S_MLP) raw.
    if not math.isnan(s_gnn_step1) and not math.isnan(s_mlp_step1):
        if M in ('M2',):
            _delta = s_gnn_step1 - s_mlp_step1
            _unclamped = _delta / (1.0 + abs(_delta))
        else:
            _denom = 1.0 - s_mlp_step1
            _unclamped = (s_gnn_step1 - s_mlp_step1) / _denom if abs(_denom) >= 1e-9 else float('nan')
    else:
        _unclamped = float('nan')

    row = {
        'Task_Idx': M,
        'Node_Idx': N,
        'Edge_Idx': E,
        'Text_Idx': T,
        'task_type': task_type,
        # Categorical metrics (NaN for non-categorical tasks)
        'KL':     kl     if task_type == 'categorical' else nan,
        'Top1':   top1   if task_type == 'categorical' else nan,
        'Top3':   top3   if task_type == 'categorical' else nan,
        'Cosine': cosine if task_type == 'categorical' else nan,
        # Edge-categorical metrics (NaN for others)
        'Accuracy': acc if task_type == 'edge_categorical' else (top1 if task_type == 'categorical' else nan),
        'F1':       f1  if task_type == 'edge_categorical' else nan,
        # Scalar metrics; R2 = pseudo_R2 for categorical tasks (already overwritten above)
        'MAE': mae if task_type in ('scalar', 'edge_scalar') else nan,
        'MSE': mse if task_type in ('scalar', 'edge_scalar') else nan,
        'R2':  r2,
        # Universal
        'Primary_Metric':       primary_metric,
        'Primary_Value':        primary_value,
        'Performance_Band':     None,
        'S_GNN_step1':          s_gnn_step1,
        'S_MLP_step1':          s_mlp_step1,
        'Final_Score':          final_score,
        'raw_gnn_score':        s_gnn_step1,
        'unclamped_score':      _unclamped,
        'scoring_formula':      'algebraic_lift' if M in ('M2',) else 'graph_lift',
        'normalized_score':     normalized_score,
        'number_of_nodes':      data.num_nodes,
        'avg_text_length':      avg_text_length,
        'text_vocab_entropy':   text_vocab_entropy,
        'label_balance_entropy': label_balance_ent,
        'output_dimension':     output_dimension,
        'run_split':            run_split,
        'degenerate':           degenerate,
        'nan_reason':           '',
    }
    return row


# ---------------------------------------------------------------------------
# M5 / M6 derived-metric helpers
# ---------------------------------------------------------------------------

def compute_aggregate_gap(test_metrics: Dict, task_type: str) -> float:
    """Compute Aggregate_Gap from raw test predictions stored in test_metrics.

    M5 (from M1 / categorical):
        TVD = 0.5 * sum|true_class_fraction[c] - mean_predicted_prob[c]|
        Uses the soft softmax output already computed for KL/top1, NOT argmax.

    M6 (from M2 / scalar):
        abs(median(true_labels) - median(predicted_labels)) over test_mask nodes.
        Median is more robust than mean for skewed scalar labels (e.g. book price, vote counts).

    Returns NaN if _raw_pred/_raw_targ absent (e.g. dense-graph bypass).
    """
    pred = test_metrics.get('_raw_pred')
    targ = test_metrics.get('_raw_targ')
    if pred is None or targ is None or len(pred) == 0:
        return float('nan')

    if task_type == 'categorical':
        # targ: (n_test, n_classes) one-hot; pred: (n_test, n_classes) soft probs
        true_dist = targ.mean(axis=0)   # empirical class fractions
        pred_dist = pred.mean(axis=0)   # mean predicted probability per class
        return float(0.5 * np.sum(np.abs(true_dist - pred_dist)))

    if task_type == 'scalar':
        # targ/pred: (n_test,) flat arrays
        # Median is more robust than mean for skewed scalar labels (e.g. price, vote counts).
        return float(abs(float(np.median(targ)) - float(np.median(pred))))

    return float('nan')


# ---------------------------------------------------------------------------
# Two-step scoring framework
# ---------------------------------------------------------------------------

def compute_pseudo_r2(test_metrics: Dict) -> float:
    """McFadden-style pseudo-R² for categorical tasks (M1 / M3).

    pseudo_R2 = 1 - LL_model / LL_null
      LL_model = sum_test log(p_model(true_class)) — uses soft softmax probs
      LL_null  = sum_test log(p_null(true_class)) where p_null(c) is the
                 empirical class frequency WITHIN THIS SAMPLE'S TEST_MASK.

    Captures confidence/calibration via softmax probabilities; a model that
    confidently predicts the wrong class gets a worse score than one that
    hedges. Use max(0, pseudo_R2) as the Step 1 score S.
    """
    pred = test_metrics.get('_raw_pred')   # (n_test, n_classes) soft probs
    targ = test_metrics.get('_raw_targ')   # (n_test, n_classes) one-hot
    if pred is None or targ is None or len(pred) == 0:
        return float('nan')
    eps = 1e-12
    ll_model = float(np.sum(targ * np.log(np.maximum(pred, eps))))
    p_null   = targ.mean(axis=0)           # empirical class freq in test_mask
    ll_null  = float(np.sum(targ * np.log(np.maximum(p_null[None, :], eps))))
    if abs(ll_null) < eps:
        return float('nan')
    return float(1.0 - ll_model / ll_null)


def compute_step1_score(test_metrics: Dict, task_type: str) -> tuple:
    """Compute Step 1 (theoretical normalization) score for a GNN or MLP run.

    Returns (s_step1, r2_or_pseudo_r2) where:
      categorical / edge_categorical: s = max(0, pseudo_R2); r2 = pseudo_R2
      scalar      / edge_scalar:      s = max(0, R2);        r2 = R2
    The second return value is what goes in the R2 column.
    """
    nan = float('nan')
    if task_type in ('categorical', 'edge_categorical'):
        psr2 = compute_pseudo_r2(test_metrics)
        s = nan if math.isnan(psr2) else max(0.0, psr2)
        return s, psr2
    else:
        r2 = test_metrics.get('r2', nan)
        s  = nan if math.isnan(r2) else max(0.0, r2)
        return s, r2


def compute_final_score(s_gnn: float, s_mlp: float) -> float:
    """Step 2 (empirical equalization): graph lift above MLP baseline.

    Final_Score = max(0, (S_GNN - S_MLP) / (1 - S_MLP))

    If S_MLP ≥ 1 (MLP already perfect) or either input is NaN, returns NaN.
    Applied to M1, M3, M4, M5 only. M2 and M6 use compute_final_score_algebraic.
    """
    if math.isnan(s_gnn) or math.isnan(s_mlp):
        return float('nan')
    denom = 1.0 - s_mlp
    if abs(denom) < 1e-9:
        return float('nan')
    return float(max(0.0, (s_gnn - s_mlp) / denom))


def compute_final_score_algebraic(s_gnn: float, s_mlp: float) -> float:
    """Algebraic sigmoid lift for M2 and M6.

    score = Δ / (1 + |Δ|)   where Δ = S_GNN - S_MLP

    Range: (-1, 1). Monotone in Δ. Zero crossing at Δ=0 (GNN == MLP).
    Avoids the (1 - S_MLP) denominator that explodes when MLP accuracy is
    near 1. Preserves ordinal ranking across the full observed range:
      Δ=-40 → -0.976,  Δ=-5 → -0.833,  Δ=0 → 0,  Δ=+5 → +0.833.
    """
    if math.isnan(s_gnn) or math.isnan(s_mlp):
        return float('nan')
    delta = s_gnn - s_mlp
    return float(delta / (1.0 + abs(delta)))


def compute_m5_step1_score(tvd_model: float, data) -> tuple:
    """Step 1 score for M5 (derived from M1).

    S_M5 = max(0, (TVD_baseline - TVD_model) / TVD_baseline)
    TVD_baseline = TVD between true test-mask class distribution and true
                  train-mask class distribution (dummy model predicting
                  training-set priors).

    Returns (s_step1, tvd_baseline). Returns (NaN, NaN) if baseline ≈ 0.
    """
    nan = float('nan')
    if math.isnan(tvd_model):
        return nan, nan
    y          = data.y.cpu().numpy()           # (N, n_classes) one-hot
    test_mask  = data.test_mask.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    test_dist  = y[test_mask].mean(axis=0)      # true class freq in test
    train_dist = y[train_mask].mean(axis=0)     # training-set class priors
    tvd_baseline = float(0.5 * np.sum(np.abs(test_dist - train_dist)))
    # Guard mirrors the ss_tot < 1e-6 pattern in _scalar_metrics: if train and
    # test class distributions are already nearly identical, the null baseline
    # is trivially well-calibrated and the relative-improvement denominator is
    # undefined. Return NaN explicitly so downstream degenerate=True is set with
    # an interpretable cause ("baseline indistinguishable") rather than a silent
    # divide-by-zero.
    if tvd_baseline < 1e-6:
        return nan, tvd_baseline
    s = max(0.0, (tvd_baseline - tvd_model) / tvd_baseline)
    return float(s), tvd_baseline


def compute_m6_step1_score(gap_model: float, data) -> tuple:
    """Step 1 score for M6 (derived from M2).

    S_M6 = max(0, (Gap_baseline - Gap_model) / Gap_baseline)
    Gap_baseline = |true_test_median - true_train_median| (dummy model
                  predicting the training-set median).

    Returns (s_step1, gap_baseline). Returns (NaN, NaN) if baseline ≈ 0.
    """
    nan = float('nan')
    if math.isnan(gap_model):
        return nan, nan
    y          = data.y.cpu().numpy().flatten()
    test_mask  = data.test_mask.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    test_med   = float(np.median(y[test_mask]))
    train_med  = float(np.median(y[train_mask]))
    gap_baseline = float(abs(test_med - train_med))
    # Guard mirrors the ss_tot < 1e-6 pattern in _scalar_metrics: if train and
    # test target medians are already nearly identical, the null baseline is
    # trivially well-calibrated (e.g. ArXiv centrality has uniform distribution
    # across train/test masks within a sample). Return NaN explicitly so
    # degenerate=True is set with an interpretable cause rather than a silent
    # divide-by-zero. Threshold matches ss_tot guard at 1e-6.
    if gap_baseline < 1e-6:
        return nan, gap_baseline
    s = max(0.0, (gap_baseline - gap_model) / gap_baseline)
    return float(s), gap_baseline


def build_derived_global_row(
    source_variant:   Dict,
    aggregate_gap:    float,
    run_split:        str,
    sample_idx:       int,
    data,                      # PyG Data object (for num_nodes and masks)
    df,                        # DataFrame (for text stats)
    output_dimension: int,
    s_gnn_step1: float = float('nan'),
    s_mlp_step1: float = float('nan'),
    final_score: float = float('nan'),
) -> Dict:
    """Build an M5 (from M1) or M6 (from M2) derived result row.

    The row shares (N, E, T, run_split, sample_idx) with the source M1/M2 row.
    Primary_Value = Final_Score; aggregate_gap stored in R2 column for transparency.
    """
    nan   = float('nan')
    M_src = source_variant['M']
    assert M_src in ('M1', 'M2'), f"build_derived_global_row: unexpected source task {M_src}"
    M_der  = 'M5' if M_src == 'M1' else 'M6'
    t_type = 'global_categorical' if M_src == 'M1' else 'global_scalar'

    avg_text_length, text_vocab_entropy = _text_stats(df, source_variant['T'])
    normalized_score = final_score * 100 if not math.isnan(final_score) else nan

    # unclamped_score for derived rows — same formula as build_result_row
    if not math.isnan(s_gnn_step1) and not math.isnan(s_mlp_step1):
        if M_src == 'M2':   # M6 derived
            _d = s_gnn_step1 - s_mlp_step1
            _unc = _d / (1.0 + abs(_d))
        else:               # M5 derived
            _den = 1.0 - s_mlp_step1
            _unc = (s_gnn_step1 - s_mlp_step1) / _den if abs(_den) >= 1e-9 else float('nan')
    else:
        _unc = float('nan')

    return {
        'Task_Idx':              M_der,
        'Node_Idx':              source_variant['N'],
        'Edge_Idx':              source_variant['E'],
        'Text_Idx':              source_variant['T'],
        'task_type':             t_type,
        'KL':    nan, 'Top1':  nan, 'Top3':   nan, 'Cosine': nan,
        'Accuracy': nan, 'F1': nan,
        'MAE':   nan, 'MSE':   nan,
        'R2':    aggregate_gap,
        'Primary_Metric':        'final_score',
        'Primary_Value':         final_score,
        'Performance_Band':      None,
        'S_GNN_step1':           s_gnn_step1,
        'S_MLP_step1':           s_mlp_step1,
        'Final_Score':           final_score,
        'raw_gnn_score':         s_gnn_step1,
        'unclamped_score':       _unc,
        'scoring_formula':       'algebraic_lift' if M_src == 'M2' else 'graph_lift',
        'normalized_score':      normalized_score,
        'number_of_nodes':       data.num_nodes,
        'avg_text_length':       avg_text_length,
        'text_vocab_entropy':    text_vocab_entropy,
        'label_balance_entropy': nan,
        'output_dimension':      output_dimension,
        'run_split':             run_split,
        'degenerate':            math.isnan(final_score),
        'nan_reason':            '',
        'Predicted_Value':       nan,
        'True_Value':            nan,
    }
