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

    def __init__(self, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
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
            return {'accuracy': acc, 'f1': f1}

        if self.task_type == 'edge_scalar':
            target_ei = data.target_edge_index[:, mask]
            out = self.model(data.x, data.edge_index, target_edge_index=target_ei)
            pred = out.cpu().numpy().flatten()
            targ = data.y[mask].cpu().numpy().flatten()
            return self._scalar_metrics(pred, targ)

        out = self.model(data.x, data.edge_index)

        if self.task_type == 'scalar':
            pred = out[mask].cpu().numpy().flatten()
            targ = data.y[mask].cpu().numpy().flatten()
            return self._scalar_metrics(pred, targ)

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
        return {'kl': kl, 'top1': top1, 'top3': top3, 'cosine': float(np.nanmean(cosines))}

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
    'normalized_score',
    'number_of_nodes',
    'avg_text_length',
    'text_vocab_entropy',
    'label_balance_entropy',
    'output_dimension',
    'run_split',
    'degenerate',
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

    # --- Primary metric and normalized score ---
    if task_type == 'categorical':
        primary_metric = 'accuracy'
        primary_value = top1
        normalized_score = top1 * 100 if not math.isnan(top1) else nan
    elif task_type == 'edge_categorical':
        primary_metric = 'accuracy'
        primary_value = acc
        normalized_score = acc * 100 if not math.isnan(acc) else nan
    else:
        # scalar, edge_scalar
        primary_metric = 'r2'
        primary_value = r2
        if math.isnan(r2):
            normalized_score = nan
        else:
            # Map R2 from [-1, 1] → [0, 100]. R2=-1 (twice mean-prediction error)
            # is the floor; anything below is clamped. R2=0 (mean prediction) → 50.
            normalized_score = (max(-1.0, min(1.0, r2)) + 1.0) / 2.0 * 100

    # --- Text stats ---
    avg_text_length, text_vocab_entropy = _text_stats(df, T)

    # --- Label balance entropy ---
    label_balance_ent = _label_balance_entropy(data, task_type)

    # --- Degenerate flag ---
    degenerate = trainer.is_degenerate(window=20, threshold=0.01)

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
        # Scalar metrics (NaN for categorical tasks)
        'MAE': mae if task_type in ('scalar', 'edge_scalar') else nan,
        'MSE': mse if task_type in ('scalar', 'edge_scalar') else nan,
        'R2':  r2  if task_type in ('scalar', 'edge_scalar') else nan,
        # Universal
        'Primary_Metric':       primary_metric,
        'Primary_Value':        primary_value,
        'Performance_Band':     None,   # Thresholds TBD
        'normalized_score':     normalized_score,
        'number_of_nodes':      data.num_nodes,
        'avg_text_length':      avg_text_length,
        'text_vocab_entropy':   text_vocab_entropy,
        'label_balance_entropy': label_balance_ent,
        'output_dimension':     output_dimension,
        'run_split':            run_split,
        'degenerate':           degenerate,
    }
    return row
