"""
Execution Engine Module for Ontological Generalization Framework
Handles GNN training, evaluation, and performance tracking.
"""

import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import KLDivLoss
from pathlib import Path
from sklearn.metrics import top_k_accuracy_score
from typing import Dict, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm


class GNNTrainer:
    """
    Execution Engine: Trains GNN models on TAG objects.
    Tracks performance metrics and manages training lifecycle.
    """
    
    def __init__(self, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 lr: float = 0.001, weight_decay: float = 5e-4,
                 epochs: int = 200, patience: int = 50,
                 task_type: str = 'categorical'):
        """
        Initialize trainer.

        Args:
            model: GNN model instance
            device: Device to train on
            lr: Learning rate
            weight_decay: L2 regularization strength
            epochs: Number of training epochs
            patience: Early stopping patience
            task_type: 'categorical' (KLDiv + top1/top3/cosine)
                       or 'scalar' (MSE + MAE/R²)
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.task_type = task_type

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        if task_type == 'scalar':
            self.criterion = torch.nn.MSELoss()
        elif task_type == 'edge_categorical':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = KLDivLoss(reduction='batchmean')

        self.training_history = []
        self.best_val_loss = float('inf')
        self.best_state = None
        
    def train_one_epoch(self, data) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        mask = data.train_mask

        if self.task_type == 'edge_categorical':
            # model.forward takes target_edge_index to score specific edges
            target_ei = data.target_edge_index[:, mask]
            out = self.model(data.x, data.edge_index, target_edge_index=target_ei)
            loss = self.criterion(out, data.y[mask])
        else:
            out = self.model(data.x, data.edge_index)
            loss = self.criterion(out[mask], data.y[mask])

        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, data, mask, num_classes: int) -> Dict[str, float]:
        self.model.eval()
        mask_np = mask.cpu().numpy()

        if mask_np.sum() == 0:
            if self.task_type == 'scalar':
                return {'mae': float('nan'), 'mse': float('nan'), 'r2': float('nan')}
            if self.task_type == 'edge_categorical':
                return {'accuracy': float('nan'), 'f1': float('nan')}
            return {'kl': float('nan'), 'top1': float('nan'),
                    'top3': float('nan'), 'cosine': float('nan')}

        if self.task_type == 'edge_categorical':
            from sklearn.metrics import f1_score
            target_ei = data.target_edge_index[:, mask]
            out = self.model(data.x, data.edge_index, target_edge_index=target_ei)
            pred_cls = out.argmax(dim=-1).cpu().numpy()
            true_cls = data.y[mask].cpu().numpy()
            acc = float((pred_cls == true_cls).mean())
            f1 = float(f1_score(true_cls, pred_cls, average='binary', zero_division=0))
            return {'accuracy': acc, 'f1': f1}

        out = self.model(data.x, data.edge_index)

        if self.task_type == 'scalar':
            pred = out[mask].cpu().numpy().flatten()
            targ = data.y[mask].cpu().numpy().flatten()
            mae = float(np.mean(np.abs(pred - targ)))
            mse = float(np.mean((pred - targ) ** 2))
            ss_res = np.sum((targ - pred) ** 2)
            ss_tot = np.sum((targ - targ.mean()) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-12))
            return {'mae': mae, 'mse': mse, 'r2': r2}

        # Categorical (node-level)
        probs = out.exp().cpu().numpy()
        targets = data.y.cpu().numpy()
        pred = probs[mask_np]
        targ = targets[mask_np]
        kl = float(np.sum(targ * (np.log(np.maximum(targ, 1e-12)) -
                                   np.log(np.maximum(pred, 1e-12)))) / len(pred))
        y_true_idx = targ.argmax(axis=1)
        y_pred_idx = pred.argmax(axis=1)
        top1 = float((y_pred_idx == y_true_idx).mean())
        try:
            top3 = top_k_accuracy_score(y_true_idx, pred, k=min(3, num_classes),
                                        labels=np.arange(num_classes))
        except Exception:
            top3 = float('nan')
        dot = (pred * targ).sum(axis=1)
        norm_pred = np.linalg.norm(pred, axis=1)
        norm_targ = np.linalg.norm(targ, axis=1)
        cosines = dot / (norm_pred * norm_targ + 1e-12)
        return {'kl': kl, 'top1': top1, 'top3': top3,
                'cosine': float(np.nanmean(cosines))}
    
    def train(self, data, num_classes: int, verbose: bool = True,
              early_stopping: bool = False,
              epoch_log_path: Optional[str] = None,
              experiment_id: Optional[str] = None) -> Dict[str, float]:
        """
        Full training loop.

        Args:
            data: PyTorch Geometric Data object
            num_classes: Number of classes
            verbose: Print progress
            early_stopping: Enable early stopping (default False for comparable results)
            epoch_log_path: If set, stream each epoch's metrics to this CSV file
            experiment_id: Identifier string written to epoch log (e.g. tag identifier)

        Returns:
            Test metrics
        """
        print(f"\n{'='*60}")
        print(f"Training on device: {self.device}")
        print(f"Epochs: {self.epochs}, LR: {self.lr}, Weight Decay: {self.weight_decay}")
        print(f"{'='*60}\n")

        no_improve_count = 0
        epoch_log_fields = {
            'categorical': ['experiment_id', 'epoch', 'loss',
                            'train_kl', 'train_top1', 'train_cosine',
                            'val_kl', 'val_top1', 'val_cosine'],
            'scalar':      ['experiment_id', 'epoch', 'loss',
                            'train_mae', 'train_r2', 'val_mae', 'val_r2'],
            'edge_categorical': ['experiment_id', 'epoch', 'loss',
                                 'train_acc', 'train_f1', 'val_acc', 'val_f1'],
        }.get(self.task_type, ['experiment_id', 'epoch', 'loss'])

        for epoch in range(1, self.epochs + 1):
            loss = self.train_one_epoch(data)
            train_metrics = self.evaluate(data, data.train_mask, num_classes)
            val_metrics = self.evaluate(data, data.val_mask, num_classes)

            self.training_history.append({
                'epoch': epoch, 'loss': loss,
                'train_metrics': train_metrics, 'val_metrics': val_metrics
            })

            # Stream epoch to CSV if requested
            if epoch_log_path is not None:
                log_path = Path(epoch_log_path)
                write_header = not log_path.exists()
                log_path.parent.mkdir(exist_ok=True, parents=True)
                if self.task_type == 'scalar':
                    epoch_row = {
                        'experiment_id': experiment_id or '', 'epoch': epoch,
                        'loss': round(loss, 6),
                        'train_mae': round(float(train_metrics['mae']), 6),
                        'train_r2':  round(float(train_metrics['r2']), 6),
                        'val_mae':   round(float(val_metrics['mae']), 6),
                        'val_r2':    round(float(val_metrics['r2']), 6),
                    }
                elif self.task_type == 'edge_categorical':
                    epoch_row = {
                        'experiment_id': experiment_id or '', 'epoch': epoch,
                        'loss': round(loss, 6),
                        'train_acc': round(float(train_metrics['accuracy']), 6),
                        'train_f1':  round(float(train_metrics['f1']), 6),
                        'val_acc':   round(float(val_metrics['accuracy']), 6),
                        'val_f1':    round(float(val_metrics['f1']), 6),
                    }
                else:
                    epoch_row = {
                        'experiment_id': experiment_id or '', 'epoch': epoch,
                        'loss': round(loss, 6),
                        'train_kl':     round(float(train_metrics['kl']), 6),
                        'train_top1':   round(float(train_metrics['top1']), 6),
                        'train_cosine': round(float(train_metrics['cosine']), 6),
                        'val_kl':       round(float(val_metrics['kl']), 6),
                        'val_top1':     round(float(val_metrics['top1']), 6),
                        'val_cosine':   round(float(val_metrics['cosine']), 6),
                    }
                with open(log_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=epoch_log_fields)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(epoch_row)

            # Track best state
            # For categorical/edge: lower loss = better. For edge use negative acc.
            if self.task_type == 'edge_categorical':
                val_track = -val_metrics.get('accuracy', 0.0)
            else:
                val_track = val_metrics.get('mse', val_metrics.get('kl', float('inf')))
            if val_track < self.best_val_loss:
                self.best_val_loss = val_track
                self.best_state = self.model.state_dict()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == 1):
                if self.task_type == 'scalar':
                    print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                          f"Train MAE {train_metrics['mae']:.4f} | "
                          f"Val MAE {val_metrics['mae']:.4f} | "
                          f"Val R² {val_metrics['r2']:.3f}")
                elif self.task_type == 'edge_categorical':
                    print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                          f"Train Acc {train_metrics['accuracy']:.3f} | "
                          f"Val Acc {val_metrics['accuracy']:.3f} | "
                          f"Val F1 {val_metrics['f1']:.3f}")
                else:
                    print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                          f"Train KL {train_metrics['kl']:.4f} | "
                          f"Val KL {val_metrics['kl']:.4f} | "
                          f"Val top1 {val_metrics['top1']:.3f} | "
                          f"Val cos {val_metrics['cosine']:.3f}")

            if early_stopping and no_improve_count >= self.patience:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break
        
        # Load best state
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        test_metrics = self.evaluate(data, data.test_mask, num_classes)

        print(f"\n{'='*60}")
        print(f"Test Results:")
        if self.task_type == 'scalar':
            print(f"  MAE:  {test_metrics['mae']:.4f}")
            print(f"  MSE:  {test_metrics['mse']:.4f}")
            print(f"  R²:   {test_metrics['r2']:.4f}")
        elif self.task_type == 'edge_categorical':
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  F1:       {test_metrics['f1']:.4f}")
        else:
            print(f"  KL Divergence: {test_metrics['kl']:.4f}")
            print(f"  Top-1 Accuracy: {test_metrics['top1']:.3f}")
            print(f"  Top-3 Accuracy: {test_metrics['top3']:.3f}")
            print(f"  Cosine Similarity: {test_metrics['cosine']:.3f}")
        print(f"{'='*60}\n")

        return test_metrics
    
    def save_model(self, filepath: str):
        """Save model state dict."""
        torch.save(self.model.state_dict(), filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"✅ Model loaded from {filepath}")


class TrainingPipeline:
    """
    High-level training pipeline for TAG experiments.
    Manages the complete training workflow.
    """
    
    def __init__(self, model_factory, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize training pipeline.
        
        Args:
            model_factory: Function to create model instances
            device: Device to train on
        """
        self.model_factory = model_factory
        self.device = device
        
    def run_experiment(self, tag, embeddings: np.ndarray, labels: np.ndarray,
                       node_list: list, num_classes: int,
                       model_config: Dict, trainer_config: Dict,
                       save_model_path: Optional[str] = None,
                       save_training_history: bool = False,
                       epoch_log_path: Optional[str] = None) -> Dict:
        """
        Run a complete training experiment on a TAG.
        
        Args:
            tag: TAG object
            embeddings: Node embeddings
            labels: Node labels
            node_list: List of node identifiers
            num_classes: Number of classes
            model_config: Model configuration dict
            trainer_config: Trainer configuration dict
            save_model_path: Path to save trained model
        
        Returns:
            Dictionary with test metrics and metadata
        """
        print(f"\n{'#'*60}")
        print(f"Experiment: {tag.get_identifier()}")
        print(f"{'#'*60}")

        task_type = getattr(tag, 'task_type', 'categorical')
        feature_dim = embeddings.shape[1]

        if task_type == 'edge_categorical':
            # labels is (edge_list, edge_label_array) tuple for edge tasks
            data = tag.to_pyg_data_edge(
                node_list, embeddings,
                edge_list=labels[0],
                edge_labels=labels[1],
                device=self.device
            )
            out_dim = 2  # binary: same / cross category
            model_type = 'sage_edge'
        else:
            data = tag.to_pyg_data(node_list, embeddings, labels, device=self.device)
            out_dim = 1 if task_type == 'scalar' else num_classes
            model_type = None  # use default from model_factory

        # Pass task_type to factory so ModelFactory can route to sage_edge
        factory_kwargs = dict(model_config)
        factory_kwargs['task_type'] = task_type
        model = self.model_factory(
            in_dim=feature_dim,
            out_dim=out_dim,
            **factory_kwargs
        )

        trainer = GNNTrainer(model, device=self.device, task_type=task_type, **trainer_config)
        
        # Train
        test_metrics = trainer.train(
            data, num_classes,
            epoch_log_path=epoch_log_path,
            experiment_id=tag.get_identifier()
        )
        
        # Save model if requested
        if save_model_path:
            trainer.save_model(save_model_path)
        
        # Prepare results
        results = {
            'tag_id': tag.get_identifier(),
            'task_idx': tag.task_idx,
            'node_idx': tag.node_idx,
            'edge_idx': tag.edge_idx,
            'text_idx': tag.text_idx,
            'task_type': task_type,
            'test_metrics': test_metrics,
            'metadata': tag.metadata,
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.shape[1] // 2,
            'feature_dim': feature_dim,
            'num_classes': num_classes
        }
        
        # Optionally include training history
        if save_training_history:
            results['training_history'] = trainer.training_history
        
        return results


class PerformanceTracker:
    """
    Tracks performance across multiple experiments.
    """
    
    def __init__(self):
        self.results = []
    
    def add_result(self, result: Dict):
        """Add experiment result."""
        self.results.append(result)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.results:
            return {}

        # Collect only metrics that actually exist across mixed task types.
        metrics = defaultdict(list)
        for result in self.results:
            for metric_name, metric_value in result.get('test_metrics', {}).items():
                metrics[metric_name].append(metric_value)
        
        summary = {}
        for metric_name, values in metrics.items():
            summary[f'{metric_name}_mean'] = np.nanmean(values)
            summary[f'{metric_name}_std'] = np.nanstd(values)
            summary[f'{metric_name}_min'] = np.nanmin(values)
            summary[f'{metric_name}_max'] = np.nanmax(values)
        
        return summary
    
    def save_results(self, filepath: str):
        """Save all results to JSON."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Results saved to {filepath}")
