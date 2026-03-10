"""
Execution Engine Module for Ontological Generalization Framework
Handles GNN training, evaluation, and performance tracking.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import KLDivLoss
from sklearn.metrics import top_k_accuracy_score
from typing import Dict, Optional, Tuple
from tqdm import tqdm


class GNNTrainer:
    """
    Execution Engine: Trains GNN models on TAG objects.
    Tracks performance metrics and manages training lifecycle.
    """
    
    def __init__(self, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 lr: float = 0.001, weight_decay: float = 5e-4, 
                 epochs: int = 200, patience: int = 50):
        """
        Initialize trainer.
        
        Args:
            model: GNN model instance
            device: Device to train on
            lr: Learning rate
            weight_decay: L2 regularization strength
            epochs: Number of training epochs
            patience: Early stopping patience
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.criterion = KLDivLoss(reduction='batchmean')
        
        self.training_history = []
        self.best_val_kl = float('inf')
        self.best_state = None
        
    def train_one_epoch(self, data) -> float:
        """
        Train for one epoch.
        
        Args:
            data: PyTorch Geometric Data object
        
        Returns:
            Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data.x, data.edge_index)
        
        # Compute loss only on training nodes with targets
        mask = data.train_mask
        target = data.y[mask]
        pred = out[mask]
        
        loss = self.criterion(pred, target)  # KLDiv between log_prob and target prob
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, data, mask, num_classes: int) -> Dict[str, float]:
        """
        Evaluate model on a data split.
        
        Args:
            data: PyTorch Geometric Data object
            mask: Boolean mask for the split
            num_classes: Number of classes
        
        Returns:
            Dictionary of metrics (kl, top1, top3, cosine)
        """
        self.model.eval()
        out = self.model(data.x, data.edge_index)  # log-probs
        
        probs = out.exp().cpu().numpy()
        targets = data.y.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        if mask_np.sum() == 0:
            return {
                "kl": float('nan'), 
                "top1": float('nan'), 
                "top3": float('nan'), 
                "cosine": float('nan')
            }
        
        pred = probs[mask_np]
        targ = targets[mask_np]
        
        # KL divergence
        kl = float(np.sum(targ * (np.log(np.maximum(targ, 1e-12)) - 
                                    np.log(np.maximum(pred, 1e-12)))) / len(pred))
        
        # Top-1 and Top-3 accuracy
        y_true_idx = targ.argmax(axis=1)
        y_pred_idx = pred.argmax(axis=1)
        top1 = (y_pred_idx == y_true_idx).mean()
        
        try:
            top3 = top_k_accuracy_score(y_true_idx, pred, k=min(3, num_classes), 
                                        labels=np.arange(num_classes))
        except:
            top3 = float('nan')
        
        # Cosine similarity between pred and true vectors
        dot = (pred * targ).sum(axis=1)
        norm_pred = np.linalg.norm(pred, axis=1)
        norm_targ = np.linalg.norm(targ, axis=1)
        cosines = dot / (norm_pred * norm_targ + 1e-12)
        cosine_mean = float(np.nanmean(cosines))
        
        return {
            "kl": kl, 
            "top1": top1, 
            "top3": top3, 
            "cosine": cosine_mean
        }
    
    def train(self, data, num_classes: int, verbose: bool = True, early_stopping: bool = False) -> Dict[str, float]:
        """
        Full training loop.
        
        Args:
            data: PyTorch Geometric Data object
            num_classes: Number of classes
            verbose: Print progress
            early_stopping: Enable early stopping (default False for comparable results)
        
        Returns:
            Test metrics
        """
        print(f"\n{'='*60}")
        print(f"Training on device: {self.device}")
        print(f"Epochs: {self.epochs}, LR: {self.lr}, Weight Decay: {self.weight_decay}")
        print(f"{'='*60}\n")
        
        no_improve_count = 0
        
        for epoch in range(1, self.epochs + 1):
            loss = self.train_one_epoch(data)
            
            train_metrics = self.evaluate(data, data.train_mask, num_classes)
            val_metrics = self.evaluate(data, data.val_mask, num_classes)
            
            # Save history
            self.training_history.append({
                'epoch': epoch,
                'loss': loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
            
            # Track best state
            if val_metrics["kl"] < self.best_val_kl:
                self.best_val_kl = val_metrics["kl"]
                self.best_state = self.model.state_dict()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch:03d} | "
                    f"Loss {loss:.4f} | "
                    f"Train KL {train_metrics['kl']:.4f} | "
                    f"Val KL {val_metrics['kl']:.4f} | "
                    f"Val top1 {val_metrics['top1']:.3f} | "
                    f"Val cos {val_metrics['cosine']:.3f}")
            
            # Early stopping (disabled by default)
            if early_stopping and no_improve_count >= self.patience:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break
        
        # Load best state
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        # Evaluate on test set
        test_metrics = self.evaluate(data, data.test_mask, num_classes)
        
        print(f"\n{'='*60}")
        print(f"Test Results:")
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
                       save_training_history: bool = False) -> Dict:
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
        
        # Convert TAG to PyG Data
        data = tag.to_pyg_data(node_list, embeddings, labels, device=self.device)
        
        # Create model
        feature_dim = embeddings.shape[1]
        model = self.model_factory(
            in_dim=feature_dim,
            out_dim=num_classes,
            **model_config
        )
        
        # Create trainer
        trainer = GNNTrainer(model, device=self.device, **trainer_config)
        
        # Train
        test_metrics = trainer.train(data, num_classes)
        
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
        
        metrics = {
            'kl': [r['test_metrics']['kl'] for r in self.results],
            'top1': [r['test_metrics']['top1'] for r in self.results],
            'top3': [r['test_metrics']['top3'] for r in self.results],
            'cosine': [r['test_metrics']['cosine'] for r in self.results]
        }
        
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
