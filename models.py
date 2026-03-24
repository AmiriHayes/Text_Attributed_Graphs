"""
GNN Models Module for Ontological Generalization Framework
Defines Graph Neural Network architectures for TAG-based learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv


class SAGEPredictor(nn.Module):
    """
    2-layer GraphSAGE + MLP head for node classification.
    Uses batch normalization and dropout for regularization.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        """
        Initialize GraphSAGE predictor.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output dimension (number of classes)
            dropout: Dropout probability
        """
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features (N, in_dim)
            edge_index: Edge indices (2, E)
        
        Returns:
            Log probabilities for each class (N, out_dim)
        """
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        out = self.mlp(x)  # logits for classes
        log_probs = F.log_softmax(out, dim=-1)  # log-probs for KLDivLoss
        return log_probs


class GCNPredictor(nn.Module):
    """
    2-layer Graph Convolutional Network + MLP head.
    Alternative architecture to GraphSAGE.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        out = self.mlp(x)
        log_probs = F.log_softmax(out, dim=-1)
        return log_probs


class GATPredictor(nn.Module):
    """
    2-layer Graph Attention Network + MLP head.
    Uses attention mechanism for message passing.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, 
                 dropout: float = 0.5, num_heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // num_heads, heads=num_heads)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        out = self.mlp(x)
        log_probs = F.log_softmax(out, dim=-1)
        return log_probs


class SAGEEdgePredictor(nn.Module):
    """
    GraphSAGE encoder + edge MLP head for binary edge classification.
    Predicts same-category (0) vs cross-category (1) for each edge.
    Node embeddings are computed first, then edge representations are
    formed by concatenating endpoint embeddings.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        self.dropout = dropout

    def encode(self, x, edge_index):
        """Return node embeddings."""
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, target_edge_index=None):
        """
        Args:
            x: Node features (N, in_dim)
            edge_index: Graph edges for message passing (2, E)
            target_edge_index: Edges to score (2, E_target).
                               If None, scores all edges in edge_index.
        Returns:
            Edge logits (E_target, out_dim)
        """
        h = self.encode(x, edge_index)
        if target_edge_index is None:
            target_edge_index = edge_index
        src = h[target_edge_index[0]]
        dst = h[target_edge_index[1]]
        edge_repr = torch.cat([src, dst], dim=-1)
        return self.edge_mlp(edge_repr)


class ModelFactory:
    """
    Factory pattern for creating GNN models.
    Supports dynamic model instantiation based on configuration.
    """

    MODELS = {
        'sage': SAGEPredictor,
        'gcn': GCNPredictor,
        'gat': GATPredictor,
        'sage_edge': SAGEEdgePredictor,
    }
    
    @classmethod
    def create_model(cls, model_type: str, in_dim: int, hidden_dim: int,
                     out_dim: int, dropout: float = 0.5,
                     task_type: str = 'categorical', **kwargs):
        """
        Create a GNN model using factory pattern.

        Args:
            model_type: Type of model ('sage', 'gcn', 'gat', 'sage_edge')
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output dimension
            dropout: Dropout probability
            task_type: 'categorical', 'scalar', or 'edge_categorical'.
                       'edge_categorical' forces model_type to 'sage_edge'.
            **kwargs: Additional model-specific arguments

        Returns:
            Instantiated model
        """
        if task_type == 'edge_categorical':
            model_type = 'sage_edge'

        if model_type not in cls.MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(cls.MODELS.keys())}")

        return cls.MODELS[model_type](in_dim, hidden_dim, out_dim, dropout, **kwargs)
    
    @classmethod
    def register_model(cls, name: str, model_class):
        """Register a new model type."""
        cls.MODELS[name] = model_class
