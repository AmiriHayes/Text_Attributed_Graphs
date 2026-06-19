"""
GNN Models Module — TAG Research (code/ version)
ModelFactory preserving the 2-layer GraphSAGE backbone from root models.py.
Heads are task-specific; output_dimension is always resolved dynamically
from data.y.shape[-1] at runtime (never hardcoded).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


# ---------------------------------------------------------------------------
# Node-level backbone (M1 categorical, M2 scalar)
# ---------------------------------------------------------------------------

class SAGEPredictor(nn.Module):
    """
    2-layer GraphSAGE + MLP head for node-level tasks.

    For M1 (categorical): out_dim = data.y.shape[-1] (number of classes,
    read dynamically), output is log_softmax (log-probs for KLDivLoss).

    For M2 (scalar): out_dim = 1, raw scalar output (no activation).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.5, log_softmax_output: bool = True):
        """
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output dimension (number of classes for M1; 1 for M2)
            dropout: Dropout probability
            log_softmax_output: True for M1 (KLDivLoss needs log-probs),
                                False for M2 (raw scalar, MSE target)
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
            nn.Linear(hidden_dim // 2, out_dim),
        )
        self.dropout = dropout
        self.log_softmax_output = log_softmax_output

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
        if self.log_softmax_output:
            return F.log_softmax(out, dim=-1)   # M1: log-probs for KLDivLoss
        return out                               # M2: raw scalar


# ---------------------------------------------------------------------------
# Graph-level backbone (M5 global categorical, M6 global scalar)
# ---------------------------------------------------------------------------

class SAGEGlobalPredictor(nn.Module):
    """
    2-layer GraphSAGE encoder + global_mean_pool + MLP head for graph-level tasks.

    For M5 (global_categorical): out_dim=2, raw logits (CrossEntropyLoss).
    For M6 (global_scalar):      out_dim=1, raw scalar (MSELoss).

    forward(x, edge_index, batch) requires a batch tensor produced by
    torch_geometric.data.DataLoader when batching multiple Data objects.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.5, log_softmax_output: bool = False):
        """
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: 2 for M5 (binary graph classification), 1 for M6 (graph regression)
            dropout: Dropout probability
            log_softmax_output: Unused — kept for API symmetry; always False for M5/M6.
                                M5 uses CrossEntropyLoss which expects raw logits.
        """
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )
        self.dropout = dropout
        self.log_softmax_output = log_softmax_output   # always False for M5/M6

    def forward(self, x, edge_index, batch):
        """
        Args:
            x:          Node features  (N_total_nodes, in_dim)
            edge_index: Graph edges    (2, E_total_edges)
            batch:      Node-to-graph mapping (N_total_nodes,) — provided by DataLoader
        Returns:
            Graph-level output (B, out_dim)  — raw logits (M5) or raw scalar (M6)
        """
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Pool all nodes in each graph to a single graph-level vector
        x = global_mean_pool(x, batch)   # (B, hidden_dim)
        return self.mlp(x)               # (B, out_dim)


# ---------------------------------------------------------------------------
# Edge-level backbone (M3 categorical, M4 scalar)
# ---------------------------------------------------------------------------

class SAGEEdgePredictor(nn.Module):
    """
    GraphSAGE encoder + edge MLP head for edge-level tasks.

    Node embeddings are produced by 2×SAGEConv layers, then each edge is
    represented by concatenating the embeddings of its two endpoints and
    passed through a small MLP.

    For M3 (edge categorical): out_dim = 2, raw logits (CrossEntropyLoss).
    For M4 (edge scalar):      out_dim = 1, raw scalar (MSELoss).
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
            nn.Linear(hidden_dim, out_dim),
        )
        self.dropout = dropout

    def encode(self, x, edge_index):
        """Return node embeddings (shared between M3 and M4)."""
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
            edge_index: Graph edges for message passing (2, E_all)
            target_edge_index: Edges to score (2, E_target).
                               If None, scores all edges in edge_index.
        Returns:
            Edge scores: (E_target, out_dim)  — raw logits (M3) or raw scalar (M4)
        """
        h = self.encode(x, edge_index)
        if target_edge_index is None:
            target_edge_index = edge_index
        src = h[target_edge_index[0]]
        dst = h[target_edge_index[1]]
        edge_repr = torch.cat([src, dst], dim=-1)
        return self.edge_mlp(edge_repr)


# ---------------------------------------------------------------------------
# ModelFactory
# ---------------------------------------------------------------------------

class ModelFactory:
    """
    Factory for instantiating GNN models by task type.

    Task routing:
      M1 — 'categorical'       → SAGEPredictor(log_softmax=True,  out_dim=data.y.shape[-1])
      M2 — 'scalar'            → SAGEPredictor(log_softmax=False, out_dim=1)
      M3 — 'edge_categorical'  → SAGEEdgePredictor(out_dim=2)
      M4 — 'edge_scalar'       → SAGEEdgePredictor(out_dim=1)
      M5 — 'global_categorical'→ SAGEGlobalPredictor(out_dim=2)
      M6 — 'global_scalar'     → SAGEGlobalPredictor(out_dim=1)
    """

    @staticmethod
    def create(task_type: str, in_dim: int, hidden_dim: int,
               out_dim: int, dropout: float = 0.5) -> nn.Module:
        """
        Create and return a GNN model for the given task type.

        Args:
            task_type: One of 'categorical', 'scalar', 'edge_categorical',
                       'edge_scalar', 'global_categorical', 'global_scalar'
            in_dim: Input feature dimension (embedding size)
            hidden_dim: Hidden dimension for SAGE layers + MLP
            out_dim: Output dimension.
                     For 'categorical': data.y.shape[-1] (dynamic, never hardcode)
                     For 'scalar': 1
                     For 'edge_categorical': 2
                     For 'edge_scalar': 1
            dropout: Dropout probability (default 0.5)

        Returns:
            Instantiated nn.Module

        Raises:
            ValueError: for unknown task_type
        """
        if task_type == 'categorical':
            # M1: log_softmax output for KLDivLoss
            return SAGEPredictor(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                dropout=dropout,
                log_softmax_output=True,
            )

        elif task_type == 'scalar':
            # M2: raw scalar output for MSELoss
            return SAGEPredictor(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=1,
                dropout=dropout,
                log_softmax_output=False,
            )

        elif task_type == 'edge_categorical':
            # M3: SAGEEdgePredictor, 2-class logits for CrossEntropyLoss
            return SAGEEdgePredictor(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=2,
                dropout=dropout,
            )

        elif task_type == 'edge_scalar':
            # M4: SAGEEdgePredictor, 1 raw scalar for MSELoss
            return SAGEEdgePredictor(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=1,
                dropout=dropout,
            )

        elif task_type == 'global_categorical':
            # M5: binary graph classification, CrossEntropyLoss (raw logits)
            return SAGEGlobalPredictor(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=2,
                dropout=dropout,
                log_softmax_output=False,
            )

        elif task_type == 'global_scalar':
            # M6: graph-level regression, MSELoss (raw scalar)
            return SAGEGlobalPredictor(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=1,
                dropout=dropout,
                log_softmax_output=False,
            )

        else:
            raise ValueError(
                f"Unknown task_type: '{task_type}'. "
                "Expected one of: 'categorical', 'scalar', 'edge_categorical', "
                "'edge_scalar', 'global_categorical', 'global_scalar'."
            )

    # ------------------------------------------------------------------
    # Convenience: map variant M-code to task_type string
    # ------------------------------------------------------------------
    M_TO_TASK_TYPE = {
        'M1': 'categorical',
        'M2': 'scalar',
        'M3': 'edge_categorical',
        'M4': 'edge_scalar',
        'M5': 'global_categorical',
        'M6': 'global_scalar',
    }

    @classmethod
    def task_type_for(cls, M: str) -> str:
        """Return the task_type string for a given M-code (e.g. 'M1' → 'categorical')."""
        if M not in cls.M_TO_TASK_TYPE:
            raise ValueError(f"Unknown M-code: '{M}'")
        return cls.M_TO_TASK_TYPE[M]
