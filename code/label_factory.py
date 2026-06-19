"""
Label Factory for TAG Construction
Generates labels based on task type (M1-M6).
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
import pandas as pd
from collections import defaultdict, Counter


class LabelFactory:
    """
    Factory for generating labels based on task type.
    """
    
    @staticmethod
    def generate_labels(task_type: str, node_type: str, data_manager,
                       df: pd.DataFrame, graph: Optional[nx.Graph] = None,
                       node_list: Optional[List] = None):
        """
        Generate labels based on task type.
        
        Args:
            task_type: 'M1', 'M2', 'M3', 'M4', 'M5', or 'M6'
            node_type: 'N7', 'N8', or 'N9'
            data_manager: GenericDataManager instance
            df: DataFrame with sample data
            graph: NetworkX graph (required for M2/M4/M5/M6)
            node_list: List of node IDs
        
        Returns:
            Label array or (edge_list, label_array) for M3/M4
        """
        if task_type == 'M1':
            # Node categorical
            return data_manager.get_category_labels(node_type, df, node_list)
        
        elif task_type == 'M2':
            # Node scalar
            return data_manager.get_scalar_labels(node_type, df, graph, node_list)
        
        elif task_type == 'M3':
            # Edge categorical (same/cross category)
            return LabelFactory._generate_edge_categorical(df, graph, node_type, data_manager)
        
        elif task_type == 'M4':
            # Edge scalar (label difference)
            return LabelFactory._generate_edge_scalar(df, graph, node_type, data_manager)
        
        elif task_type == 'M5':
            # Global categorical
            return LabelFactory._generate_global_categorical(graph, data_manager.dataset)
        
        elif task_type == 'M6':
            # Global scalar
            return LabelFactory._generate_global_scalar(graph, data_manager.dataset)
        
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
    
    @staticmethod
    def _generate_edge_categorical(df: pd.DataFrame, graph: nx.Graph,
                                   node_type: str, data_manager) -> Tuple[List, np.ndarray]:
        """
        M3: Edge categorical labels (same/cross category).
        
        Returns:
            (edge_list, labels) where labels[i] = 0 (same category) or 1 (cross category)
        """
        # Get category mapping
        if node_type == 'N7':
            node_to_cat = dict(zip(df['primary_id'], df['categorical_label']))
        
        elif node_type == 'N8':
            # For N8, compute dominant category
            sec_to_cats = defaultdict(list)
            
            for _, row in df.iterrows():
                sec_id = row['secondary_id']
                cat = row['categorical_label']
                
                if sec_id is None:
                    continue
                if isinstance(cat, float) and np.isnan(cat):
                    continue
                
                # Handle list (arxiv) vs single value (amazon)
                if isinstance(sec_id, list):
                    for sid in sec_id:
                        sec_to_cats[sid].append(cat)
                else:
                    sec_to_cats[sec_id].append(cat)
            
            # Compute mode category
            node_to_cat = {}
            for nid, cats in sec_to_cats.items():
                if cats:
                    node_to_cat[nid] = Counter(cats).most_common(1)[0][0]
        
        else:
            raise ValueError("M3 undefined for N9")
        
        # Label each edge
        edge_list = list(graph.edges())
        labels = []
        
        for u, v in edge_list:
            cat_u = node_to_cat.get(u)
            cat_v = node_to_cat.get(v)
            
            if cat_u is None or cat_v is None:
                labels.append(-1)  # Unknown—will be masked
            elif cat_u == cat_v:
                labels.append(0)  # Same category
            else:
                labels.append(1)  # Cross category
        
        return edge_list, np.array(labels, dtype=np.int64)
    
    @staticmethod
    def _generate_edge_scalar(df: pd.DataFrame, graph: nx.Graph,
                             node_type: str, data_manager) -> Tuple[List, np.ndarray]:
        """
        M4: Edge scalar labels (absolute difference in scalar labels).
        
        Returns:
            (edge_list, labels) where labels[i] = |scalar_u - scalar_v|
        """
        # First get node-level scalar labels
        node_list = list(graph.nodes())
        scalar_labels = data_manager.get_scalar_labels(node_type, df, graph, node_list)
        
        # Create mapping
        node_to_scalar = dict(zip(node_list, scalar_labels.flatten()))
        
        # Compute edge labels
        edge_list = list(graph.edges())
        labels = []
        
        for u, v in edge_list:
            scalar_u = node_to_scalar.get(u, 0.0)
            scalar_v = node_to_scalar.get(v, 0.0)
            
            diff = abs(scalar_u - scalar_v)
            labels.append(diff)
        
        return edge_list, np.array(labels, dtype=np.float32)
    
    @staticmethod
    def _generate_global_categorical(graph: nx.Graph, dataset: str) -> int:
        """
        M5: Global categorical label.
        
        ArXiv: entropy > 0.5 -> 1, else 0
        History: density > 0.05 -> 1, else 0
        """
        if dataset == 'arxiv':
            entropy = LabelFactory._compute_graph_entropy(graph)
            return 1 if entropy > 0.5 else 0
        
        elif dataset == 'history':
            density = nx.density(graph)
            return 1 if density > 0.05 else 0
        
        elif dataset == 'amazon':
            clustering = nx.average_clustering(graph)
            return 1 if clustering > 0.3 else 0
        
        else:
            raise ValueError(f"M5 not defined for {dataset}")
    
    @staticmethod
    def _generate_global_scalar(graph: nx.Graph, dataset: str) -> float:
        """
        M6: Global scalar label.
        
        ArXiv: graph entropy value
        History: network density value
        """
        if dataset == 'arxiv':
            return LabelFactory._compute_graph_entropy(graph)
        
        elif dataset == 'history':
            return nx.density(graph)
        
        elif dataset == 'amazon':
            return nx.average_clustering(graph)
        
        else:
            raise ValueError(f"M6 not defined for {dataset}")
    
    @staticmethod
    def _compute_graph_entropy(graph: nx.Graph) -> float:
        """
        Compute graph entropy for ArXiv M5/M6.
        Based on degree distribution entropy.
        """
        if graph.number_of_nodes() == 0:
            return 0.0
        
        # Get degree sequence
        degrees = [d for n, d in graph.degree()]
        
        if len(degrees) == 0:
            return 0.0
        
        # Compute degree distribution
        degree_counts = Counter(degrees)
        total = sum(degree_counts.values())
        
        # Compute entropy
        entropy = 0.0
        for count in degree_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return float(entropy)

# Made with Bob
