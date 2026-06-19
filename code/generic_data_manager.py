"""
Generic Data Manager Implementation
Single implementation that works across all datasets via YAML-driven configuration.
"""

import json
import yaml
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import defaultdict, Counter

from base_data_manager import BaseDataManager


class GenericDataManager(BaseDataManager):
    """
    Unified data manager that works across arxiv, amazon, and history datasets.
    All dataset-specific behavior is driven by {dataset}_dataset.yaml config.
    """
    
    def __init__(self, dataset: str, base_path: str = "data"):
        """
        Initialize data manager.
        
        Args:
            dataset: 'arxiv', 'amazon', or 'history'
            base_path: Base directory containing data/ folder
        """
        self.dataset = dataset
        self.base_path = Path(base_path)
        self.dataset_path = self.base_path / dataset
        
        # Load dataset configuration
        config_path = self.base_path / "configs" / f"{dataset}_dataset.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Cache for loaded DataFrames and full embedding arrays
        self._cache = {}
        self._emb_cache = {}  # keyed by (node_type, fidelity, split) → full ndarray
    
    def load_data(self, split: str, sample_idx: Optional[int] = None) -> pd.DataFrame:
        """Load raw data for a given split."""
        cache_key = f"{split}_{sample_idx}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if sample_idx is not None:
            path = self.dataset_path / split / "samples" / f"sample_{sample_idx:02d}.jsonl"
        else:
            path = self.dataset_path / split / "raw.jsonl"
        
        df = pd.read_json(path, lines=True)
        self._cache[cache_key] = df
        return df
    
    def get_node_list(self, node_type: str, df: pd.DataFrame) -> List:
        """Get list of node identifiers for a given node type."""
        if node_type == 'N7':
            # Unique primary entities in df order (dict.fromkeys preserves insertion order).
            # Amazon has multiple reviews per product ASIN; returning duplicates would
            # cause shape mismatches downstream since graph nodes are deduplicated.
            return list(dict.fromkeys(df['primary_id'].tolist()))
        
        elif node_type == 'N8':
            # Secondary entities (authors/users)
            if not self.config['has_secondary_id']:
                raise ValueError(f"N8 unavailable for {self.dataset}")
            
            # Extract unique secondary IDs
            unique_ids = set()
            for sec_id in df['secondary_id']:
                if sec_id is None:
                    continue
                # Handle list (arxiv) vs single value (amazon)
                if isinstance(sec_id, list):
                    unique_ids.update(sec_id)
                else:
                    unique_ids.add(sec_id)
            
            return sorted(list(unique_ids))
        
        elif node_type == 'N9':
            # Aggregate entities (categories/families)
            return sorted(df['aggregate_id'].dropna().unique().tolist())
        
        else:
            raise ValueError(f"Unknown node_type: {node_type}")
    
    def get_node_text(self, primary_id, fidelity: str, df: pd.DataFrame) -> str:
        """Get text for a node at specified fidelity level."""
        row = df[df['primary_id'] == primary_id]
        if len(row) == 0:
            return ""
        
        row = row.iloc[0]
        
        if fidelity == 'T12a':
            return row['text_fidelity_a'] or ""
        elif fidelity == 'T12b':
            return row['text_fidelity_b'] or ""
        elif fidelity == 'T12e':
            return ""  # T12e is always empty by definition
        else:
            raise ValueError(f"Unknown fidelity: {fidelity}")
    
    def get_embeddings(self, node_type: str, fidelity: str, split: str,
                      node_list: Optional[List] = None) -> np.ndarray:
        """Get embeddings for nodes at specified fidelity level."""
        emb_path = self.dataset_path / split / "embeddings"
        
        if node_type == 'N7':
            # Primary entity embeddings
            prefix = self.config['embedding_prefix']
            
            if fidelity == 'T12a':
                # Contextual embeddings
                if self.config['contextual_embedding_available']:
                    emb_file = emb_path / f"{prefix}_embeddings_contextual.npy"
                else:
                    # Fall back to T12b if contextual not available
                    emb_file = emb_path / f"{prefix}_embeddings.npy"
            elif fidelity == 'T12b':
                # Standard embeddings
                emb_file = emb_path / f"{prefix}_embeddings.npy"
            elif fidelity == 'T12e':
                prefix = self.config['embedding_prefix']
                ref = np.load(emb_path / f"{prefix}_embeddings.npy", mmap_mode='r')
                n_rows = len(node_list) if node_list is not None else len(self.load_data(split))
                return np.zeros((n_rows, ref.shape[1]), dtype=np.float32)
            else:
                raise ValueError(f"Unknown fidelity: {fidelity}")
            
            cache_key = (node_type, fidelity, split)
            if cache_key not in self._emb_cache:
                self._emb_cache[cache_key] = np.load(emb_file)
            embeddings = self._emb_cache[cache_key]

            if node_list is not None:
                full_df = self.load_data(split)
                pid_to_first_idx: dict = {}
                for i, pid in enumerate(full_df['primary_id']):
                    if pid not in pid_to_first_idx:
                        pid_to_first_idx[pid] = i
                indices = [pid_to_first_idx[pid] for pid in node_list
                           if pid in pid_to_first_idx]
                embeddings = embeddings[indices]
            
            return embeddings
        
        elif node_type == 'N8':
            # Secondary entity embeddings
            if not self.config['has_secondary_id']:
                raise ValueError(f"N8 unavailable for {self.dataset}")
            
            prefix = self.config['secondary_embedding_prefix']
            emb_file = emb_path / f"{prefix}_embeddings.npy"
            index_file = emb_path / f"{prefix}_index.json"
            
            cache_key = (node_type, fidelity, split)
            if cache_key not in self._emb_cache:
                self._emb_cache[cache_key] = np.load(emb_file)
            embeddings = self._emb_cache[cache_key]

            with open(index_file, 'r') as f:
                index = json.load(f)

            if node_list is not None:
                index_map = {nid: i for i, nid in enumerate(index)}
                indices = [index_map[nid] for nid in node_list if nid in index_map]
                embeddings = embeddings[indices]

            return embeddings

        elif node_type == 'N9':
            # Aggregate entity embeddings
            prefix = self.config['aggregate_embedding_prefix']
            emb_file = emb_path / f"{prefix}_embeddings.npy"
            index_file = emb_path / f"{prefix}_index.json"
            
            cache_key = (node_type, fidelity, split)
            if cache_key not in self._emb_cache:
                self._emb_cache[cache_key] = np.load(emb_file)
            embeddings = self._emb_cache[cache_key]

            with open(index_file, 'r') as f:
                index = json.load(f)

            if node_list is not None:
                index_map = {nid: i for i, nid in enumerate(index)}
                indices = [index_map[nid] for nid in node_list if nid in index_map]
                embeddings = embeddings[indices]

            return embeddings

        else:
            raise ValueError(f"Unknown node_type: {node_type}")
    
    def get_category_labels(self, node_type: str, df: pd.DataFrame,
                           node_list: Optional[List] = None) -> np.ndarray:
        """Get categorical labels for M1/M3 tasks."""
        if node_type == 'N7':
            # Build pid → first label mapping so Amazon (multiple reviews per
            # product) returns exactly one label per unique primary_id in node_list.
            pid_to_label: dict = {}
            for pid, lbl in zip(df['primary_id'], df['categorical_label']):
                if pid not in pid_to_label:
                    pid_to_label[pid] = lbl

            if node_list is None:
                node_list = self.get_node_list('N7', df)

            labels = np.array([pid_to_label.get(pid, np.nan) for pid in node_list])

            unique_labels = sorted(df['categorical_label'].dropna().unique())
            label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            n_classes = len(unique_labels)

            class_indices = np.array([label_to_idx.get(lbl, -1) for lbl in labels])
            one_hot = np.zeros((len(node_list), n_classes), dtype=np.float32)
            valid_mask = class_indices >= 0
            one_hot[valid_mask, class_indices[valid_mask]] = 1.0
            return one_hot
        
        elif node_type == 'N8':
            # Secondary entity labels: aggregate from primary entities
            if not self.config['has_secondary_id']:
                raise ValueError(f"N8 unavailable for {self.dataset}")
            
            # Build mapping: secondary_id -> list of categorical_labels
            sec_to_labels = defaultdict(list)
            for _, row in df.iterrows():
                sec_id = row['secondary_id']
                cat_label = row['categorical_label']
                
                if sec_id is None or pd.isna(cat_label):
                    continue
                
                # Handle list (arxiv) vs single value (amazon)
                if isinstance(sec_id, list):
                    for sid in sec_id:
                        sec_to_labels[sid].append(cat_label)
                else:
                    sec_to_labels[sec_id].append(cat_label)
            
            # Compute mode (most common) label for each secondary entity
            if node_list is None:
                node_list = self.get_node_list('N8', df)
            
            mode_labels = []
            for nid in node_list:
                labels_list = sec_to_labels.get(nid, [])
                if labels_list:
                    mode_label = Counter(labels_list).most_common(1)[0][0]
                    mode_labels.append(mode_label)
                else:
                    mode_labels.append(None)
            
            # Convert to one-hot
            unique_labels = sorted(df['categorical_label'].dropna().unique())
            label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            n_classes = len(unique_labels)
            
            one_hot = np.zeros((len(node_list), n_classes), dtype=np.float32)
            for i, label in enumerate(mode_labels):
                if label is not None and label in label_to_idx:
                    one_hot[i, label_to_idx[label]] = 1.0
            
            return one_hot
        
        elif node_type == 'N9':
            # N9 categorical labels are tautological (node IS the category)
            # This should not be called for M1+N9 (excluded by remove list)
            raise ValueError("M1+N9 is tautological and should be excluded")
        
        else:
            raise ValueError(f"Unknown node_type: {node_type}")
    
    def get_scalar_labels(self, node_type: str, df: pd.DataFrame,
                         graph=None, node_list: Optional[List] = None) -> np.ndarray:
        """Get scalar labels for M2/M4 tasks."""
        if node_type == 'N7':
            # Primary entity scalar labels
            if self.dataset == 'arxiv':
                # ArXiv M2: compute centrality from graph (node_list may be passed
                # in; fall back to unique primary_ids in df)
                if graph is None:
                    raise ValueError("Graph required for ArXiv M2 centrality computation")
                nl = node_list if node_list is not None else list(dict.fromkeys(df['primary_id']))
                return self._compute_centrality(graph, nl)
            else:
                # Amazon/History: use scalar_label column.
                # Build pid → first scalar mapping to handle Amazon's multiple
                # reviews per product (node_list contains unique primary_ids).
                if node_list is None:
                    node_list = self.get_node_list('N7', df)
                pid_to_scalar: dict = {}
                for pid, val in zip(df['primary_id'], df['scalar_label']):
                    if pid not in pid_to_scalar:
                        pid_to_scalar[pid] = val
                labels = np.array([pid_to_scalar.get(pid, np.nan)
                                   for pid in node_list], dtype=np.float32)
                return labels.reshape(-1, 1)
        
        elif node_type == 'N8':
            # Secondary entity scalar labels: aggregate from primary entities
            if not self.config['has_secondary_id']:
                raise ValueError(f"N8 unavailable for {self.dataset}")
            
            if self.dataset == 'arxiv':
                # ArXiv: mean centrality of secondary entity's primary entities
                if graph is None:
                    raise ValueError("Graph required for ArXiv M2 centrality computation")
                
                # First compute N7 centralities
                n7_centralities = self._compute_centrality(graph, df['primary_id'].tolist())
                n7_cent_map = dict(zip(df['primary_id'], n7_centralities.flatten()))
                
                # Aggregate to N8
                sec_to_cents = defaultdict(list)
                for _, row in df.iterrows():
                    sec_id = row['secondary_id']
                    prim_id = row['primary_id']
                    
                    if sec_id is None:
                        continue
                    
                    cent = n7_cent_map.get(prim_id, 0.0)
                    
                    # Handle list (arxiv) vs single value (amazon)
                    if isinstance(sec_id, list):
                        for sid in sec_id:
                            sec_to_cents[sid].append(cent)
                    else:
                        sec_to_cents[sec_id].append(cent)
                
                # Compute mean for each N8 node
                if node_list is None:
                    node_list = self.get_node_list('N8', df)
                
                mean_cents = np.array([
                    np.mean(sec_to_cents.get(nid, [0.0])) for nid in node_list
                ], dtype=np.float32)
                
                return mean_cents.reshape(-1, 1)
            else:
                # Amazon: mean helpful_vote of user's reviews
                sec_to_scalars = defaultdict(list)
                for _, row in df.iterrows():
                    sec_id = row['secondary_id']
                    scalar = row['scalar_label']
                    
                    if sec_id is None or pd.isna(scalar):
                        continue
                    
                    sec_to_scalars[sec_id].append(float(scalar))
                
                if node_list is None:
                    node_list = self.get_node_list('N8', df)
                
                mean_scalars = np.array([
                    np.mean(sec_to_scalars.get(nid, [0.0])) for nid in node_list
                ], dtype=np.float32)
                
                return mean_scalars.reshape(-1, 1)
        
        elif node_type == 'N9':
            # Aggregate entity scalar labels
            if self.dataset == 'arxiv':
                # Mean centrality of category's papers
                if graph is None:
                    raise ValueError("Graph required for ArXiv M2 centrality computation")
                
                n7_centralities = self._compute_centrality(graph, df['primary_id'].tolist())
                n7_cent_map = dict(zip(df['primary_id'], n7_centralities.flatten()))
                
                agg_to_cents = defaultdict(list)
                for _, row in df.iterrows():
                    agg_id = row['aggregate_id']
                    prim_id = row['primary_id']
                    
                    if pd.isna(agg_id):
                        continue
                    
                    cent = n7_cent_map.get(prim_id, 0.0)
                    agg_to_cents[agg_id].append(cent)
                
                if node_list is None:
                    node_list = self.get_node_list('N9', df)
                
                mean_cents = np.array([
                    np.mean(agg_to_cents.get(nid, [0.0])) for nid in node_list
                ], dtype=np.float32)
                
                return mean_cents.reshape(-1, 1)
            else:
                # Amazon/History: mean scalar of aggregate's primary entities
                agg_to_scalars = defaultdict(list)
                for _, row in df.iterrows():
                    agg_id = row['aggregate_id']
                    scalar = row['scalar_label']
                    
                    if pd.isna(agg_id) or pd.isna(scalar):
                        continue
                    
                    agg_to_scalars[agg_id].append(float(scalar))
                
                if node_list is None:
                    node_list = self.get_node_list('N9', df)
                
                mean_scalars = np.array([
                    np.mean(agg_to_scalars.get(nid, [0.0])) for nid in node_list
                ], dtype=np.float32)
                
                return mean_scalars.reshape(-1, 1)
        
        else:
            raise ValueError(f"Unknown node_type: {node_type}")
    
    def _compute_centrality(self, graph: nx.Graph, node_list: List) -> np.ndarray:
        """
        Compute centrality composite for ArXiv M2.
        Normalized sum of PageRank, clustering coefficient, and degree centrality.
        """
        pagerank = nx.pagerank(graph, alpha=0.85)
        clustering = nx.clustering(graph)
        degree_cent = nx.degree_centrality(graph)
        
        def normalize(d, keys):
            vals = np.array([d.get(k, 0.0) for k in keys], dtype=np.float64)
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                vals = (vals - vmin) / (vmax - vmin)
            return vals
        
        pr = normalize(pagerank, node_list)
        cl = normalize(clustering, node_list)
        dc = normalize(degree_cent, node_list)
        
        combined = (pr + cl + dc) / 3.0
        return combined.astype(np.float32).reshape(-1, 1)
    
    def get_structural_edges(self, df: pd.DataFrame) -> List[Tuple]:
        """Get pre-given structural edges (E10c, History only)."""
        if not self.config['has_structural_edges']:
            raise ValueError(f"E10c structural edges unavailable for {self.dataset}")
        
        edges = []
        for _, row in df.iterrows():
            src = row['primary_id']
            targets = row['structural_edges']
            
            if targets is None:
                continue
            
            for tgt in targets:
                edges.append((src, tgt))
        
        return edges
    
    def get_config(self) -> Dict:
        """Get dataset configuration."""
        return self.config

# Made with Bob
