"""
TAG Constructor Module for Ontological Generalization Framework
Implements Builder pattern for constructing Text-Augmented Graphs (TAGs).
"""

import json
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import combinations
from sklearn.base import defaultdict
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split


class TAGConfiguration:
    """
    Configuration class for TAG ontological indices.
    Defines the complete ontology space.
    """
    
    # Task indices [1-6]
    TASKS = {
        1: 'node_categorical',
        2: 'node_scalar', 
        3: 'edge_categorical',
        4: 'edge_scalar',
        5: 'global_categorical',
        6: 'global_scalar'
    }
    
    # Node indices [7-9]
    NODES = {
        7: 'paper',      # Primary: Paper/Article
        8: 'author',     # Secondary: Author
        9: 'journal'     # Aggregate: Journal (not implemented yet)
    }
    
    # Edge indices [10-11c]
    EDGES = {
        10: 'coauthorship',           # Ground Truth
        11: 'cosine_similarity',      # Semantic
        12: 'citation_stats',         # Structural (not implemented)
        13: 'topic_tags'              # Functional (not implemented)
    }
    
    # Edge similarity thresholds for cosine
    SIMILARITY_THRESHOLDS = {
        'a': 0.60,
        'b': 0.75,
        'c': 0.90
    }
    
    # Text fidelity indices [12a-12e]
    TEXT_FIDELITIES = {
        'a': 'super',      # Full text
        'b': 'standard',   # 75% text
        'c': 'mid',        # 50% text
        'd': 'poor',       # 25% text
        'e': 'baseline'    # Title only
    }


class TAG:
    """
    Text-Augmented Graph (TAG) object.
    Contains graph structure, node features, labels, and ontological metadata.
    """
    
    def __init__(self, task_idx: int, node_idx: int, edge_idx: int, 
                 text_idx: str, graph: nx.Graph = None):
        """
        Initialize TAG.
        
        Args:
            task_idx: Task ontological index [1-6]
            node_idx: Node type index [7-9]
            edge_idx: Edge type index [10-13]
            text_idx: Text fidelity index ['a'-'e']
            graph: NetworkX graph object
        """
        self.task_idx = task_idx
        self.node_idx = node_idx
        self.edge_idx = edge_idx
        self.text_idx = text_idx
        self.graph = graph
        self.data = None  # PyTorch Geometric Data object
        self.metadata = {}
    
    def get_identifier(self) -> str:
        base = f"T{self.task_idx}_N{self.node_idx}_E{self.edge_idx}_X{self.text_idx}"
        if hasattr(self, 'similarity_threshold') and self.similarity_threshold:
            base += f"_S{self.similarity_threshold}"
        return base
    
    def to_pyg_data(self, node_list: List, embeddings: np.ndarray, 
                    labels: np.ndarray, device: str = 'cpu') -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data object.
        
        Args:
            node_list: List of node identifiers
            embeddings: Node embeddings (N, feature_dim)
            labels: Node labels (N, num_classes)
            device: Device to place tensors on
        
        Returns:
            PyTorch Geometric Data object
        """
        # Create node index mapping
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        N = len(node_list)
        
        # Ensure all nodes exist in graph
        G_copy = self.graph.copy() if self.graph else nx.Graph()
        for node in node_list:
            if node not in G_copy:
                G_copy.add_node(node)
        
        # Extract edges
        edges = [(node_to_idx[u], node_to_idx[v]) 
                 for u, v in G_copy.edges() 
                 if u in node_to_idx and v in node_to_idx]
        
        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
            edge_index = to_undirected(edge_index)
        
        # Create Data object
        data = Data()
        data.num_nodes = N
        data.x = torch.tensor(embeddings, dtype=torch.float32).to(device)
        data.y = torch.tensor(labels, dtype=torch.float32).to(device)
        data.edge_index = edge_index.to(device)
        
        # Create train/val/test splits
        has_target = (data.y.sum(dim=1) > 0).cpu().numpy()
        idx = np.where(has_target)[0]
        
        if len(idx) < 3:
            print(f"⚠ WARNING: Only {len(idx)} nodes with labels")
            # Use all data for splits
            idx = np.arange(N)
        
        train_idx, test_idx = train_test_split(idx, train_size=0.7, random_state=42)
        val_idx, _ = train_test_split(train_idx, test_size=0.15, random_state=42)
        
        mask_train = np.zeros(N, bool)
        mask_train[train_idx] = True
        mask_val = np.zeros(N, bool)
        mask_val[val_idx] = True
        mask_test = np.zeros(N, bool)
        mask_test[test_idx] = True
        
        data.train_mask = torch.tensor(mask_train).to(device)
        data.val_mask = torch.tensor(mask_val).to(device)
        data.test_mask = torch.tensor(mask_test).to(device)
        
        self.data = data
        return data


class TAGBuilder:
    """
    Builder pattern for constructing TAG variants.
    Implements the complete ontological framework.
    """
    
    def __init__(self, data_manager):
        """
        Initialize TAG builder.
        
        Args:
            data_manager: ArxivDataManager instance with loaded data
        """
        self.data_manager = data_manager
        self.config = TAGConfiguration()
        
    def build_coauthorship_graph(self, node_type: str = 'author') -> nx.Graph:
        """
        Build coauthorship graph.
        
        Args:
            node_type: 'author' or 'paper'
        
        Returns:
            NetworkX graph with coauthorship edges
        """
        G = nx.Graph()
        df = self.data_manager.df
        
        if node_type == 'author':
            # Author-author coauthorship
            unique_authors = set(self.data_manager.author_list)
            for author in tqdm(unique_authors, desc="Adding author nodes"):
                G.add_node(author, type="author")
            
            for row in tqdm(df.itertuples(index=False), total=len(df), desc="Adding coauthorship edges"):
                authors = getattr(row, 'authors_parsed', [])
                if not isinstance(authors, list) or len(authors) < 2:
                    continue
                
                clean_authors = [self.data_manager.handle_author(a) for a in authors]
                clean_authors = [a for a in clean_authors if a and "|" in a]
                
                for a1, a2 in combinations(clean_authors, 2):
                    if a1 == a2:
                        continue
                    if G.has_edge(a1, a2):
                        G[a1][a2]["weight"] += 1
                    else:
                        G.add_edge(a1, a2, type="coauthor", weight=1)
        
        elif node_type == 'paper':
            # Article-article coauthorship (shared authors)
            for article_id in tqdm(df['id'], desc="Adding article nodes"):
                G.add_node(article_id, type='article')
            
            # Map article -> authors
            article_to_authors = {}
            for i, row in df.iterrows():
                authors = getattr(row, 'authors_parsed', [])
                if not isinstance(authors, list):
                    continue
                author_list = [self.data_manager.handle_author(a) for a in authors]
                author_list = [a for a in author_list if a]
                article_to_authors[row['id']] = set(author_list)
            
            # Connect articles with shared authors
            article_ids = list(df['id'])
            for i, article_id in enumerate(tqdm(article_ids, desc="Adding coauthorship edges")):
                authors_i = article_to_authors.get(article_id, set())
                for j in range(i + 1, len(article_ids)):
                    neighbor_id = article_ids[j]
                    authors_j = article_to_authors.get(neighbor_id, set())
                    shared = authors_i.intersection(authors_j)
                    weight = len(shared)
                    if weight > 0:
                        G.add_edge(article_id, neighbor_id, type="coauthorship", weight=weight)
        
        print(f"✅ Coauthorship graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def build_similarity_graph(self, embeddings: np.ndarray, node_list: List,
                               threshold: float = 0.75, k: int = 10,
                               node_type: str = 'author') -> nx.Graph:
        """
        Build cosine similarity graph using k-NN.
        
        Args:
            embeddings: Node embeddings
            node_list: List of node identifiers
            threshold: Minimum cosine similarity threshold
            k: Number of nearest neighbors
            node_type: Type of nodes
        
        Returns:
            NetworkX graph with similarity edges
        """
        G = nx.Graph()
        
        # Add nodes
        for node in tqdm(node_list, desc="Adding nodes"):
            G.add_node(node, type=node_type)
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = normalize(embeddings, axis=1)
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=1)
        nbrs.fit(embeddings_norm)
        distances, indices = nbrs.kneighbors(embeddings_norm)
        
        # Add edges
        for i, node in enumerate(tqdm(node_list, desc="Adding similarity edges")):
            for j, dist in zip(indices[i][1:], distances[i][1:]):
                sim = 1 - dist
                if sim < threshold:
                    continue
                neighbor = node_list[j]
                G.add_edge(node, neighbor, type="cosine", weight=sim)
        
        print(f"✅ Similarity graph (threshold={threshold}): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def build(self, task_idx: int, node_idx: int, edge_idx: int, 
              text_idx: str, similarity_threshold: Optional[str] = None) -> TAG:
        """
        Build a TAG based on ontological indices.
        
        Args:
            task_idx: Task index [1-6]
            node_idx: Node type index [7-9]
            edge_idx: Edge type index [10-13]
            text_idx: Text fidelity index ['a'-'e']
            similarity_threshold: For cosine similarity, specify 'a', 'b', or 'c'
        
        Returns:
            TAG object
        """
        # Validate indices
        if task_idx not in self.config.TASKS:
            raise ValueError(f"Invalid task_idx: {task_idx}")
        if node_idx not in self.config.NODES:
            raise ValueError(f"Invalid node_idx: {node_idx}")
        if edge_idx not in self.config.EDGES:
            raise ValueError(f"Invalid edge_idx: {edge_idx}")
        if text_idx not in self.config.TEXT_FIDELITIES:
            raise ValueError(f"Invalid text_idx: {text_idx}")
        
        # Get configuration
        node_type = self.config.NODES[node_idx]
        edge_type = self.config.EDGES[edge_idx]
        text_fidelity = self.config.TEXT_FIDELITIES[text_idx]
        
        print(f"\n{'='*60}")
        print(f"Building TAG: Task={task_idx}, Node={node_type}, Edge={edge_type}, Text={text_fidelity}")
        print(f"{'='*60}")
        
        # Build graph based on edge type
        if edge_type == 'coauthorship':
            graph = self.build_coauthorship_graph(node_type=node_type)
        elif edge_type == 'cosine_similarity':
            # Determine threshold
            if similarity_threshold:
                threshold = self.config.SIMILARITY_THRESHOLDS[similarity_threshold]
            else:
                threshold = 0.75  # default
            
            if node_type == 'author':
                embeddings = self.data_manager.author_embeddings
                node_list = self.data_manager.author_list
            else:  # paper
                embeddings = self.data_manager.article_embeddings
                node_list = list(self.data_manager.df['id'])
            
            graph = self.build_similarity_graph(embeddings, node_list, threshold, node_type=node_type)
        else:
            raise NotImplementedError(f"Edge type {edge_type} not implemented yet")
        
        # Create TAG object
        tag = TAG(task_idx, node_idx, edge_idx, text_idx, graph)
        tag.metadata = {
            'task': self.config.TASKS[task_idx],
            'node_type': node_type,
            'edge_type': edge_type,
            'text_fidelity': text_fidelity,
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges()
        }
        tag.similarity_threshold = similarity_threshold  # e.g. 'b' or 'c'
        
        return tag
    
    def save_tag(self, tag: TAG, output_dir: str):
        """Save TAG to disk."""
        from networkx.readwrite import json_graph
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        filename = f"TAG_{tag.get_identifier()}.json"
        data = json_graph.node_link_data(tag.graph)
        
        with open(output_path / filename, 'w') as f:
            json.dump(data, f)
        
        # Save metadata
        with open(output_path / f"TAG_{tag.get_identifier()}_meta.json", 'w') as f:
            json.dump(tag.metadata, f, indent=2)
        
        print(f"✅ Saved TAG to {output_path / filename}")
    
    def load_tag(self, filepath: str) -> TAG:
        """Load TAG from disk."""
        from networkx.readwrite import json_graph
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        graph = json_graph.node_link_graph(data)
        
        # Extract indices from filename
        # Expected format: TAG_T{task}_N{node}_E{edge}_X{text}.json
        filename = Path(filepath).stem
        parts = filename.split('_')
        task_idx = int(parts[1][1:])
        node_idx = int(parts[2][1:])
        edge_idx = int(parts[3][1:])
        text_idx = parts[4][1:]
        
        tag = TAG(task_idx, node_idx, edge_idx, text_idx, graph)
        
        # Load metadata if exists
        meta_path = str(filepath).replace('.json', '_meta.json')
        if Path(meta_path).exists():
            with open(meta_path, 'r') as f:
                tag.metadata = json.load(f)
        
        return tag
