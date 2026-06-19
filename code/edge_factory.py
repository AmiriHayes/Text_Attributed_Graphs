"""
Edge Factory for TAG Construction
Builds edges based on edge type (E10a-E11c) and node type.
"""

import numpy as np
import networkx as nx
import pandas as pd
from typing import List, Tuple, Optional, Dict
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity


class EdgeFactory:
    """
    Factory for constructing edges based on edge type.
    Each method returns a list of (source, target, [weight]) tuples.
    """
    
    @staticmethod
    def build_edges(edge_type: str, node_type: str, data_manager,
                   df, node_list: List, **kwargs) -> List[Tuple]:
        """
        Build edges based on edge type.
        
        Args:
            edge_type: 'E10a', 'E10b', 'E10c', 'E11a', 'E11b', 'E11c'
            node_type: 'N7', 'N8', or 'N9'
            data_manager: GenericDataManager instance
            df: DataFrame with sample data
            node_list: List of node IDs for this graph
            **kwargs: Additional parameters (embeddings, threshold, etc.)
        
        Returns:
            List of (source, target) or (source, target, weight) tuples
        """
        if edge_type == 'E10a':
            return EdgeFactory._build_categorical_gt(df, node_type, node_list)
        
        elif edge_type == 'E10b':
            return EdgeFactory._build_scalar_gt(df, node_type, node_list, data_manager)
        
        elif edge_type == 'E10c':
            return EdgeFactory._build_binary_participation(df, node_type, node_list, data_manager)
        
        elif edge_type == 'E11a':
            embeddings = kwargs.get('embeddings')
            threshold = kwargs.get('threshold', 0.75)
            return EdgeFactory._build_semantic_similarity(node_list, embeddings, threshold)
        
        elif edge_type == 'E11b':
            base_graph = kwargs.get('base_graph')
            return EdgeFactory._build_structural_similarity(node_list, base_graph)
        
        elif edge_type == 'E11c':
            # Functional similarity: same as E10a but framed as similarity
            return EdgeFactory._build_categorical_gt(df, node_type, node_list)
        
        else:
            raise ValueError(f"Unknown edge_type: {edge_type}")
    
    @staticmethod
    def _build_categorical_gt(df, node_type: str, node_list: List) -> List[Tuple]:
        """
        E10a/E11c: Connect nodes with same aggregate_id.
        """
        edges = []
        
        if node_type == 'N7':
            # Group by aggregate_id
            node_to_agg = dict(zip(df['primary_id'], df['aggregate_id']))
            agg_to_nodes = defaultdict(list)
            
            for nid in node_list:
                agg = node_to_agg.get(nid)
                if agg is not None and not (isinstance(agg, float) and np.isnan(agg)):
                    agg_to_nodes[agg].append(nid)
            
            # Connect all nodes in same aggregate
            for nodes in agg_to_nodes.values():
                for u, v in combinations(nodes, 2):
                    edges.append((u, v))
        
        elif node_type == 'N8':
            # For N8, need to compute dominant aggregate_id per secondary entity
            sec_to_agg = defaultdict(list)
            
            for _, row in df.iterrows():
                sec_id = row['secondary_id']
                agg_id = row['aggregate_id']
                
                if sec_id is None:
                    continue
                if isinstance(agg_id, float) and np.isnan(agg_id):
                    continue
                
                # Handle list (arxiv) vs single value (amazon)
                if isinstance(sec_id, list):
                    for sid in sec_id:
                        sec_to_agg[sid].append(agg_id)
                else:
                    sec_to_agg[sec_id].append(agg_id)
            
            # Compute mode aggregate for each N8 node
            node_to_agg = {}
            for nid in node_list:
                agg_list = sec_to_agg.get(nid, [])
                if agg_list:
                    mode_agg = Counter(agg_list).most_common(1)[0][0]
                    node_to_agg[nid] = mode_agg
            
            # Group by dominant aggregate
            agg_to_nodes = defaultdict(list)
            for nid, agg in node_to_agg.items():
                agg_to_nodes[agg].append(nid)
            
            # Connect nodes with same dominant aggregate
            for nodes in agg_to_nodes.values():
                for u, v in combinations(nodes, 2):
                    edges.append((u, v))
        
        elif node_type == 'N9':
            # N9 nodes ARE aggregates—undefined
            raise ValueError("E10a/E11c undefined for N9 (nodes ARE categories)")
        
        return edges
    
    @staticmethod
    def _build_scalar_gt(df, node_type: str, node_list: List, data_manager) -> List[Tuple]:
        """
        E10b: Dataset-specific weighted edges.
        """
        dataset = data_manager.dataset
        
        if dataset == 'arxiv':
            return EdgeFactory._build_arxiv_coauthorship(df, node_type, node_list)
        elif dataset == 'amazon':
            return EdgeFactory._build_amazon_scalar_gt(df, node_type, node_list)
        elif dataset == 'history':
            return EdgeFactory._build_history_neighbour_cooccurrence(df, node_list)
        else:
            raise ValueError(f"E10b not defined for {dataset}")
    
    @staticmethod
    def _build_arxiv_coauthorship(df, node_type: str, node_list: List) -> List[Tuple]:
        """
        ArXiv E10b: Weighted by shared authors (N7) or co-authored papers (N8).
        """
        edges = defaultdict(int)
        
        if node_type == 'N7':
            # Paper-paper edges weighted by shared authors
            # Build inverted index: author -> papers
            author_to_papers = defaultdict(set)
            for _, row in df.iterrows():
                paper_id = row['primary_id']
                if paper_id not in node_list:
                    continue
                
                authors = row['secondary_id'] or []
                for author in authors:
                    author_to_papers[author].add(paper_id)
            
            # For each author, connect all their papers
            for papers in author_to_papers.values():
                for p1, p2 in combinations(papers, 2):
                    edge_key = tuple(sorted([p1, p2]))
                    edges[edge_key] += 1
        
        elif node_type == 'N8':
            # Author-author edges weighted by co-authored papers
            # Build paper -> authors mapping
            paper_to_authors = {}
            for _, row in df.iterrows():
                authors = row['secondary_id'] or []
                # Filter to authors in node_list
                authors_in_graph = [a for a in authors if a in node_list]
                if len(authors_in_graph) >= 2:
                    paper_to_authors[row['primary_id']] = authors_in_graph
            
            # For each paper, connect all co-authors
            for authors in paper_to_authors.values():
                for a1, a2 in combinations(authors, 2):
                    edge_key = tuple(sorted([a1, a2]))
                    edges[edge_key] += 1
        
        elif node_type == 'N9':
            # Category-category edges: aggregate from N7 level
            # Count cross-category shared-authorship links
            # First build N7-level edges
            n7_edges = EdgeFactory._build_arxiv_coauthorship(df, 'N7', df['primary_id'].tolist())
            
            # Map papers to categories
            paper_to_cat = dict(zip(df['primary_id'], df['aggregate_id']))
            
            # Aggregate to N9
            for u, v, weight in n7_edges:
                cat_u = paper_to_cat.get(u)
                cat_v = paper_to_cat.get(v)
                
                if cat_u and cat_v and cat_u != cat_v and cat_u in node_list and cat_v in node_list:
                    edge_key = tuple(sorted([cat_u, cat_v]))
                    edges[edge_key] += weight
        
        return [(u, v, w) for (u, v), w in edges.items()]
    
    @staticmethod
    def _build_amazon_scalar_gt(df, node_type: str, node_list: List) -> List[Tuple]:
        """
        Amazon E10b: Product-product by shared users, user-user by shared products.
        """
        edges = defaultdict(int)
        
        if node_type == 'N7':
            # Product-product edges weighted by shared users
            user_to_products = defaultdict(set)
            for _, row in df.iterrows():
                product_id = row['primary_id']
                user_id = row['secondary_id']
                
                if product_id not in node_list or user_id is None:
                    continue
                
                user_to_products[user_id].add(product_id)
            
            # Connect products reviewed by same user
            for products in user_to_products.values():
                for p1, p2 in combinations(products, 2):
                    edge_key = tuple(sorted([p1, p2]))
                    edges[edge_key] += 1
        
        elif node_type == 'N8':
            # User-user edges weighted by reviewed same products
            product_to_users = defaultdict(set)
            for _, row in df.iterrows():
                user_id = row['secondary_id']
                product_id = row['primary_id']
                
                if user_id is None or user_id not in node_list:
                    continue
                
                product_to_users[product_id].add(user_id)
            
            # Connect users who reviewed same product
            for users in product_to_users.values():
                for u1, u2 in combinations(users, 2):
                    edge_key = tuple(sorted([u1, u2]))
                    edges[edge_key] += 1
        
        elif node_type == 'N9':
            # Family-family edges: aggregate from N7 level
            n7_edges = EdgeFactory._build_amazon_scalar_gt(df, 'N7', df['primary_id'].tolist())
            
            product_to_family = dict(zip(df['primary_id'], df['aggregate_id']))
            
            # Aggregate to N9
            for u, v, weight in n7_edges:
                fam_u = product_to_family.get(u)
                fam_v = product_to_family.get(v)
                
                if fam_u and fam_v and fam_u != fam_v and fam_u in node_list and fam_v in node_list:
                    edge_key = tuple(sorted([fam_u, fam_v]))
                    edges[edge_key] += weight
        
        return [(u, v, w) for (u, v), w in edges.items()]
    
    @staticmethod
    def _build_history_neighbour_cooccurrence(df, node_list: List) -> List[Tuple]:
        """
        History E10b: Weighted by shared entries in structural_edges lists.
        """
        # Build node -> neighbours mapping
        node_to_neighbours = {}
        for _, row in df.iterrows():
            nid = row['primary_id']
            if nid in node_list:
                neighbours = set(row['structural_edges'] or [])
                node_to_neighbours[nid] = neighbours
        
        # Count shared neighbours for each pair
        edges = defaultdict(int)
        for n1, n2 in combinations(node_list, 2):
            neighbours1 = node_to_neighbours.get(n1, set())
            neighbours2 = node_to_neighbours.get(n2, set())
            
            shared = len(neighbours1 & neighbours2)
            if shared > 0:
                edges[(n1, n2)] = shared
        
        return [(u, v, w) for (u, v), w in edges.items()]
    
    @staticmethod
    def _build_binary_participation(df, node_type: str, node_list: List, data_manager) -> List[Tuple]:
        """
        E10c: Binary (0/1) N-N participation edges projected from the dataset's
        pre-given relational structure.

        ArXiv  N7: paper-paper if they share ≥1 author
               N8: author-author if they share ≥1 paper
               N9: category-category if any cross-category paper pair shares an author
        Amazon N7: product-product if reviewed by ≥1 same user
               N8: user-user if they reviewed ≥1 same product
               N9: family-family if any cross-family product pair shares a reviewer
        History N7: structural_edges column (already N-N)
        """
        dataset = data_manager.dataset
        node_set = set(node_list)

        if dataset == 'history':
            return data_manager.get_structural_edges(df)

        if dataset == 'arxiv':
            if node_type == 'N7':
                # paper-paper: share ≥1 author
                author_to_papers = defaultdict(set)
                for _, row in df.iterrows():
                    pid = row['primary_id']
                    if pid not in node_set:
                        continue
                    for a in (row['secondary_id'] or []):
                        author_to_papers[a].add(pid)
                seen = set()
                edges = []
                for papers in author_to_papers.values():
                    for p1, p2 in combinations(papers, 2):
                        key = (min(p1, p2), max(p1, p2))
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)
                return edges

            elif node_type == 'N8':
                # author-author: share ≥1 paper
                paper_to_authors = defaultdict(set)
                for _, row in df.iterrows():
                    for a in (row['secondary_id'] or []):
                        if a in node_set:
                            paper_to_authors[row['primary_id']].add(a)
                seen = set()
                edges = []
                for authors in paper_to_authors.values():
                    for a1, a2 in combinations(authors, 2):
                        key = (min(a1, a2), max(a1, a2))
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)
                return edges

            elif node_type == 'N9':
                # category-category: any cross-cat paper pair shares an author
                paper_to_cat = dict(zip(df['primary_id'], df['aggregate_id']))
                author_to_cats = defaultdict(set)
                for _, row in df.iterrows():
                    cat = paper_to_cat.get(row['primary_id'])
                    if cat not in node_set:
                        continue
                    for a in (row['secondary_id'] or []):
                        author_to_cats[a].add(cat)
                seen = set()
                edges = []
                for cats in author_to_cats.values():
                    cats_in = [c for c in cats if c in node_set]
                    for c1, c2 in combinations(cats_in, 2):
                        key = (min(c1, c2), max(c1, c2))
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)
                return edges

        if dataset == 'amazon':
            if node_type == 'N7':
                # product-product: reviewed by ≥1 same user
                user_to_products = defaultdict(set)
                for _, row in df.iterrows():
                    pid = row['primary_id']
                    uid = row['secondary_id']
                    if pid in node_set and uid is not None:
                        user_to_products[uid].add(pid)
                seen = set()
                edges = []
                for products in user_to_products.values():
                    for p1, p2 in combinations(products, 2):
                        key = (min(p1, p2), max(p1, p2))
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)
                return edges

            elif node_type == 'N8':
                # user-user: reviewed ≥1 same product
                product_to_users = defaultdict(set)
                for _, row in df.iterrows():
                    uid = row['secondary_id']
                    if uid in node_set:
                        product_to_users[row['primary_id']].add(uid)
                seen = set()
                edges = []
                for users in product_to_users.values():
                    for u1, u2 in combinations(users, 2):
                        key = (min(u1, u2), max(u1, u2))
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)
                return edges

            elif node_type == 'N9':
                # family-family: cross-family products share a reviewer
                product_to_fam = dict(zip(df['primary_id'], df['aggregate_id']))
                user_to_fams = defaultdict(set)
                for _, row in df.iterrows():
                    uid = row['secondary_id']
                    fam = product_to_fam.get(row['primary_id'])
                    if uid is not None and fam in node_set:
                        user_to_fams[uid].add(fam)
                seen = set()
                edges = []
                for fams in user_to_fams.values():
                    fams_in = [f for f in fams if f in node_set]
                    for f1, f2 in combinations(fams_in, 2):
                        key = (min(f1, f2), max(f1, f2))
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)
                return edges

        raise ValueError(f"E10c not defined for {dataset}")
    
    @staticmethod
    def _build_semantic_similarity(node_list: List, embeddings: np.ndarray,
                                   threshold: float = 0.75) -> List[Tuple]:
        """
        E11a: Semantic similarity edges based on cosine similarity.
        """
        # Replace NaN/inf and guard zero-norm rows
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = (embeddings / norms).astype(np.float64)

        # For unit-norm vectors cosine similarity == dot product.
        # np.errstate suppresses spurious BLAS FP-exception flags.
        with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
            sim_matrix = embeddings @ embeddings.T

        # Vectorized upper-triangle extraction (avoids O(n²) Python loop)
        i_idx, j_idx = np.where(np.triu(sim_matrix >= threshold, k=1))
        return [(node_list[i], node_list[j], float(sim_matrix[i, j]))
                for i, j in zip(i_idx, j_idx)]

    @staticmethod
    def _build_structural_similarity(node_list: List, base_graph: nx.Graph,
                                     k: int = 50) -> List[Tuple]:
        """
        E11b: Structural similarity based on degree centrality.
        Each node is connected to its k most centrality-similar neighbors.
        Top-k avoids degenerate near-complete graphs when centrality values
        cluster tightly (e.g. all-low-degree nodes in sparse history graphs).
        """
        degree_cent = nx.degree_centrality(base_graph)
        cents = np.array([degree_cent.get(n, 0.0) for n in node_list], dtype=np.float64)
        n = len(node_list)
        k = min(k, n - 1)

        diff = np.abs(cents[:, None] - cents[None, :])
        np.fill_diagonal(diff, np.inf)

        # Indices of k smallest diffs per node (not sorted, but that's fine)
        knn = np.argpartition(diff, k, axis=1)[:, :k]

        seen = set()
        edges = []
        for i in range(n):
            for j in knn[i]:
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    sim = float(max(0.0, 1.0 - diff[i, j]))
                    edges.append((node_list[i], node_list[j], sim))
        return edges

# Made with Bob
