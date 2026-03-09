"""
Data Backbone Module for Ontological Generalization Framework
Handles ArXiv data loading, subsetting, and preprocessing.
"""

import json
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class ArxivDataManager:
    """
    Data Backbone: Manages ArXiv dataset with efficient subsetting.
    Supports multiple text fidelity levels and embedding generation.
    """
    
    def __init__(self, data_path: str, subset_size: Optional[int] = None):
        """
        Initialize the data manager.
        
        Args:
            data_path: Path to ArXiv JSONL file
            subset_size: Number of articles to load (None = all)
        """
        self.data_path = Path(data_path)
        self.subset_size = subset_size
        self.df = None
        self.author_list = None
        self.article_embeddings = None
        self.author_embeddings = None
        self.embedder = None
        self.cat2id = None
        self.num_classes = None
        self.author_categories = None
        
    def load_data(self) -> pd.DataFrame:
        """Load ArXiv data with optional subsetting."""
        print(f"Loading data from {self.data_path}...")
        if self.subset_size:
            self.df = pd.read_json(self.data_path, lines=True, nrows=self.subset_size)
        else:
            self.df = pd.read_json(self.data_path, lines=True)
        
        # Parse authors_parsed field
        self.df['authors_parsed'] = self.df['authors_parsed'].apply(self._parse_authors_field)
        self.df['n_authors'] = self.df['authors_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Parse categories and create category_broad labels
        print("Parsing categories...")
        self.df["cat_top"] = self.df["categories"].apply(lambda c: str(c).split()[0].split(".")[0].strip())
        
        # Create a mapping: top-level category → integer ID
        unique_cats = sorted(self.df["cat_top"].unique())
        self.cat2id = {c: i for i, c in enumerate(unique_cats)}
        self.num_classes = len(unique_cats)
        
        print(f"Category mapping: {self.cat2id}")
        print(f"NUM_CLASSES = {self.num_classes}")
        
        # Assign numeric labels
        self.df["category_broad"] = self.df["cat_top"].map(self.cat2id)
        
        print(f"✅ Loaded {len(self.df)} articles with {self.num_classes} categories")
        print(f"Category distribution:\n{self.df['cat_top'].value_counts().head(10)}")
        
        return self.df
    
    @staticmethod
    def _parse_authors_field(val):
        """Convert stringified authors_parsed field to Python list."""
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            for parser in (ast.literal_eval, json.loads):
                try:
                    parsed = parser(val)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    continue
            return []
        return []
    
    @staticmethod
    def handle_author(author_parsed_instance: List) -> str:
        """Convert ['Ortega-Cerda', 'Joaquim', ''] -> 'Joaquim|Ortega-Cerda'"""
        try:
            first = author_parsed_instance[1].strip() if len(author_parsed_instance) > 1 else ''
            last = author_parsed_instance[0].strip() if len(author_parsed_instance) > 0 else ''
            return f"{first}|{last}"
        except Exception:
            return ""
    
    def extract_unique_authors(self) -> set:
        """Extract unique authors from the dataset and compute their category labels."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        unique_authors = set()
        author_paper_categories = defaultdict(list)
        
        for row in tqdm(self.df.itertuples(index=False), total=len(self.df), desc="Extracting authors"):
            authors = getattr(row, 'authors_parsed', [])
            if not isinstance(authors, list):
                continue
            
            # Get the category for this paper
            cat_broad = getattr(row, 'category_broad', None)
            
            for a in authors:
                author_clean = self.handle_author(a)
                if author_clean and "|" in author_clean:
                    unique_authors.add(author_clean)
                    if cat_broad is not None:
                        author_paper_categories[author_clean].append(cat_broad)
        
        self.author_list = sorted(list(unique_authors))
        
        # Compute most common category for each author
        print("Computing author category labels...")
        self.author_categories = {}
        for author in self.author_list:
            if author in author_paper_categories and len(author_paper_categories[author]) > 0:
                # Assign most common category
                counter = Counter(author_paper_categories[author])
                self.author_categories[author] = counter.most_common(1)[0][0]
            else:
                # If no papers, assign -1 (will be filtered out)
                self.author_categories[author] = -1
        
        authors_with_labels = sum(1 for cat in self.author_categories.values() if cat >= 0)
        print(f"✅ Found {len(self.author_list)} unique authors")
        print(f"✅ {authors_with_labels} authors have category labels")
        
        return unique_authors
    
    def generate_text_fidelity_variant(self, text: str, fidelity: str) -> str:
        """
        Generate text variants based on fidelity level.
        
        Args:
            text: Original text (title + abstract)
            fidelity: One of ['super', 'standard', 'mid', 'poor', 'baseline']
        
        Returns:
            Modified text according to fidelity level
        """
        if fidelity == 'super':
            return text  # Full text
        elif fidelity == 'standard':
            # Use first 75% of words
            words = text.split()
            return ' '.join(words[:int(len(words) * 0.75)])
        elif fidelity == 'mid':
            # Use first 50% of words
            words = text.split()
            return ' '.join(words[:int(len(words) * 0.5)])
        elif fidelity == 'poor':
            # Use first 25% of words
            words = text.split()
            return ' '.join(words[:int(len(words) * 0.25)])
        elif fidelity == 'baseline':
            # Only title (no abstract)
            return text.split('.')[0] if '.' in text else text[:50]
        else:
            raise ValueError(f"Unknown fidelity: {fidelity}")
    
    def compute_embeddings(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        text_fidelity: str = 'super',
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute article and author embeddings.
        
        Args:
            model_name: SentenceTransformer model name
            text_fidelity: Text fidelity level (12a-12e)
            force_recompute: Recompute even if embeddings exist
        
        Returns:
            (article_embeddings, author_embeddings)
        """
        if self.df is None or self.author_list is None:
            raise ValueError("Load data and extract authors first")
        
        # Initialize embedder
        if self.embedder is None or force_recompute:
            print(f"Loading embedding model: {model_name}")
            self.embedder = SentenceTransformer(model_name)
        
        # Compute article embeddings
        print(f"Computing embeddings with text fidelity: {text_fidelity}")
        article_embeddings = np.zeros((len(self.df), 384))
        author_sums = defaultdict(lambda: np.zeros(384))
        author_counts = defaultdict(int)
        
        for i, row in enumerate(tqdm(self.df.itertuples(index=False), total=len(self.df), desc="Computing embeddings")):
            authors = getattr(row, 'authors_parsed', [])
            if not isinstance(authors, list) or len(authors) == 0:
                continue
            
            # Combine title + abstract
            text_parts = []
            if hasattr(row, 'title') and isinstance(row.title, str):
                text_parts.append(row.title)
            if hasattr(row, 'abstract') and isinstance(row.abstract, str):
                text_parts.append(row.abstract)
            if not text_parts:
                continue
            
            combined_text = " ".join(text_parts)
            # Apply text fidelity transformation
            combined_text = self.generate_text_fidelity_variant(combined_text, text_fidelity)
            
            # Compute embedding
            emb = self.embedder.encode(combined_text, show_progress_bar=False)
            article_embeddings[i] = emb
            
            # Accumulate for author embeddings
            for a in authors:
                author_id = self.handle_author(a)
                if not author_id:
                    continue
                author_sums[author_id] += emb
                author_counts[author_id] += 1
        
        # Compute author embeddings (average of their papers)
        author_embeddings = np.zeros((len(self.author_list), 384))
        for idx, author_id in enumerate(self.author_list):
            count = author_counts[author_id]
            if count > 0:
                author_embeddings[idx] = author_sums[author_id] / count
        
        self.article_embeddings = article_embeddings
        self.author_embeddings = author_embeddings
        
        print(f"✅ Article embeddings: {article_embeddings.shape}")
        print(f"✅ Author embeddings: {author_embeddings.shape}")
        
        return article_embeddings, author_embeddings
    
    def save_embeddings(self, output_dir: str, prefix: str = ""):
        """Save computed embeddings and author index."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        if self.article_embeddings is not None:
            np.save(output_path / f"{prefix}article_embeddings.npy", self.article_embeddings)
        if self.author_embeddings is not None:
            np.save(output_path / f"{prefix}author_embeddings.npy", self.author_embeddings)
        if self.author_list is not None:
            with open(output_path / f"{prefix}author_index.json", "w") as f:
                json.dump(self.author_list, f)
        
        print(f"✅ Saved embeddings to {output_path}")
    
    def load_embeddings(self, input_dir: str, prefix: str = ""):
        """Load precomputed embeddings."""
        input_path = Path(input_dir)
        
        self.article_embeddings = np.load(input_path / f"{prefix}article_embeddings.npy")
        self.author_embeddings = np.load(input_path / f"{prefix}author_embeddings.npy")
        with open(input_path / f"{prefix}author_index.json", "r") as f:
            self.author_list = json.load(f)
        
        print(f"✅ Loaded embeddings from {input_path}")
        return self.article_embeddings, self.author_embeddings
    
    def get_category_labels(self, num_classes: int = None) -> np.ndarray:
        """
        Extract category labels for papers (one-hot encoded).
        
        Args:
            num_classes: Number of classes (uses self.num_classes if None)
        
        Returns:
            One-hot encoded labels (N x num_classes)
        """
        if self.df is None:
            raise ValueError("Data not loaded")
        
        if num_classes is None:
            if self.num_classes is None:
                raise ValueError("No categories found. Ensure load_data() was called.")
            num_classes = self.num_classes
        
        N = len(self.df)
        Y = np.zeros((N, num_classes), dtype=np.float32)
        
        if 'category_broad' in self.df.columns:
            for i, row in enumerate(self.df.itertuples(index=False)):
                cat = getattr(row, "category_broad", None)
                if cat is not None and 0 <= int(cat) < num_classes:
                    Y[i, int(cat)] = 1.0
        
        print(f"✅ Generated paper labels: {Y.shape}, {Y.sum()} nodes with labels")
        return Y
    
    def get_author_category_labels(self, num_classes: int = None) -> np.ndarray:
        """
        Get category labels for authors based on their most common paper category.
        
        Args:
            num_classes: Number of classes (uses self.num_classes if None)
        
        Returns:
            One-hot encoded labels (N_authors x num_classes)
        """
        if self.author_list is None or self.author_categories is None:
            raise ValueError("Authors not extracted. Call extract_unique_authors() first.")
        
        if num_classes is None:
            if self.num_classes is None:
                raise ValueError("No categories found. Ensure load_data() was called.")
            num_classes = self.num_classes
        
        N = len(self.author_list)
        Y = np.zeros((N, num_classes), dtype=np.float32)
        
        for i, author in enumerate(self.author_list):
            cat = self.author_categories.get(author, -1)
            if cat is not None and 0 <= int(cat) < num_classes:
                Y[i, int(cat)] = 1.0
        
        print(f"✅ Generated author labels: {Y.shape}, {Y.sum()} nodes with labels")
        return Y