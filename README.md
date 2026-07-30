# TAG Ontological Generalization Framework Research

Research for systematic exploration of Text-Augmented Graph (TAG) variants using an **Ontological Generalization Framework**. This system implements Factory and Builder design patterns to scale from dozen to thousands of graph learning experiments @ NJIT Fall '25, Advisor: Kristina Wicke.

Latest Poster: [Link Here](https://drive.google.com/file/d/1qWZzVyVS7GusbsD7BZXw6rxxHbufvlQ7/view?usp=sharing)

Amiri's Fall Report: [Link Here](https://drive.google.com/file/d/1dAJoczUQUu_gy-kUDrsbtnQ4bQ4pC5yo/view?usp=sharing)

Research Notebook: [Link Here](https://drive.google.com/file/d/1qH5AGSc__xBFFu91OIY_U0_JcyhC_I68/view?usp=sharing)

---

## 🏗️ Architecture

The framework follows a modular architecture with five core components:

```
┌─────────────────────────────────────────────────────────────┐
│                   ONTOLOGICAL FRAMEWORK                      │
│  Tasks [1-6] × Nodes [7-9] × Edges [10-13] × Text [a-e]    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│ Data Manager  │───▶│ TAG Constructor│───▶│   Trainer    │
│  (Backbone)   │    │   (Builder)    │    │   (Engine)   │
└───────────────┘    └────────────────┘    └──────────────┘
        │                     │                     │
        └────────────────┬────┴──────────────────┬──┘
                         ▼                       ▼
                  ┌──────────────┐      ┌──────────────┐
                  │   Analyzer   │──────│    Models    │
                  │(Meta-Analysis)│      │  (Factory)   │
                  └──────────────┘      └──────────────┘
```

### **1. Data Backbone** (`data_manager.py`)
Handles ArXiv dataset loading, preprocessing, and embedding generation.

**Key Features:**
- Efficient data subsetting for large datasets (1.7M articles)
- Multiple text fidelity levels (super, standard, mid, poor, baseline)
- Author and article embedding computation using SentenceTransformers
- Caching and serialization support

### **2. TAG Constructor** (`constructor.py`)
Builds Text-Augmented Graphs using the Builder pattern.

**Ontological Indices:**
- **Tasks [1-6]**: Categorical/Scalar × Node/Edge/Global
- **Nodes [7-9]**: Paper, Author, Journal
- **Edges [10-13]**: Coauthorship, Semantic Similarity, Citation Stats, Topic Tags
- **Text [a-e]**: Super (100%), Standard (75%), Mid (50%), Poor (25%), Baseline (title only)

### **3. GNN Models** (`models.py`)
Implements Graph Neural Networks using the Factory pattern.

**Supported Architectures:**
- GraphSAGE (default)
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)

### **4. Execution Engine** (`trainer.py`)
Manages GNN training, evaluation, and performance tracking.

**Metrics:**
- KL Divergence
- Top-1 and Top-3 Accuracy
- Cosine Similarity
- Training history and early stopping

### **5. Meta-Analysis** (`analyzer.py`)
Decision Tree-based analysis of ontological patterns.

**Features:**
- Performance band categorization (Excellent/Good/Fair/Poor)
- Decision Tree learning on ontological indices
- Feature importance analysis
- Best configuration discovery

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SocialAnalysis.git
cd SocialAnalysis

# Install dependencies
pip install -r requirements.txt

# Ensure PyTorch Geometric is properly installed
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Basic Usage

```python
from data_manager import ArxivDataManager
from constructor import TAGBuilder
from models import ModelFactory
from trainer import TrainingPipeline
from analyzer import DecisionTreeAnalyzer

# 1. Load data
data_manager = ArxivDataManager('data/arxiv_subset_10k.jsonl', subset_size=10000)
data_manager.load_data()
data_manager.extract_unique_authors()
data_manager.compute_embeddings()

# 2. Build a TAG
tag_builder = TAGBuilder(data_manager)
tag = tag_builder.build(
    task_idx=1,          # Node categorical
    node_idx=8,          # Author nodes
    edge_idx=10,         # Coauthorship edges
    text_idx='a'         # Super fidelity
)

# 3. Convert to PyG Data
embeddings = data_manager.author_embeddings
labels = ... # Your labels
data = tag.to_pyg_data(data_manager.author_list, embeddings, labels)

# 4. Train GNN
model = ModelFactory.create_model('sage', in_dim=384, hidden_dim=256, out_dim=10)
trainer = GNNTrainer(model, epochs=200)
metrics = trainer.train(data, num_classes=10)

# 5. Analyze results
analyzer = DecisionTreeAnalyzer()
analyzer.add_experiment(1, 8, 10, 'a', metrics)
analyzer.train_decision_tree()
```

### Running Full Pipeline

```bash
# Run with default configuration (2 node types × 2 edge types × 2 text fidelities × 2 thresholds = 12 experiments)
python code/main.py --data_path data/arxiv_subset_10k.jsonl --output_dir output

# Run with custom configuration
python code/main.py \
    --data_path data/arxiv_subset_10k.jsonl \
    --output_dir output \
    --model_type sage \
    --hidden_dim 256 \
    --epochs 200 \
    --lr 0.001 \
    --device cuda \
    --num_classes 10
```

---

## 📊 Ontological Space Exploration

The framework is designed to **scale from dozens to thousands** of variants. Here's how to customize the ontological space:

### Editing `main.py`

```python
config = {
    # Define which ontological indices to explore
    'task_indices': [1, 2],              # Node categorical, Node scalar
    'node_indices': [7, 8],               # Paper, Author
    'edge_indices': [10, 11],             # Coauthorship, Cosine similarity
    'text_indices': ['a', 'b', 'c', 'd', 'e'],  # All fidelity levels
    'similarity_thresholds': ['a', 'b', 'c'],   # 60%, 75%, 90%
}
```

**Example Scaling:**
- **Current**: 1 task × 2 nodes × 2 edges × 2 texts × 2 thresholds = **12 variants**
- **Expanded**: 6 tasks × 3 nodes × 4 edges × 5 texts = **360 variants**
- **Custom**: Define your own combinations for targeted exploration

---

## 🔬 Experiment Workflow

### 1. Prepare Data
```python
# Load and preprocess ArXiv data
data_manager = ArxivDataManager('data/arxiv_subset_10k.jsonl')
data_manager.load_data()
data_manager.extract_unique_authors()

# Compute embeddings with specific text fidelity
data_manager.compute_embeddings(text_fidelity='standard')  # 75% text
data_manager.save_embeddings('data/', prefix='standard_')
```

### 2. Build TAG Variants
```python
tag_builder = TAGBuilder(data_manager)

# Author coauthorship graph
tag_coauthor = tag_builder.build(1, 8, 10, 'a')

# Paper similarity graph (75% threshold)
tag_similarity = tag_builder.build(1, 7, 11, 'b', similarity_threshold='b')

# Save for later use
tag_builder.save_tag(tag_coauthor, 'output/tags/')
```

### 3. Train and Evaluate
```python
# Setup pipeline
pipeline = TrainingPipeline(
    model_factory=lambda **kw: ModelFactory.create_model('sage', **kw),
    device='cuda'
)

# Run experiment
results = pipeline.run_experiment(
    tag=tag_coauthor,
    embeddings=embeddings,
    labels=labels,
    node_list=node_list,
    num_classes=10,
    model_config={'hidden_dim': 256, 'dropout': 0.5},
    trainer_config={'lr': 0.001, 'epochs': 200}
)

print(f"Test Top-1 Accuracy: {results['test_metrics']['top1']:.3f}")
```

### 4. Meta-Analysis
```python
analyzer = DecisionTreeAnalyzer()

# Add multiple experiments
for result in all_results:
    analyzer.add_experiment(
        result['task_idx'], result['node_idx'], 
        result['edge_idx'], result['text_idx'],
        result['test_metrics']
    )

# Train Decision Tree
analyzer.train_decision_tree()

# Visualize patterns
analyzer.plot_decision_tree(save_path='output/analysis/tree.png')

# Find best configurations
best = analyzer.analyze_best_configurations(top_k=5)

# Analyze by ontology
patterns = analyzer.analyze_by_ontology()
```

---

## 📁 Project Structure

```
SocialAnalysis/
├── code/
│   ├── data_manager.py          # Data loading and preprocessing
│   ├── constructor.py           # TAG builder with ontological indices
│   ├── models.py                # GNN architectures (Factory pattern)
│   ├── trainer.py               # Training engine and pipeline
│   ├── analyzer.py              # Meta-analysis and decision trees
│   ├── main.py                  # Orchestration script
│   ├── experiments.ipynb        # Original experimental notebook
│   └── data/                    # Dataset directory
│       ├── arxiv_subset_10k.jsonl
│       ├── article_embeddings.npy
│       ├── author_embeddings.npy
│       └── author_index.json
├── output/                      # Output directory (auto-created)
│   ├── models/                  # Trained model checkpoints
│   └── analysis/                # Analysis results and plots
│       ├── decision_tree_table.csv
│       ├── decision_tree.png
│       └── all_results.json
├── requirements.txt
└── README.md
```

---

## 🧪 Example Use Cases

### Use Case 1: Compare Node Types
```python
# Author-level vs Paper-level analysis
config['node_indices'] = [7, 8]  # Paper and Author
config['edge_indices'] = [10]     # Coauthorship only
config['text_indices'] = ['a']    # Super fidelity only
# Result: 1 × 2 × 1 × 1 = 2 experiments
```

### Use Case 2: Text Fidelity Ablation
```python
# Study impact of text quality
config['task_indices'] = [1]
config['node_indices'] = [7]
config['edge_indices'] = [10]
config['text_indices'] = ['a', 'b', 'c', 'd', 'e']  # All fidelities
# Result: 1 × 1 × 1 × 5 = 5 experiments
```

### Use Case 3: Edge Type Comparison
```python
# Ground truth vs Semantic edges
config['task_indices'] = [1]
config['node_indices'] = [8]
config['edge_indices'] = [10, 11]  # Coauthorship vs Cosine
config['similarity_thresholds'] = ['a', 'b', 'c']  # All thresholds for cosine
# Result: 1 × 1 × 2 × 1 × 3 = 6 experiments (adjusted for edge 10)
```

---

## 📊 Output and Results

### Decision Tree Table Format
```
Task_Idx | Node_Idx | Edge_Idx | Text_Idx | KL     | Top1  | Top3  | Cosine | Performance_Band
---------|----------|----------|----------|--------|-------|-------|--------|------------------
1        | 8        | 10       | a        | 0.455  | 0.891 | 0.946 | 0.908  | Excellent
1        | 7        | 11       | b        | 0.512  | 0.872 | 0.935 | 0.895  | Good
...
```

### Performance Bands
- **Excellent**: Top-1 ≥ 0.90
- **Good**: Top-1 ≥ 0.80
- **Fair**: Top-1 ≥ 0.70
- **Poor**: Top-1 < 0.70

---

## 🛠️ Advanced Configuration

### Custom GNN Architecture
```python
from models import ModelFactory

class MyCustomGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        # Your custom implementation
        pass
    
    def forward(self, x, edge_index):
        # Your forward pass
        pass

# Register custom model
ModelFactory.register_model('custom', MyCustomGNN)

# Use it in experiments
config['model_type'] = 'custom'
```

### Custom Edge Type
```python
from constructor import TAGBuilder

class CustomTAGBuilder(TAGBuilder):
    def build_citation_graph(self, node_type='paper'):
        # Implement your custom edge logic
        G = nx.DiGraph()
        # ... build citation network
        return G
```

---

## 📈 Performance Optimization

### For Large Datasets (>100K articles)
1. **Use subsetting**: `subset_size=50000` during development
2. **Precompute embeddings**: Set `use_precomputed_embeddings=True`
3. **Reduce epochs**: Use `epochs=100` with `patience=20` for faster iterations
4. **GPU acceleration**: Ensure CUDA is available for PyTorch

### Memory Management
```python
# Limit batch size for large graphs
config['use_mini_batching'] = True  # Not implemented, but design consideration

# Save models selectively
config['save_models'] = False  # Only save best models
```
