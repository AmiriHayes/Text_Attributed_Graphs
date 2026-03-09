"""
Example usage script for Ontological Generalization Framework
Demonstrates basic usage patterns.
"""

import numpy as np
from pathlib import Path

# Import framework modules
from data_manager import ArxivDataManager
from constructor import TAGBuilder, TAGConfiguration
from models import ModelFactory
from trainer import GNNTrainer
from analyzer import DecisionTreeAnalyzer


def example_basic_workflow():
    """Example: Basic workflow with a single TAG."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Workflow")
    print("="*80)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    data_manager = ArxivDataManager('code/data/arxiv_subset_10k.jsonl', subset_size=1000)
    data_manager.load_data()
    data_manager.extract_unique_authors()
    
    # Step 2: Compute embeddings
    print("\n2. Computing embeddings...")
    data_manager.compute_embeddings(text_fidelity='super')
    
    # Step 3: Build a TAG
    print("\n3. Building TAG...")
    tag_builder = TAGBuilder(data_manager)
    tag = tag_builder.build(
        task_idx=1,         # Node categorical
        node_idx=8,         # Author
        edge_idx=10,        # Coauthorship
        text_idx='a'        # Super fidelity
    )
    
    # Step 4: Prepare data for training
    print("\n4. Preparing training data...")
    embeddings = data_manager.author_embeddings
    node_list = data_manager.author_list
    
    # Get real category labels for authors (based on their papers)
    num_classes = data_manager.num_classes
    labels = data_manager.get_author_category_labels(num_classes=num_classes)
    
    # Convert to PyG data
    data = tag.to_pyg_data(node_list, embeddings, labels, device='cpu')
    
    # Step 5: Train GNN
    print("\n5. Training GNN...")
    model = ModelFactory.create_model('sage', in_dim=384, hidden_dim=64, out_dim=num_classes, dropout=0.3)
    trainer = GNNTrainer(model, device='cpu', epochs=20, patience=10)
    metrics = trainer.train(data, num_classes, verbose=True)
    
    print(f"\n✅ Final Results:")
    print(f"   Top-1 Accuracy: {metrics['top1']:.3f}")
    print(f"   KL Divergence: {metrics['kl']:.4f}")


def example_multiple_tags():
    """Example: Compare multiple TAG variants."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Compare Multiple TAG Variants")
    print("="*80)
    
    # Setup
    data_manager = ArxivDataManager('code/data/arxiv_subset_10k.jsonl', subset_size=1000)
    data_manager.load_data()
    data_manager.extract_unique_authors()
    data_manager.compute_embeddings()
    
    tag_builder = TAGBuilder(data_manager)
    analyzer = DecisionTreeAnalyzer()
    
    # Define TAG variants to test
    variants = [
        {'task_idx': 1, 'node_idx': 8, 'edge_idx': 10, 'text_idx': 'a', 'name': 'Author-Coauthorship-Super'},
        {'task_idx': 1, 'node_idx': 8, 'edge_idx': 10, 'text_idx': 'b', 'name': 'Author-Coauthorship-Standard'},
        {'task_idx': 1, 'node_idx': 8, 'edge_idx': 11, 'text_idx': 'a', 'name': 'Author-Similarity-Super', 'threshold': 'b'},
    ]
    
    # Test each variant
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Testing: {variant['name']}")
        print(f"{'='*60}")
        
        # Build TAG
        tag = tag_builder.build(
            task_idx=variant['task_idx'],
            node_idx=variant['node_idx'],
            edge_idx=variant['edge_idx'],
            text_idx=variant['text_idx'],
            similarity_threshold=variant.get('threshold')
        )
        
        # Prepare data
        embeddings = data_manager.author_embeddings
        node_list = data_manager.author_list
        num_classes = 5
        labels = np.random.rand(len(node_list), num_classes).astype(np.float32)
        labels = labels / labels.sum(axis=1, keepdims=True)
        
        data = tag.to_pyg_data(node_list, embeddings, labels, device='cpu')
        
        # Train
        model = ModelFactory.create_model('sage', in_dim=384, hidden_dim=64, out_dim=num_classes)
        trainer = GNNTrainer(model, device='cpu', epochs=20, patience=10)
        metrics = trainer.train(data, num_classes, verbose=False)
        
        # Add to analyzer
        analyzer.add_experiment(
            variant['task_idx'], variant['node_idx'],
            variant['edge_idx'], variant['text_idx'],
            metrics
        )
        
        print(f"Results: Top-1={metrics['top1']:.3f}, KL={metrics['kl']:.4f}")
    
    # Analyze results
    print(f"\n{'='*80}")
    print("Meta-Analysis")
    print(f"{'='*80}")
    analyzer.build_decision_tree_table()
    analyzer.analyze_best_configurations(top_k=3)


def example_custom_model():
    """Example: Using a custom GNN model."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom GNN Model")
    print("="*80)
    
    # Setup data
    data_manager = ArxivDataManager('code/data/arxiv_subset_10k.jsonl', subset_size=500)
    data_manager.load_data()
    data_manager.extract_unique_authors()
    data_manager.compute_embeddings()
    
    # Build TAG
    tag_builder = TAGBuilder(data_manager)
    tag = tag_builder.build(1, 8, 10, 'a')
    
    # Prepare data
    embeddings = data_manager.author_embeddings
    node_list = data_manager.author_list
    num_classes = 5
    labels = np.random.rand(len(node_list), num_classes).astype(np.float32)
    labels = labels / labels.sum(axis=1, keepdims=True)
    data = tag.to_pyg_data(node_list, embeddings, labels, device='cpu')
    
    # Try different model types
    model_types = ['sage', 'gcn', 'gat']
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Testing {model_type.upper()} model")
        print(f"{'='*60}")
        
        model = ModelFactory.create_model(
            model_type=model_type,
            in_dim=384,
            hidden_dim=64,
            out_dim=num_classes,
            dropout=0.3
        )
        
        trainer = GNNTrainer(model, device='cpu', epochs=20, patience=10)
        metrics = trainer.train(data, num_classes, verbose=False)
        
        print(f"Results: Top-1={metrics['top1']:.3f}, Top-3={metrics['top3']:.3f}")


def example_save_load():
    """Example: Save and load TAGs and models."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Save and Load")
    print("="*80)
    
    output_dir = Path('output/examples')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup
    data_manager = ArxivDataManager('code/data/arxiv_subset_10k.jsonl', subset_size=500)
    data_manager.load_data()
    data_manager.extract_unique_authors()
    data_manager.compute_embeddings()
    
    # Save embeddings
    print("\n1. Saving embeddings...")
    data_manager.save_embeddings(str(output_dir), prefix='example_')
    
    # Build and save TAG
    print("\n2. Building and saving TAG...")
    tag_builder = TAGBuilder(data_manager)
    tag = tag_builder.build(1, 8, 10, 'a')
    tag_builder.save_tag(tag, str(output_dir))
    
    # Train and save model
    print("\n3. Training and saving model...")
    embeddings = data_manager.author_embeddings
    node_list = data_manager.author_list
    num_classes = 5
    labels = np.random.rand(len(node_list), num_classes).astype(np.float32)
    labels = labels / labels.sum(axis=1, keepdims=True)
    data = tag.to_pyg_data(node_list, embeddings, labels, device='cpu')
    
    model = ModelFactory.create_model('sage', in_dim=384, hidden_dim=64, out_dim=num_classes)
    trainer = GNNTrainer(model, device='cpu', epochs=10)
    trainer.train(data, num_classes, verbose=False)
    trainer.save_model(str(output_dir / 'example_model.pt'))
    
    # Load everything back
    print("\n4. Loading saved data...")
    new_data_manager = ArxivDataManager('code/data/arxiv_subset_10k.jsonl', subset_size=500)
    new_data_manager.load_data()
    new_data_manager.extract_unique_authors()
    new_data_manager.load_embeddings(str(output_dir), prefix='example_')
    
    tag_builder2 = TAGBuilder(new_data_manager)
    loaded_tag = tag_builder2.load_tag(str(output_dir / 'TAG_T1_N8_E10_Xa.json'))
    
    print(f"\n✅ Loaded TAG: {loaded_tag.get_identifier()}")
    print(f"   Nodes: {loaded_tag.graph.number_of_nodes()}")
    print(f"   Edges: {loaded_tag.graph.number_of_edges()}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("ONTOLOGICAL GENERALIZATION FRAMEWORK - EXAMPLES")
    print("="*80)
    
    # Run examples
    # Comment out any examples you don't want to run
    
    # example_basic_workflow()
    # example_multiple_tags()
    # example_custom_model()
    example_save_load()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80 + "\n")
