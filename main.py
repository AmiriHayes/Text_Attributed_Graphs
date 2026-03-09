"""
Main Orchestration Script for Ontological Generalization Framework
Demonstrates end-to-end pipeline for TAG experiments.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
import torch

# Import framework modules
from data_manager import ArxivDataManager
from constructor import TAGBuilder, TAGConfiguration
from models import ModelFactory
from trainer import TrainingPipeline, PerformanceTracker
from analyzer import DecisionTreeAnalyzer


class OntologicalExperimentPipeline:
    """
    Complete pipeline for running ontological experiments.
    Demonstrates scalable iteration through TAG variants.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary with all parameters
        """
        self.config = config
        self.data_manager = None
        self.tag_builder = None
        self.analyzer = DecisionTreeAnalyzer()
        self.performance_tracker = PerformanceTracker()
        
    def setup_data(self):
        """Load and prepare data."""
        print("\n" + "="*80)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("="*80)
        
        # Initialize data manager
        self.data_manager = ArxivDataManager(
            # data_path=self.config['data_path'],
            data_path="data/arxiv_subset_10k.jsonl",
            subset_size=self.config.get('subset_size', None)
        )
        
        # Load data
        self.data_manager.load_data()
        self.data_manager.extract_unique_authors()
        
        # Compute or load embeddings
        if self.config.get('use_precomputed_embeddings', False):
            self.data_manager.load_embeddings(
                input_dir=self.config['embedding_dir'],
                prefix=self.config.get('embedding_prefix', '')
            )
        else:
            # Compute embeddings with default text fidelity
            self.data_manager.compute_embeddings(
                model_name=self.config.get('sbert_model', 'all-MiniLM-L6-v2'),
                text_fidelity='super',  # Will recompute for different fidelities
                force_recompute=self.config.get('force_recompute', False)
            )
            
            # Save embeddings
            if self.config.get('save_embeddings', True):
                self.data_manager.save_embeddings(
                    output_dir=self.config['embedding_dir'],
                    prefix=self.config.get('embedding_prefix', '')
                )
        
        # Initialize TAG builder
        self.tag_builder = TAGBuilder(self.data_manager)
        
        print("\n✅ Data setup complete\n")
    
    def define_ontological_space(self) -> List[Dict]:
        """
        Define the ontological space to iterate through.
        Returns list of experiment configurations.
        
        This is where you scale from 23 variants to thousands by
        specifying which combinations to explore.
        """
        configs = TAGConfiguration()
        experiment_space = []
        
        # Define which indices to iterate through
        task_indices = self.config.get('task_indices', [1])  # Default: node_categorical
        node_indices = self.config.get('node_indices', [7, 8])  # paper, author
        edge_indices = self.config.get('edge_indices', [10, 11])  # coauthorship, cosine
        text_indices = self.config.get('text_indices', ['a', 'b', 'c'])  # super, standard, mid
        similarity_thresholds = self.config.get('similarity_thresholds', ['b'])  # 75%
        
        # Generate all combinations
        for task_idx in task_indices:
            for node_idx in node_indices:
                for edge_idx in edge_indices:
                    for text_idx in text_indices:
                        # For cosine similarity, include threshold variants
                        if edge_idx == 11:  # cosine_similarity
                            for threshold in similarity_thresholds:
                                experiment_space.append({
                                    'task_idx': task_idx,
                                    'node_idx': node_idx,
                                    'edge_idx': edge_idx,
                                    'text_idx': text_idx,
                                    'similarity_threshold': threshold
                                })
                        else:
                            experiment_space.append({
                                'task_idx': task_idx,
                                'node_idx': node_idx,
                                'edge_idx': edge_idx,
                                'text_idx': text_idx,
                                'similarity_threshold': None
                            })
        
        print(f"\n{'='*80}")
        print(f"ONTOLOGICAL EXPERIMENT SPACE: {len(experiment_space)} variants")
        print(f"{'='*80}")
        print(f"Tasks: {task_indices}")
        print(f"Nodes: {node_indices}")
        print(f"Edges: {edge_indices}")
        print(f"Text Fidelities: {text_indices}")
        print(f"Similarity Thresholds: {similarity_thresholds}")
        print(f"{'='*80}\n")
        
        return experiment_space
    
    def run_experiment(self, exp_config: Dict) -> Dict:
        """
        Run a single experiment.
        
        Args:
            exp_config: Experiment configuration
        
        Returns:
            Results dictionary
        """
        # Build TAG
        tag = self.tag_builder.build(
            task_idx=exp_config['task_idx'],
            node_idx=exp_config['node_idx'],
            edge_idx=exp_config['edge_idx'],
            text_idx=exp_config['text_idx'],
            similarity_threshold=exp_config['similarity_threshold']
        )
        
        # Get appropriate embeddings and labels based on node type
        # Use dynamic num_classes from data_manager
        num_classes = self.data_manager.num_classes or self.config.get('num_classes', 10)
        
        if exp_config['node_idx'] == 8:  # author
            embeddings = self.data_manager.author_embeddings
            node_list = self.data_manager.author_list
            # For authors, use most common category from their papers
            labels = self.data_manager.get_author_category_labels(num_classes=num_classes)
        else:  # paper (7) or journal (9)
            embeddings = self.data_manager.article_embeddings
            node_list = list(self.data_manager.df['id'])
            labels = self.data_manager.get_category_labels(num_classes=num_classes)
        
        # Normalize embeddings
        from sklearn.preprocessing import normalize
        embeddings = normalize(embeddings, axis=1).astype(np.float32)
        
        # Setup training pipeline
        def model_factory(**kwargs):
            return ModelFactory.create_model(
                model_type=self.config.get('model_type', 'sage'),
                **kwargs
            )
        
        pipeline = TrainingPipeline(
            model_factory=model_factory,
            device=self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Run experiment
        model_config = {
            'hidden_dim': self.config.get('hidden_dim', 256),
            'dropout': self.config.get('dropout', 0.5)
        }
        
        trainer_config = {
            'lr': self.config.get('lr', 0.001),
            'weight_decay': self.config.get('weight_decay', 5e-4),
            'epochs': self.config.get('epochs', 200),
            'patience': self.config.get('patience', 50)
        }
        
        # Save model path
        save_model_path = None
        if self.config.get('save_models', False):
            save_model_path = str(Path(self.config['output_dir']) / 'models' / f"model_{tag.get_identifier()}.pt")
            Path(save_model_path).parent.mkdir(exist_ok=True, parents=True)
        
        results = pipeline.run_experiment(
            tag=tag,
            embeddings=embeddings,
            labels=labels,
            node_list=node_list,
            num_classes=num_classes,
            model_config=model_config,
            trainer_config=trainer_config,
            save_model_path=save_model_path,
            save_training_history=self.config.get('save_training_history', False)
        )
        
        return results
    
    def run_all_experiments(self):
        """Run all experiments in the ontological space."""
        print("\n" + "="*80)
        print("STEP 2: RUNNING ONTOLOGICAL EXPERIMENTS")
        print("="*80)
        
        # Define experiment space
        experiment_space = self.define_ontological_space()
        
        # Run experiments
        for i, exp_config in enumerate(experiment_space):
            print(f"\n{'#'*80}")
            print(f"EXPERIMENT {i+1}/{len(experiment_space)}")
            print(f"{'#'*80}")
            
            try:
                results = self.run_experiment(exp_config)
                
                # Track results
                self.performance_tracker.add_result(results)
                
                # Add to analyzer
                self.analyzer.add_experiment(
                    task_idx=exp_config['task_idx'],
                    node_idx=exp_config['node_idx'],
                    edge_idx=exp_config['edge_idx'],
                    text_idx=exp_config['text_idx'],
                    test_metrics=results['test_metrics']
                )
                
            except Exception as e:
                print(f"❌ Experiment failed: {e}")
                continue
        
        print("\n✅ All experiments complete\n")
    
    def analyze_results(self):
        """Perform meta-analysis on results."""
        print("\n" + "="*80)
        print("STEP 3: META-ANALYSIS")
        print("="*80)
        
        # Build Decision Tree Table
        self.analyzer.build_decision_tree_table()
        
        # Train Decision Tree
        if len(self.analyzer.dt_table) >= 5:
            dt_results = self.analyzer.train_decision_tree(
                target_metric='Performance_Band',
                max_depth=self.config.get('dt_max_depth', 5),
                min_samples_split=self.config.get('dt_min_samples_split', 5)
            )
            
            # Plot Decision Tree
            if self.config.get('plot_decision_tree', True):
                plot_path = Path(self.config['output_dir']) / 'analysis' / 'decision_tree.png'
                plot_path.parent.mkdir(exist_ok=True, parents=True)
                self.analyzer.plot_decision_tree(save_path=str(plot_path))
        
        # Analyze best configurations
        self.analyzer.analyze_best_configurations(top_k=self.config.get('top_k', 5))
        
        # Analyze by ontology
        self.analyzer.analyze_by_ontology()
        
        # Save analysis
        analysis_dir = Path(self.config['output_dir']) / 'analysis'
        self.analyzer.save_analysis(str(analysis_dir))
        
        # Save performance tracker
        results_path = analysis_dir / 'all_results.json'
        self.performance_tracker.save_results(str(results_path))
        
        print("\n✅ Meta-analysis complete\n")
    
    def run(self):
        """Execute the complete pipeline."""
        print("\n" + "="*80)
        print("ONTOLOGICAL GENERALIZATION FRAMEWORK")
        print("Production-Grade TAG Experiment Pipeline")
        print("="*80)
        
        # Step 1: Setup data
        self.setup_data()
        
        # Step 2: Run experiments
        self.run_all_experiments()
        
        # Step 3: Analyze results
        self.analyze_results()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Ontological Generalization Framework')
    parser.add_argument('--data_path', type=str, default='data\arxiv_subset_10k.jsonl',
                        help='Path to ArXiv JSONL file')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for results')
    parser.add_argument('--embedding_dir', type=str, default='data',
                        help='Directory for embeddings')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Number of articles to use (None = all)')
    parser.add_argument('--use_precomputed', action='store_true',
                        help='Use precomputed embeddings')
    parser.add_argument('--model_type', type=str, default='sage',
                        choices=['sage', 'gcn', 'gat'],
                        help='GNN model type')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Build configuration
    config = {
        # Data
        'data_path': args.data_path,
        'subset_size': args.subset_size,
        'embedding_dir': args.embedding_dir,
        'use_precomputed_embeddings': args.use_precomputed,
        'save_embeddings': True,
        
        # Ontological space
        'task_indices': [1],  # node_categorical 
        'node_indices': [7, 8],  # paper, author
        'edge_indices': [10, 11],  # coauthorship, cosine
        'text_indices': ['a', 'b'],  # super, standard
        'similarity_thresholds': ['b', 'c'],  # 75%, 90%
        
        # Model
        'model_type': args.model_type,
        'hidden_dim': args.hidden_dim,
        'dropout': 0.5,
        'num_classes': args.num_classes,
        
        # Training
        'lr': args.lr,
        'weight_decay': 5e-4,
        'epochs': args.epochs,
        'patience': 50,
        'device': device,
        
        # Output
        'output_dir': args.output_dir,
        'save_models': True,  # Set to True to save trained models        'save_training_history': True,  # Set to True to save training curves        
        'dt_max_depth': 8,
        'dt_min_samples_split': 8,
        'plot_decision_tree': True,
        'top_k': 5,
    }
    
    # Create and run pipeline
    pipeline = OntologicalExperimentPipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()