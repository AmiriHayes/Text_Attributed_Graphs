"""
Main Orchestration Script for Ontological Generalization Framework
Demonstrates end-to-end pipeline for TAG experiments.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import torch
import math
import csv
from collections import Counter
from sklearn.preprocessing import normalize

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

        # Build TAG
        tag = self.tag_builder.build(
            task_idx=exp_config['task_idx'],
            node_idx=exp_config['node_idx'],
            edge_idx=exp_config['edge_idx'],
            text_idx=exp_config['text_idx'],
            similarity_threshold=exp_config['similarity_threshold']
        )

        # Get embeddings and labels based on node type
        num_classes = self.data_manager.num_classes or self.config.get('num_classes', 10)

        if exp_config['node_idx'] == 8:  # author
            embeddings = self.data_manager.author_embeddings
            node_list = self.data_manager.author_list
            labels = self.data_manager.get_author_category_labels(num_classes=num_classes)
        else:  # paper (7) or journal (9)
            embeddings = self.data_manager.article_embeddings
            node_list = list(self.data_manager.df['id'])
            labels = self.data_manager.get_category_labels(num_classes=num_classes)

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

        # Compute and attach stats
        G = tag.graph
        all_words = []
        for row in self.data_manager.df.itertuples():
            text = f"{getattr(row, 'title', '') or ''} {getattr(row, 'abstract', '') or ''}"
            all_words.extend(text.lower().split())

        counts = Counter(all_words)
        total = sum(counts.values())
        text_vocab_entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())

        label_counts = self.data_manager.df['cat_top'].value_counts()
        label_probs = label_counts / label_counts.sum()
        label_balance_entropy = -sum(p * math.log2(p) for p in label_probs)

        avg_text_length = np.mean([
            len(str(getattr(row, 'title', '') or '')) + len(str(getattr(row, 'abstract', '') or ''))
            for row in self.data_manager.df.itertuples()
        ])

        results['stats'] = {
            'number_of_nodes': G.number_of_nodes(),
            'avg_text_length': avg_text_length,
            'text_vocab_entropy': text_vocab_entropy,
            'label_balance_entropy': label_balance_entropy,
            'task_type': 'node_categorical',
            'output_dimension': num_classes,
            'normalized_score': results['test_metrics']['top1'],
        }

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
                self.performance_tracker.add_result(results)

                stream_path = Path(self.config['output_dir']) / 'analysis' / 'results_stream.csv'
                stream_path.parent.mkdir(exist_ok=True, parents=True)
                row = {
                    'Task_Idx': exp_config['task_idx'],
                    'Node_Idx': exp_config['node_idx'],
                    'Edge_Idx': exp_config['edge_idx'],
                    'Text_Idx': exp_config['text_idx'],
                    **results['test_metrics'],
                    **results.get('stats', {})
                }
                write_header = not stream_path.exists()
                with open(stream_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

                self.analyzer.add_experiment(
                    task_idx=exp_config['task_idx'],
                    node_idx=exp_config['node_idx'],
                    edge_idx=exp_config['edge_idx'],
                    text_idx=exp_config['text_idx'],
                    test_metrics=results['test_metrics'],
                    stats=results.get('stats', None)   # <-- add this
                )
                
            except Exception as e:
                print(f"Experiment failed: {e}")
                continue
        
            # if i + 1 == 23:
            #     print("\n⏹ Stopping after experiment 23 as planned.")
            #     break
        
        print("\n✅ All experiments complete\n")
    
    def analyze_results(self):
        """Perform meta-analysis on results."""
        print("\n" + "="*80)
        print("STEP 3: META-ANALYSIS")
        print("="*80)
        
        # Build Decision Tree Table
        self.analyzer.build_decision_tree_table()
        
        # Analyze best configurations
        self.analyzer.analyze_best_configurations(top_k=self.config.get('top_k', 5))
        
        # Analyze by ontology
        self.analyzer.analyze_by_ontology()
        
        # Save analysis and results FIRST before anything that can crash
        analysis_dir = Path(self.config['output_dir']) / 'analysis'
        self.analyzer.save_analysis(str(analysis_dir))
        results_path = analysis_dir / 'all_results.json'
        self.performance_tracker.save_results(str(results_path))
        
        # Train Decision Tree
        if len(self.analyzer.dt_table) >= 5:
            dt_results = self.analyzer.train_decision_tree(
                target_metric='Performance_Band',
                max_depth=self.config.get('dt_max_depth', 5),
                min_samples_split=self.config.get('dt_min_samples_split', 5)
            )
            
            # Plot Decision Tree
            if self.config.get('plot_decision_tree', True):
                try:
                    plot_path = Path(self.config['output_dir']) / 'analysis' / 'decision_tree.png'
                    self.analyzer.plot_decision_tree(save_path=str(plot_path))
                except Exception as e:
                    print(f"⚠ Decision tree plot failed: {e}")
        
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


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        # Data
        'data_path': r'data\arxiv_subset_10k.jsonl',
        'subset_size': None,                  # int to load a subset, None = all
        'embedding_dir': 'data',              # folder where .npy files are saved/loaded
        'use_precomputed_embeddings': True,
        'save_embeddings': True,

        # Ontological space
        'task_indices': [1],                  # node_categorical
        'node_indices': [7, 8],               # paper, author
        'edge_indices': [10, 11],             # coauthorship, cosine
        'text_indices': ['a', 'b', 'c', 'd', 'e'],          # super, standard
        'similarity_thresholds': ['a', 'b', 'c'],           # 75%, 90%    

        # Model
        'model_type': 'sage',                 # sage | gcn | gat
        'hidden_dim': 256,
        'dropout': 0.5,

        # Training
        'lr': 0.001,
        'weight_decay': 5e-4,
        'epochs': 200,
        'patience': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # Output
        'output_dir': f'output/run_{run_id}',
        'save_models': True,
        'save_training_history': True,
        'dt_max_depth': 8,
        'dt_min_samples_split': 8,
        'plot_decision_tree': True,
        'top_k': 5,
    }

    pipeline = OntologicalExperimentPipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()