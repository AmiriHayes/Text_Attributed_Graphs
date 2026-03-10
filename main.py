"""
Main Orchestration Script for Ontological Generalization Framework
Demonstrates end-to-end pipeline for TAG experiments.
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import torch
import math
import csv
import json
from collections import Counter
from sklearn.preprocessing import normalize
from tqdm import tqdm

# Import framework modules
from data_manager import ArxivDataManager
from constructor import TAGBuilder, TAGConfiguration
from models import ModelFactory
from trainer import TrainingPipeline, PerformanceTracker
from analyzer import DecisionTreeAnalyzer


def generate_random_subsets(
    json_path: str,
    n_subsets: int = 8,
    subset_size: int = 10000,
    output_dir: str = 'data/subsets',
    seed: int = 42
) -> List[str]:
    """
    Generate n random non-overlapping subsets from the full arxiv snapshot.
    Saves each as a separate JSONL file with its own embedding directory.
    Single pass through the source file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if subsets already exist — skip generation if so
    existing = [os.path.join(output_dir, f'subset_{i:02d}.jsonl') for i in range(n_subsets)]
    if all(os.path.exists(p) for p in existing):
        print(f"✅ All {n_subsets} subsets already exist, skipping generation.")
        return existing

    print(f"Counting total records in {json_path}...")
    total = sum(1 for _ in open(json_path, 'r', encoding='utf-8'))
    print(f"Total records: {total:,}")

    required = n_subsets * subset_size
    if required > total:
        raise ValueError(
            f"Need {required:,} records for {n_subsets} non-overlapping subsets "
            f"of {subset_size}, but only {total:,} available."
        )

    random.seed(seed)
    all_indices = random.sample(range(total), required)
    index_sets = [
        set(all_indices[i * subset_size:(i + 1) * subset_size])
        for i in range(n_subsets)
    ]

    subset_records = [[] for _ in range(n_subsets)]
    print("Reading file in single pass...")
    with open(json_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(tqdm(f, total=total)):
            for subset_idx, index_set in enumerate(index_sets):
                if line_idx in index_set:
                    subset_records[subset_idx].append(line)
                    break

    subset_paths = []
    for i, records in enumerate(subset_records):
        subset_path = os.path.join(output_dir, f'subset_{i:02d}.jsonl')
        with open(subset_path, 'w', encoding='utf-8') as f:
            f.writelines(records)
        emb_dir = os.path.join(output_dir, f'embeddings_{i:02d}')
        os.makedirs(emb_dir, exist_ok=True)
        subset_paths.append(subset_path)
        print(f"✅ Subset {i:02d}: {len(records)} records → {subset_path}")

    print(f"\n✅ Generated {n_subsets} subsets of {subset_size} records each.\n")
    return subset_paths


class OntologicalExperimentPipeline:
    """
    Complete pipeline for running ontological experiments.
    """

    def __init__(self, config: Dict):
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

        self.data_manager = ArxivDataManager(
            data_path=self.config['data_path'],
            subset_size=self.config.get('subset_size', None)
        )

        self.data_manager.load_data()
        self.data_manager.extract_unique_authors()

        if self.config.get('use_precomputed_embeddings', False):
            self.data_manager.load_embeddings(
                input_dir=self.config['embedding_dir'],
                prefix=self.config.get('embedding_prefix', '')
            )
        else:
            self.data_manager.compute_embeddings(
                model_name=self.config.get('sbert_model', 'all-MiniLM-L6-v2'),
                text_fidelity='super',
                force_recompute=self.config.get('force_recompute', False)
            )
            if self.config.get('save_embeddings', True):
                self.data_manager.save_embeddings(
                    output_dir=self.config['embedding_dir'],
                    prefix=self.config.get('embedding_prefix', '')
                )

        self.tag_builder = TAGBuilder(self.data_manager)
        print("\n✅ Data setup complete\n")

    def define_ontological_space(self) -> List[Dict]:
        configs = TAGConfiguration()
        experiment_space = []

        task_indices = self.config.get('task_indices', [1])
        node_indices = self.config.get('node_indices', [7, 8])
        edge_indices = self.config.get('edge_indices', [10, 11])
        text_indices = self.config.get('text_indices', ['a', 'b', 'c'])
        similarity_thresholds = self.config.get('similarity_thresholds', ['b'])

        for task_idx in task_indices:
            for node_idx in node_indices:
                for edge_idx in edge_indices:
                    for text_idx in text_indices:
                        if edge_idx == 11:
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

        tag = self.tag_builder.build(
            task_idx=exp_config['task_idx'],
            node_idx=exp_config['node_idx'],
            edge_idx=exp_config['edge_idx'],
            text_idx=exp_config['text_idx'],
            similarity_threshold=exp_config['similarity_threshold']
        )

        num_classes = self.data_manager.num_classes or self.config.get('num_classes', 10)

        if exp_config['node_idx'] == 8:
            embeddings = self.data_manager.author_embeddings
            node_list = self.data_manager.author_list
            labels = self.data_manager.get_author_category_labels(num_classes=num_classes)
        else:
            embeddings = self.data_manager.article_embeddings
            node_list = list(self.data_manager.df['id'])
            labels = self.data_manager.get_category_labels(num_classes=num_classes)

        embeddings = normalize(embeddings, axis=1).astype(np.float32)

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
            save_model_path = str(
                Path(self.config['output_dir']) / 'models' / f"model_{tag.get_identifier()}.pt"
            )
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

        G = tag.graph
        all_words = []
        for row in self.data_manager.df.itertuples():
            text = f"{getattr(row, 'title', '') or ''} {getattr(row, 'abstract', '') or ''}"
            all_words.extend(text.lower().split())

        counts = Counter(all_words)
        total_words = sum(counts.values())
        text_vocab_entropy = -sum(
            (c / total_words) * math.log2(c / total_words) for c in counts.values()
        )

        label_counts = self.data_manager.df['cat_top'].value_counts()
        label_probs = label_counts / label_counts.sum()
        label_balance_entropy = -sum(p * math.log2(p) for p in label_probs)

        avg_text_length = np.mean([
            len(str(getattr(row, 'title', '') or '')) + len(str(getattr(row, 'abstract', '') or ''))
            for row in self.data_manager.df.itertuples()
        ])

        results['stats'] = {
            'number_of_nodes': G.number_of_nodes(),
            'avg_text_length': round(float(avg_text_length), 4),
            'text_vocab_entropy': text_vocab_entropy,
            'label_balance_entropy': float(label_balance_entropy),
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

        experiment_space = self.define_ontological_space()

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
                    'Similarity_Threshold': exp_config.get('similarity_threshold', ''),
                    **results['test_metrics'],
                    **results.get('stats', {})
                }

                write_header = not stream_path.exists()
                with open(stream_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
                
                # Stream result to JSON
                json_stream_path = Path(self.config['output_dir']) / 'analysis' / 'results_stream.json'
                existing = []
                if json_stream_path.exists():
                    with open(json_stream_path, 'r') as f:
                        existing = json.load(f)
                existing.append({
                    'Task_Idx': exp_config['task_idx'],
                    'Node_Idx': exp_config['node_idx'],
                    'Edge_Idx': exp_config['edge_idx'],
                    'Text_Idx': exp_config['text_idx'],
                    'Similarity_Threshold': exp_config.get('similarity_threshold', ''),
                    **results['test_metrics'],
                    **results.get('stats', {})
                })
                with open(json_stream_path, 'w') as f:
                    json.dump(existing, f, indent=2)

                self.analyzer.add_experiment(
                    task_idx=exp_config['task_idx'],
                    node_idx=exp_config['node_idx'],
                    edge_idx=exp_config['edge_idx'],
                    text_idx=exp_config['text_idx'],
                    test_metrics=results['test_metrics'],
                    stats=results.get('stats', None)
                )

            except Exception as e:
                print(f"⚠ Experiment {i+1} failed: {e}")
                continue

        print("\n✅ All experiments complete\n")

    def analyze_results(self):
        """Perform meta-analysis on results."""
        print("\n" + "="*80)
        print("STEP 3: META-ANALYSIS")
        print("="*80)

        self.analyzer.build_decision_tree_table()
        self.analyzer.analyze_best_configurations(top_k=self.config.get('top_k', 5))
        self.analyzer.analyze_by_ontology()

        analysis_dir = Path(self.config['output_dir']) / 'analysis'
        self.analyzer.save_analysis(str(analysis_dir))
        results_path = analysis_dir / 'all_results.json'
        self.performance_tracker.save_results(str(results_path))

        if len(self.analyzer.dt_table) >= 5:
            self.analyzer.train_decision_tree(
                target_metric='Performance_Band',
                max_depth=self.config.get('dt_max_depth', 5),
                min_samples_split=self.config.get('dt_min_samples_split', 5)
            )

            if self.config.get('plot_decision_tree', True):
                try:
                    plot_path = Path(self.config['output_dir']) / 'analysis' / 'decision_tree.png'
                    self.analyzer.plot_decision_tree(save_path=str(plot_path))
                    print(f"✅ Decision tree plot saved to {plot_path}")
                except Exception as e:
                    print(f"⚠ Decision tree plot failed: {e}")

        print("\n✅ Meta-analysis complete\n")

    def run(self):
        """Execute the complete pipeline."""
        print("\n" + "="*80)
        print("ONTOLOGICAL GENERALIZATION FRAMEWORK")
        print(f"Subset: {self.config['data_path']}")
        print("="*80)

        self.setup_data()
        self.run_all_experiments()
        self.analyze_results()

        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_config = {
        'subset_size': None,

        # Ontological space
        'task_indices': [1],
        'node_indices': [7, 8],
        'edge_indices': [10, 11],
        'text_indices': ['a', 'e'],
        'similarity_thresholds': ['a', 'c'],

        # Model
        'model_type': 'sage',
        'hidden_dim': 256,
        'dropout': 0.5,

        # Training
        'lr': 0.001,
        'weight_decay': 5e-4,
        'epochs': 100,
        'patience': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # Output
        'save_models': False,           # disabled to save disk space across 8 runs
        'save_training_history': False,
        'dt_max_depth': 8,
        'dt_min_samples_split': 8,
        'plot_decision_tree': True,
        'top_k': 5,
        'use_precomputed_embeddings': False,
        'save_embeddings': True,
    }

    # Step 1: Generate 8 random non-overlapping subsets from the 200k file
    subset_paths = generate_random_subsets(
        json_path=r'data\arxiv_subset_200k.jsonl',
        n_subsets=8,
        subset_size=10000,
        output_dir='data/subsets',
        seed=42
    )

    # Step 2: Run full 40-variant pipeline on each subset
    for subset_idx, subset_path in enumerate(subset_paths):
        print(f"\n{'='*80}")
        print(f"SUBSET RUN {subset_idx + 1}/8  —  {subset_path}")
        print(f"{'='*80}")

        config = {
            **base_config,
            'data_path': subset_path,
            'embedding_dir': f'data/subsets/embeddings_{subset_idx:02d}',
            'output_dir': f'output/run_{run_id}/subset_{subset_idx:02d}',
        }

        try:
            pipeline = OntologicalExperimentPipeline(config)
            pipeline.run()
        except Exception as e:
            print(f"⚠ Subset {subset_idx} run failed: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"ALL SUBSET RUNS COMPLETE")
    print(f"Results in: output/run_{run_id}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()