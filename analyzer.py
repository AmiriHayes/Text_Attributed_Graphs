"""
Meta-Analysis Module for Ontological Generalization Framework
Manages Decision Tree analysis across TAG experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class PerformanceBand:
    """Performance band categorization for meta-analysis."""
    
    @staticmethod
    def categorize_kl(kl: float) -> str:
        """Categorize KL divergence into performance bands."""
        if kl < 0.3:
            return 'Excellent'
        elif kl < 0.5:
            return 'Good'
        elif kl < 1.0:
            return 'Fair'
        else:
            return 'Poor'
    
    @staticmethod
    def categorize_accuracy(acc: float) -> str:
        """Categorize accuracy into performance bands."""
        if acc >= 0.9:
            return 'Excellent 10/10'
        elif acc >= 0.8:
            return 'Good 9/10'
        elif acc >= 0.7:
            return 'Fair 8/10'
        elif acc >= 0.6:
            return 'Fair 7/10'
        elif acc >= 0.5:
            return 'Fair 6/10'
        elif acc >= 0.4:
            return 'Poor 5/10'
        else:
            return 'Poor <5/10'
    
    @staticmethod
    def categorize_metric(metric_name: str, value: float) -> str:
        """Categorize any metric into performance bands."""
        if metric_name == 'kl':
            return PerformanceBand.categorize_kl(value)
        elif metric_name in ['top1', 'top3', 'cosine']:
            return PerformanceBand.categorize_accuracy(value)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")


class DecisionTreeAnalyzer:
    """
    Meta-Analysis Engine: Analyzes ontological indices using Decision Trees.
    Tracks experiments and learns patterns across TAG variants.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.experiments = []
        self.dt_table = pd.DataFrame()
        self.decision_tree = None
        self.feature_encoders = {}
        
    def add_experiment(self, task_idx: int, node_idx: int, edge_idx: int,
                   text_idx: str, test_metrics: Dict, stats: Dict = None):
        
        primary_metric = test_metrics.get('top1', 0.0)
        performance_band = PerformanceBand.categorize_accuracy(primary_metric)
        
        experiment = {
            'Task_Idx': task_idx,
            'Node_Idx': node_idx,
            'Edge_Idx': edge_idx,
            'Text_Idx': text_idx,
            'KL': test_metrics.get('kl', np.nan),
            'Top1': test_metrics.get('top1', np.nan),
            'Top3': test_metrics.get('top3', np.nan),
            'Cosine': test_metrics.get('cosine', np.nan),
            'Performance_Band': performance_band,
            # Stats columns — default to nan if not provided
            'number_of_nodes': stats.get('number_of_nodes', np.nan) if stats else np.nan,
            'avg_text_length': stats.get('avg_text_length', np.nan) if stats else np.nan,
            'text_vocab_entropy': stats.get('text_vocab_entropy', np.nan) if stats else np.nan,
            'label_balance_entropy': stats.get('label_balance_entropy', np.nan) if stats else np.nan,
            'task_type': stats.get('task_type', np.nan) if stats else np.nan,
            'output_dimension': stats.get('output_dimension', np.nan) if stats else np.nan,
            'normalized_score': stats.get('normalized_score', np.nan) if stats else np.nan,
        }
        
        self.experiments.append(experiment)
        print(f"✅ Added experiment: T{task_idx}_N{node_idx}_E{edge_idx}_X{text_idx} -> {performance_band}")

    def build_decision_tree_table(self) -> pd.DataFrame:
        """
        Build the Decision Tree Table from all experiments.
        
        Returns:
            DataFrame with all experiments
        """
        self.dt_table = pd.DataFrame(self.experiments)
        print(f"\n{'='*60}")
        print(f"Decision Tree Table: {len(self.dt_table)} experiments")
        print(f"{'='*60}")
        print(self.dt_table.head(10))
        return self.dt_table
    
    def train_decision_tree(self, target_metric: str = 'Performance_Band',
                           max_depth: int = 5, min_samples_split: int = 5,
                           test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train a Decision Tree Classifier on the experiments.
        
        Args:
            target_metric: Target column to predict
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            test_size: Test set size
            random_state: Random seed
        
        Returns:
            Dictionary with training results
        """
        if len(self.dt_table) == 0:
            raise ValueError("No experiments added. Call add_experiment() first.")
        
        # Prepare features and target
        feature_cols = [
            'Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx',
            'number_of_nodes', 'avg_text_length',
            'text_vocab_entropy', 'label_balance_entropy', 'output_dimension'
        ]
        # Filter to only columns that exist and have data
        feature_cols = [c for c in feature_cols if c in self.dt_table.columns 
                        and self.dt_table[c].notna().any()]
        X = self.dt_table[feature_cols].copy()
        
        # Encode Text_Idx (categorical)
        if 'Text_Idx' not in self.feature_encoders:
            self.feature_encoders['Text_Idx'] = LabelEncoder()
            X['Text_Idx'] = self.feature_encoders['Text_Idx'].fit_transform(X['Text_Idx'])
        else:
            X['Text_Idx'] = self.feature_encoders['Text_Idx'].transform(X['Text_Idx'])
        
        # Encode target if necessary
        y = self.dt_table[target_metric].copy()
        if y.dtype == 'object':
            if target_metric not in self.feature_encoders:
                self.feature_encoders[target_metric] = LabelEncoder()
                y = self.feature_encoders[target_metric].fit_transform(y)
            else:
                y = self.feature_encoders[target_metric].transform(y)
        
        # Train/test split
        if len(X) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            # Too few samples for splitting
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Train Decision Tree
        self.decision_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        self.decision_tree.fit(X_train, y_train)
        
        # Evaluate
        train_acc = self.decision_tree.score(X_train, y_train)
        test_acc = self.decision_tree.score(X_test, y_test)
        
        print(f"\n{'='*60}")
        print(f"Decision Tree Training Results")
        print(f"{'='*60}")
        print(f"Features: {feature_cols}")
        print(f"Target: {target_metric}")
        print(f"Train Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")
        print(f"Tree Depth: {self.decision_tree.get_depth()}")
        print(f"Number of Leaves: {self.decision_tree.get_n_leaves()}")
        print(f"{'='*60}\n")
        
        # Feature importances
        importances = self.decision_tree.feature_importances_
        for feat, imp in zip(feature_cols, importances):
            print(f"  {feat}: {imp:.3f}")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'tree_depth': self.decision_tree.get_depth(),
            'n_leaves': self.decision_tree.get_n_leaves(),
            'feature_importances': dict(zip(feature_cols, importances))
        }
    
    def plot_decision_tree(self, figsize: tuple = (20, 10), 
                          fontsize: int = 10, save_path: Optional[str] = None):
        """
        Visualize the trained Decision Tree.
        
        Args:
            figsize: Figure size
            fontsize: Font size for labels
            save_path: Path to save the plot
        """
        if self.decision_tree is None:
            raise ValueError("No trained decision tree. Call train_decision_tree() first.")
        
        plt.figure(figsize=figsize)
        
        # feature_names = ['Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx']
        feature_names = [
            'Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx',
            'number_of_nodes', 'avg_text_length',
            'text_vocab_entropy', 'label_balance_entropy', 'output_dimension'
        ]
        
        # Get class names
        if 'Performance_Band' in self.feature_encoders:
            class_names = self.feature_encoders['Performance_Band'].classes_
        else:
            class_names = None
        
        plot_tree(
            self.decision_tree,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=fontsize
        )
        
        plt.title("Decision Tree: Ontological Index → Performance Band", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Decision tree plot saved to {save_path}")
        
        plt.show()
    
    def predict_performance(self, task_idx: int, node_idx: int, 
                           edge_idx: int, text_idx: str) -> str:
        """
        Predict performance band for a new TAG configuration.
        
        Args:
            task_idx: Task index
            node_idx: Node index
            edge_idx: Edge index
            text_idx: Text fidelity index
        
        Returns:
            Predicted performance band
        """
        if self.decision_tree is None:
            raise ValueError("No trained decision tree. Call train_decision_tree() first.")
        
        # Prepare input
        X = pd.DataFrame({
            'Task_Idx': [task_idx],
            'Node_Idx': [node_idx],
            'Edge_Idx': [edge_idx],
            'Text_Idx': [text_idx]
        })
        
        # Encode Text_Idx
        X['Text_Idx'] = self.feature_encoders['Text_Idx'].transform(X['Text_Idx'])
        
        # Predict
        y_pred = self.decision_tree.predict(X)
        
        # Decode
        if 'Performance_Band' in self.feature_encoders:
            performance_band = self.feature_encoders['Performance_Band'].inverse_transform(y_pred)[0]
        else:
            performance_band = y_pred[0]
        
        return performance_band
    
    def analyze_best_configurations(self, top_k: int = 5) -> pd.DataFrame:
        """
        Analyze and return the best performing configurations.
        
        Args:
            top_k: Number of top configurations to return
        
        Returns:
            DataFrame with top configurations
        """
        if len(self.dt_table) == 0:
            raise ValueError("No experiments added.")
        
        # Sort by Top1 accuracy (descending)
        sorted_table = self.dt_table.sort_values('Top1', ascending=False)
        
        print(f"\n{'='*60}")
        print(f"Top {top_k} Configurations by Top-1 Accuracy")
        print(f"{'='*60}")
        print(sorted_table.head(top_k))
        
        return sorted_table.head(top_k)
    
    def analyze_by_ontology(self) -> Dict:
        """
        Analyze performance grouped by ontological indices.
        
        Returns:
            Dictionary with grouped statistics
        """
        if len(self.dt_table) == 0:
            raise ValueError("No experiments added.")
        
        results = {}
        
        # Group by Task
        results['by_task'] = self.dt_table.groupby('Task_Idx')['Top1'].agg(['mean', 'std', 'count'])
        
        # Group by Node
        results['by_node'] = self.dt_table.groupby('Node_Idx')['Top1'].agg(['mean', 'std', 'count'])
        
        # Group by Edge
        results['by_edge'] = self.dt_table.groupby('Edge_Idx')['Top1'].agg(['mean', 'std', 'count'])
        
        # Group by Text
        results['by_text'] = self.dt_table.groupby('Text_Idx')['Top1'].agg(['mean', 'std', 'count'])
        
        print(f"\n{'='*60}")
        print(f"Performance by Task Index")
        print(f"{'='*60}")
        print(results['by_task'])
        
        print(f"\n{'='*60}")
        print(f"Performance by Node Index")
        print(f"{'='*60}")
        print(results['by_node'])
        
        print(f"\n{'='*60}")
        print(f"Performance by Edge Index")
        print(f"{'='*60}")
        print(results['by_edge'])
        
        print(f"\n{'='*60}")
        print(f"Performance by Text Fidelity")
        print(f"{'='*60}")
        print(results['by_text'])
        
        return results
    
    def save_analysis(self, output_dir: str):
        """
        Save all analysis results to disk.
        
        Args:
            output_dir: Output directory
        """
        from pathlib import Path
        import json
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save Decision Tree Table
        self.dt_table.to_csv(output_path / "decision_tree_table.csv", index=False)
        
        # Save experiments as JSON
        with open(output_path / "experiments.json", 'w') as f:
            json.dump(self.experiments, f, indent=2)
        
        # Save Decision Tree model
        if self.decision_tree is not None:
            import joblib
            joblib.dump(self.decision_tree, output_path / "decision_tree_model.pkl")
        
        print(f"✅ Analysis saved to {output_path}")
    
    def load_analysis(self, input_dir: str):
        """
        Load analysis results from disk.
        
        Args:
            input_dir: Input directory
        """
        from pathlib import Path
        import json
        
        input_path = Path(input_dir)
        
        # Load experiments
        with open(input_path / "experiments.json", 'r') as f:
            self.experiments = json.load(f)
        
        # Load Decision Tree Table
        self.dt_table = pd.read_csv(input_path / "decision_tree_table.csv")
        
        # Load Decision Tree model
        model_path = input_path / "decision_tree_model.pkl"
        if model_path.exists():
            import joblib
            self.decision_tree = joblib.load(model_path)
        
        print(f"✅ Analysis loaded from {input_path}")
