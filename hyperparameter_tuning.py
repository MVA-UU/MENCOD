"""
Hyperparameter Tuning for Hybrid Outlier Detection

This module implements hyperparameter optimization to find the best weights
for combining different models in the hybrid outlier detection system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import itertools
from time import time
import json

from models.hybrid_models import HybridOutlierDetector, ModelWeights


@dataclass
class TuningResult:
    """Results from hyperparameter tuning."""
    weights: ModelWeights
    outlier_position: int
    total_candidates: int
    outlier_score: float
    percentile: float
    evaluation_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'citation_weight': self.weights.citation_network,
            'content_weight': self.weights.content_similarity,
            'outlier_position': self.outlier_position,
            'total_candidates': self.total_candidates,
            'outlier_score': self.outlier_score,
            'percentile': self.percentile,
            'evaluation_time': self.evaluation_time
        }


class HyperparameterTuner:
    """Hyperparameter tuning system for hybrid outlier detection."""
    
    def __init__(self, dataset_name: str = "Appenzeller-Herzog_2019"):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            dataset_name: Name of the synergy dataset to use
        """
        self.dataset_name = dataset_name
        self.results = []
        self.best_result = None
        
    def grid_search(self, 
                   citation_weights: List[float],
                   content_weights: Optional[List[float]] = None,
                   simulation_df: Optional[pd.DataFrame] = None) -> List[TuningResult]:
        """
        Perform grid search over weight combinations.
        
        Args:
            citation_weights: List of citation network weights to try
            content_weights: List of content similarity weights to try (auto-normalized if None)
            simulation_df: Simulation data (loaded from file if None)
        
        Returns:
            List of tuning results sorted by performance
        """
        if simulation_df is None:
            simulation_df = pd.read_csv('data/simulation.csv')
        
        # Prepare training data (exclude rank 26 outlier)
        training_data = simulation_df.copy()
        training_data['label_included'] = training_data['asreview_ranking'].apply(
            lambda x: 1 if x <= 25 else 0
        )
        
        # Get outlier info
        outlier_row = simulation_df[simulation_df['record_id'] == 497]
        outlier_id = outlier_row.iloc[0]['openalex_id']
        
        # Create search pool (outlier + irrelevant documents)
        irrelevant_docs = simulation_df[simulation_df['label_included'] == 0]['openalex_id'].tolist()
        search_pool = [outlier_id] + irrelevant_docs
        
        print(f"=== GRID SEARCH HYPERPARAMETER TUNING ===")
        print(f"Outlier ID: {outlier_id}")
        print(f"Search pool size: {len(search_pool)}")
        print(f"Citation weights to try: {len(citation_weights)}")
        
        # Generate weight combinations
        if content_weights is None:
            # Auto-normalize: content weight = 1 - citation weight
            weight_combinations = [(cit, 1.0 - cit) for cit in citation_weights if 0 <= cit <= 1]
        else:
            # Use provided combinations
            weight_combinations = list(itertools.product(citation_weights, content_weights))
            # Filter to valid combinations that sum to ~1.0
            weight_combinations = [(c, s) for c, s in weight_combinations 
                                 if abs(c + s - 1.0) < 0.01]
        
        print(f"Weight combinations to test: {len(weight_combinations)}")
        
        results = []
        total_combinations = len(weight_combinations)
        
        for i, (citation_weight, content_weight) in enumerate(weight_combinations):
            print(f"\nTesting combination {i+1}/{total_combinations}")
            print(f"Citation: {citation_weight:.3f}, Content: {content_weight:.3f}")
            
            start_time = time()
            
            # Create and fit model
            weights = ModelWeights(
                citation_network=citation_weight,
                content_similarity=content_weight
            )
            
            detector = HybridOutlierDetector(
                dataset_name=self.dataset_name,
                model_weights=weights
            )
            
            try:
                detector.fit(training_data)
                scores = detector.predict_relevance_scores(search_pool)
                
                # Find outlier position
                sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                outlier_position = None
                outlier_score = 0.0
                
                for pos, (doc_id, score) in enumerate(sorted_results, 1):
                    if doc_id == outlier_id:
                        outlier_position = pos
                        outlier_score = score
                        break
                
                if outlier_position is None:
                    outlier_position = len(search_pool)  # Worst case
                
                percentile = ((len(search_pool) - outlier_position) / len(search_pool)) * 100
                evaluation_time = time() - start_time
                
                result = TuningResult(
                    weights=weights,
                    outlier_position=outlier_position,
                    total_candidates=len(search_pool),
                    outlier_score=outlier_score,
                    percentile=percentile,
                    evaluation_time=evaluation_time
                )
                
                results.append(result)
                
                print(f"Position: {outlier_position}/{len(search_pool)} "
                      f"({percentile:.1f}th percentile), "
                      f"Score: {outlier_score:.4f}, "
                      f"Time: {evaluation_time:.1f}s")
                
            except Exception as e:
                print(f"Error with weights ({citation_weight}, {content_weight}): {e}")
                continue
        
        # Sort by performance (lower position is better)
        results.sort(key=lambda x: x.outlier_position)
        
        self.results = results
        if results:
            self.best_result = results[0]
        
        print(f"\n=== GRID SEARCH COMPLETE ===")
        print(f"Tested {len(results)} valid combinations")
        if self.best_result:
            print(f"Best result: Position {self.best_result.outlier_position} "
                  f"({self.best_result.percentile:.1f}th percentile)")
            print(f"Best weights: Citation={self.best_result.weights.citation_network:.3f}, "
                  f"Content={self.best_result.weights.content_similarity:.3f}")
        
        return results
    
    def random_search(self, 
                     n_iterations: int = 50,
                     simulation_df: Optional[pd.DataFrame] = None,
                     seed: Optional[int] = None) -> List[TuningResult]:
        """
        Perform random search over weight space.
        
        Args:
            n_iterations: Number of random combinations to try
            simulation_df: Simulation data (loaded from file if None)
            seed: Random seed for reproducibility
        
        Returns:
            List of tuning results sorted by performance
        """
        if seed is not None:
            np.random.seed(seed)
        
        if simulation_df is None:
            simulation_df = pd.read_csv('data/simulation.csv')
        
        # Prepare data same as grid search
        training_data = simulation_df.copy()
        training_data['label_included'] = training_data['asreview_ranking'].apply(
            lambda x: 1 if x <= 25 else 0
        )
        
        outlier_row = simulation_df[simulation_df['record_id'] == 497]
        outlier_id = outlier_row.iloc[0]['openalex_id']
        
        irrelevant_docs = simulation_df[simulation_df['label_included'] == 0]['openalex_id'].tolist()
        search_pool = [outlier_id] + irrelevant_docs
        
        print(f"=== RANDOM SEARCH HYPERPARAMETER TUNING ===")
        print(f"Iterations: {n_iterations}")
        print(f"Search pool size: {len(search_pool)}")
        
        results = []
        
        for i in range(n_iterations):
            # Generate random weights that sum to 1
            citation_weight = np.random.uniform(0.0, 1.0)
            content_weight = 1.0 - citation_weight
            
            print(f"\nIteration {i+1}/{n_iterations}")
            print(f"Citation: {citation_weight:.3f}, Content: {content_weight:.3f}")
            
            start_time = time()
            
            weights = ModelWeights(
                citation_network=citation_weight,
                content_similarity=content_weight
            )
            
            detector = HybridOutlierDetector(
                dataset_name=self.dataset_name,
                model_weights=weights
            )
            
            try:
                detector.fit(training_data)
                scores = detector.predict_relevance_scores(search_pool)
                
                # Find outlier position
                sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                outlier_position = None
                outlier_score = 0.0
                
                for pos, (doc_id, score) in enumerate(sorted_results, 1):
                    if doc_id == outlier_id:
                        outlier_position = pos
                        outlier_score = score
                        break
                
                if outlier_position is None:
                    outlier_position = len(search_pool)
                
                percentile = ((len(search_pool) - outlier_position) / len(search_pool)) * 100
                evaluation_time = time() - start_time
                
                result = TuningResult(
                    weights=weights,
                    outlier_position=outlier_position,
                    total_candidates=len(search_pool),
                    outlier_score=outlier_score,
                    percentile=percentile,
                    evaluation_time=evaluation_time
                )
                
                results.append(result)
                
                print(f"Position: {outlier_position}/{len(search_pool)} "
                      f"({percentile:.1f}th percentile), "
                      f"Score: {outlier_score:.4f}")
                
            except Exception as e:
                print(f"Error with weights ({citation_weight:.3f}, {content_weight:.3f}): {e}")
                continue
        
        # Sort by performance
        results.sort(key=lambda x: x.outlier_position)
        
        self.results = results
        if results:
            self.best_result = results[0]
        
        print(f"\n=== RANDOM SEARCH COMPLETE ===")
        print(f"Tested {len(results)} combinations")
        if self.best_result:
            print(f"Best result: Position {self.best_result.outlier_position} "
                  f"({self.best_result.percentile:.1f}th percentile)")
            print(f"Best weights: Citation={self.best_result.weights.citation_network:.3f}, "
                  f"Content={self.best_result.weights.content_similarity:.3f}")
        
        return results
    
    def bayesian_search(self, n_iterations: int = 30) -> List[TuningResult]:
        """
        Perform Bayesian optimization (placeholder - requires scikit-optimize).
        
        Args:
            n_iterations: Number of iterations
        
        Returns:
            List of tuning results
        """
        print("Bayesian optimization not implemented yet.")
        print("Install scikit-optimize for Bayesian hyperparameter tuning.")
        return []
    
    def analyze_results(self, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze tuning results and show top performers.
        
        Args:
            top_n: Number of top results to show
        
        Returns:
            DataFrame with analysis of top results
        """
        if not self.results:
            print("No results to analyze. Run tuning first.")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        results_data = [result.to_dict() for result in self.results[:top_n]]
        df = pd.DataFrame(results_data)
        
        print(f"=== TOP {top_n} HYPERPARAMETER RESULTS ===")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Show statistics
        all_positions = [r.outlier_position for r in self.results]
        all_percentiles = [r.percentile for r in self.results]
        
        print(f"\n=== PERFORMANCE STATISTICS ===")
        print(f"Best position: {min(all_positions)}")
        print(f"Worst position: {max(all_positions)}")
        print(f"Mean position: {np.mean(all_positions):.1f}")
        print(f"Median position: {np.median(all_positions):.1f}")
        print(f"Best percentile: {max(all_percentiles):.1f}th")
        print(f"Mean percentile: {np.mean(all_percentiles):.1f}th")
        
        return df
    
    def save_results(self, filename: str = "hyperparameter_results.json"):
        """Save tuning results to file."""
        if not self.results:
            print("No results to save.")
            return
        
        results_data = [result.to_dict() for result in self.results]
        
        with open(filename, 'w') as f:
            json.dump({
                'results': results_data,
                'best_result': self.best_result.to_dict() if self.best_result else None,
                'num_results': len(self.results)
            }, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def plot_results(self):
        """Plot tuning results (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Cannot plot results.")
            return
        
        if not self.results:
            print("No results to plot.")
            return
        
        citation_weights = [r.weights.citation_network for r in self.results]
        positions = [r.outlier_position for r in self.results]
        percentiles = [r.percentile for r in self.results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Position vs Citation Weight
        ax1.scatter(citation_weights, positions, alpha=0.6)
        ax1.set_xlabel('Citation Network Weight')
        ax1.set_ylabel('Outlier Position (lower is better)')
        ax1.set_title('Outlier Position vs Citation Weight')
        ax1.grid(True, alpha=0.3)
        
        # Percentile vs Citation Weight
        ax2.scatter(citation_weights, percentiles, alpha=0.6)
        ax2.set_xlabel('Citation Network Weight')
        ax2.set_ylabel('Percentile (higher is better)')
        ax2.set_title('Percentile vs Citation Weight')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Run hyperparameter tuning experiments."""
    print("=== HYPERPARAMETER TUNING FOR HYBRID OUTLIER DETECTION ===")
    
    tuner = HyperparameterTuner()
    
    # Grid search with different granularities
    print("\n1. Coarse grid search...")
    coarse_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    coarse_results = tuner.grid_search(citation_weights=coarse_weights)
    
    if coarse_results:
        print("\nCoarse search complete. Analyzing results...")
        coarse_df = tuner.analyze_results(top_n=5)
        
        # Fine-grained search around best result
        best_citation = tuner.best_result.weights.citation_network
        print(f"\n2. Fine grid search around best citation weight: {best_citation:.3f}")
        
        # Create fine grid around best result
        fine_range = 0.1
        fine_step = 0.02
        fine_start = max(0.0, best_citation - fine_range)
        fine_end = min(1.0, best_citation + fine_range)
        fine_weights = np.arange(fine_start, fine_end + fine_step, fine_step).tolist()
        
        fine_results = tuner.grid_search(citation_weights=fine_weights)
        
        if fine_results:
            print("\nFine search complete. Final results...")
            final_df = tuner.analyze_results(top_n=10)
    
    # Random search for comparison
    print("\n3. Random search for comparison...")
    random_results = tuner.random_search(n_iterations=50, seed=42)
    
    if random_results:
        print("\nRandom search complete. Comparing with grid search...")
        random_df = tuner.analyze_results(top_n=5)
    
    # Save all results
    tuner.save_results("hyperparameter_tuning_results.json")
    
    # Final recommendation
    if tuner.best_result:
        print(f"\n=== FINAL RECOMMENDATION ===")
        print(f"Best weights found:")
        print(f"  Citation Network: {tuner.best_result.weights.citation_network:.3f}")
        print(f"  Content Similarity: {tuner.best_result.weights.content_similarity:.3f}")
        print(f"Performance:")
        print(f"  Outlier Position: {tuner.best_result.outlier_position}")
        print(f"  Percentile: {tuner.best_result.percentile:.1f}th")
        print(f"  Score: {tuner.best_result.outlier_score:.4f}")


if __name__ == "__main__":
    main() 