"""
Evaluation and Results Analysis Module

Handles analysis of outlier detection results including:
- Outlier document retrieval and ranking
- Method comparison and performance analysis
- Score breakdown and summary generation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Handles analysis and evaluation of outlier detection results."""
    
    def __init__(self, outlier_results: Dict[str, np.ndarray], 
                 features_df: pd.DataFrame):
        """
        Initialize results analyzer.
        
        Args:
            outlier_results: Dictionary with outlier scores from different methods
            features_df: DataFrame with extracted features
        """
        self.outlier_results = outlier_results
        self.features_df = features_df
    
    def get_outlier_documents(self, method: str = 'ensemble', 
                            top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Get outlier documents with detailed information.
        
        Args:
            method: Method to use for ranking ('ensemble', 'lof_embeddings', 'lof_network', 
                   'lof_mixed', 'isolation_forest')
            top_k: Number of top outliers to return (None for all)
            
        Returns:
            DataFrame with outlier documents and detailed scores
        """
        score_key = f'{method}_scores'
        
        if score_key not in self.outlier_results:
            raise ValueError(f"Unknown method: {method}. Available methods: "
                           f"{[k.replace('_scores', '') for k in self.outlier_results.keys() if k.endswith('_scores')]}")
        
        scores = self.outlier_results[score_key]
        doc_ids = self.outlier_results['openalex_ids']
        
        # Get individual method scores for Multi-LOF
        lof_embeddings_scores = self.outlier_results.get('lof_embeddings_scores', np.zeros(len(doc_ids)))
        lof_network_scores = self.outlier_results.get('lof_network_scores', np.zeros(len(doc_ids)))
        lof_mixed_scores = self.outlier_results.get('lof_mixed_scores', np.zeros(len(doc_ids)))
        isolation_forest_scores = self.outlier_results.get('isolation_forest_scores', np.zeros(len(doc_ids)))
        ensemble_scores = self.outlier_results.get('ensemble_scores', np.zeros(len(doc_ids)))
        
        # Create ranked results
        results = self._create_ranked_results(
            scores, doc_ids, lof_embeddings_scores, lof_network_scores, 
            lof_mixed_scores, isolation_forest_scores, ensemble_scores, top_k
        )
        
        return pd.DataFrame(results).sort_values('outlier_score', ascending=False)
    
    def get_method_comparison(self) -> pd.DataFrame:
        """
        Compare results across all Multi-LOF outlier detection methods.
        
        Returns:
            DataFrame with method comparison statistics
        """
        methods = ['lof_embeddings', 'lof_network', 'lof_mixed', 'isolation_forest', 'ensemble']
        comparison = []
        
        method_names = {
            'lof_embeddings': 'LOF (Embeddings)',
            'lof_network': 'LOF (Network Features)',
            'lof_mixed': 'LOF (Mixed Features)',
            'isolation_forest': 'Isolation Forest',
            'ensemble': 'Multi-LOF Ensemble'
        }
        
        for method in methods:
            score_key = f'{method}_scores'
            
            if score_key not in self.outlier_results:
                continue
                
            scores = self.outlier_results[score_key]
            
            comparison.append({
                'method': method_names.get(method, method),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'score_range': np.max(scores) - np.min(scores),
            })
        
        return pd.DataFrame(comparison)
    
    def get_detailed_outlier_breakdown(self, top_k: int = 10) -> pd.DataFrame:
        """
        Get detailed breakdown of outlier scores showing all Multi-LOF method contributions.
        
        Args:
            top_k: Number of top outliers to show
            
        Returns:
            DataFrame with detailed Multi-LOF score breakdown
        """
        doc_ids = self.outlier_results['openalex_ids']
        lof_embeddings_scores = self.outlier_results.get('lof_embeddings_scores', np.zeros(len(doc_ids)))
        lof_network_scores = self.outlier_results.get('lof_network_scores', np.zeros(len(doc_ids)))
        lof_mixed_scores = self.outlier_results.get('lof_mixed_scores', np.zeros(len(doc_ids)))
        isolation_forest_scores = self.outlier_results.get('isolation_forest_scores', np.zeros(len(doc_ids)))
        ensemble_scores = self.outlier_results.get('ensemble_scores', np.zeros(len(doc_ids)))
        
        # Create detailed breakdown
        breakdown = []
        for i, doc_id in enumerate(doc_ids):
            breakdown.append({
                'document_id': doc_id,
                'lof_embeddings_score': float(lof_embeddings_scores[i]),
                'lof_network_score': float(lof_network_scores[i]),
                'lof_mixed_score': float(lof_mixed_scores[i]),
                'isolation_forest_score': float(isolation_forest_scores[i]),
                'ensemble_score': float(ensemble_scores[i]),
                'rank_by_ensemble': 0  # Will be filled after sorting
            })
        
        # Convert to DataFrame and sort by ensemble score
        breakdown_df = pd.DataFrame(breakdown)
        breakdown_df = breakdown_df.sort_values('ensemble_score', ascending=False)
        breakdown_df['rank_by_ensemble'] = range(1, len(breakdown_df) + 1)
        
        return breakdown_df.head(top_k)
    
    def print_outlier_score_summary(self, top_k: int = 10):
        """
        Print a formatted summary of Multi-LOF outlier scores.
        
        Args:
            top_k: Number of top outliers to display
        """
        breakdown_df = self.get_detailed_outlier_breakdown(top_k)
        
        print(f"\n" + "=" * 90)
        print(f"TOP {top_k} OUTLIERS - MULTI-LOF DETAILED SCORE BREAKDOWN")
        print("=" * 90)
        print(f"{'Rank':<5} {'Document ID':<20} {'LOF-Emb':<10} {'LOF-Net':<10} {'LOF-Mix':<10} {'IsolFor':<10} {'Ensemble':<10}")
        print("-" * 90)
        
        for _, row in breakdown_df.iterrows():
            print(f"{row['rank_by_ensemble']:<5} "
                  f"{row['document_id']:<20} "
                  f"{row['lof_embeddings_score']:<10.4f} "
                  f"{row['lof_network_score']:<10.4f} "
                  f"{row['lof_mixed_score']:<10.4f} "
                  f"{row['isolation_forest_score']:<10.4f} "
                  f"{row['ensemble_score']:<10.4f}")
        
        print("=" * 90)
        print("LOF-Emb = LOF on Embeddings (semantic outliers)")
        print("LOF-Net = LOF on Network Features (structural outliers)")
        print("LOF-Mix = LOF on Mixed Features (hybrid outliers)")
        print("IsolFor = Isolation Forest (global anomalies)")
        print("Ensemble = Multi-LOF weighted combination of all methods")
        print("=" * 90)
    
    def print_method_comparison(self):
        """Print formatted Multi-LOF method comparison."""
        comparison = self.get_method_comparison()
        
        print(f"\n" + "=" * 75)
        print("MULTI-LOF METHOD COMPARISON")
        print("=" * 75)
        print(f"{'Method':<25} {'Mean Score':<12} {'Std Score':<12} {'Score Range':<12}")
        print("-" * 75)
        
        for _, row in comparison.iterrows():
            print(f"{row['method']:<25} {row['mean_score']:<12.4f} "
                  f"{row['std_score']:<12.4f} {row['score_range']:<12.4f}")
    
    def _create_ranked_results(self, scores: np.ndarray, doc_ids: np.ndarray,
                             lof_embeddings_scores: np.ndarray, lof_network_scores: np.ndarray,
                             lof_mixed_scores: np.ndarray, isolation_forest_scores: np.ndarray,
                             ensemble_scores: np.ndarray, top_k: Optional[int]) -> list:
        """Create ranked results with detailed Multi-LOF information."""
        # Rank by score for top_k selection
        score_rank_pairs = list(zip(scores, range(len(scores))))
        score_rank_pairs.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for rank, (score, i) in enumerate(score_rank_pairs):
            if top_k is None or rank < top_k:
                doc_id = doc_ids[i]
                doc_features = self.features_df[
                    self.features_df['openalex_id'] == doc_id
                ].iloc[0]
                
                result = {
                    'rank': rank + 1,
                    'document_id': doc_id,
                    'outlier_score': float(score),
                    'lof_embeddings_score': float(lof_embeddings_scores[i]),
                    'lof_network_score': float(lof_network_scores[i]),
                    'lof_mixed_score': float(lof_mixed_scores[i]),
                    'isolation_forest_score': float(isolation_forest_scores[i]),
                    'ensemble_score': float(ensemble_scores[i]),
                    # Network features
                    'degree': float(doc_features.get('degree', 0)),
                    'pagerank': float(doc_features.get('pagerank', 0)),
                    'betweenness': float(doc_features.get('betweenness', 0)),
                    'clustering': float(doc_features.get('clustering', 0)),
                    # Semantic features
                    'semantic_similarity_to_relevant': float(doc_features.get('semantic_similarity_to_relevant', 0)),
                    'semantic_isolation_score': float(doc_features.get('semantic_isolation_score', 1)),
                }
                
                results.append(result)
        
        return results 