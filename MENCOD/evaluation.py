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
                 features_df: pd.DataFrame,
                 ensemble_weights: Dict[str, float] = None):
        """
        Initialize results analyzer.
        
        Args:
            outlier_results: Dictionary with outlier scores from different methods
            features_df: DataFrame with extracted features
            ensemble_weights: Dictionary with ensemble weights for each method
        """
        self.outlier_results = outlier_results
        self.features_df = features_df
        self.ensemble_weights = ensemble_weights or {}
    
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
    
    def get_document_subscores(self, document_id: str) -> Dict[str, float]:
        """
        Get individual method subscores for a specific document.
        
        Args:
            document_id: OpenAlex ID of the document
            
        Returns:
            Dictionary with individual method scores for the document
        """
        doc_ids = self.outlier_results['openalex_ids']
        
        try:
            doc_index = list(doc_ids).index(document_id)
        except ValueError:
            raise ValueError(f"Document {document_id} not found in results")
        
        # Extract individual subscores
        subscores = {
            'lof_embeddings_score': float(self.outlier_results.get('lof_embeddings_scores', [0] * len(doc_ids))[doc_index]),
            'lof_network_score': float(self.outlier_results.get('lof_network_scores', [0] * len(doc_ids))[doc_index]),
            'lof_mixed_score': float(self.outlier_results.get('lof_mixed_scores', [0] * len(doc_ids))[doc_index]),
            'isolation_forest_score': float(self.outlier_results.get('isolation_forest_scores', [0] * len(doc_ids))[doc_index]),
            'ensemble_score': float(self.outlier_results.get('ensemble_scores', [0] * len(doc_ids))[doc_index])
        }
        
        return subscores
    
    def get_comprehensive_document_analysis(self, document_id: str) -> Dict:
        """
        Get comprehensive analysis for a specific document including ranks, percentiles, and ensemble weights.
        
        Args:
            document_id: OpenAlex ID of the document
            
        Returns:
            Dictionary with comprehensive analysis including ranks, percentiles, and weights
        """
        doc_ids = self.outlier_results['openalex_ids']
        
        try:
            doc_index = list(doc_ids).index(document_id)
        except ValueError:
            raise ValueError(f"Document {document_id} not found in results")
        
        # Get all method scores
        methods = {
            'LOF-Embeddings': self.outlier_results.get('lof_embeddings_scores', [0] * len(doc_ids)),
            'LOF-Network': self.outlier_results.get('lof_network_scores', [0] * len(doc_ids)),
            'LOF-Mixed': self.outlier_results.get('lof_mixed_scores', [0] * len(doc_ids)),
            'Isolation Forest': self.outlier_results.get('isolation_forest_scores', [0] * len(doc_ids)),
            'Ensemble': self.outlier_results.get('ensemble_scores', [0] * len(doc_ids))
        }
        
        # Calculate ranks and percentiles for each method
        analysis = {
            'document_id': document_id,
            'total_documents': len(doc_ids),
            'methods': {}
        }
        
        for method_name, scores in methods.items():
            # Get this document's score
            doc_score = float(scores[doc_index])
            
            # Calculate rank (1-based, 1 = highest score)
            scores_array = np.array(scores)
            rank = int(np.sum(scores_array > doc_score) + 1)
            
            # Calculate percentile
            percentile = ((len(scores) - rank + 1) / len(scores)) * 100
            
            analysis['methods'][method_name] = {
                'raw_score': doc_score,
                'rank': rank,
                'percentile': percentile
            }
        
        # Add ensemble weights information
        analysis['ensemble_weights'] = self.ensemble_weights.copy()
        
        return analysis
    
    def print_document_subscores(self, document_id: str):
        """
        Print formatted individual method subscores for a specific document.
        
        Args:
            document_id: OpenAlex ID of the document
        """
        try:
            subscores = self.get_document_subscores(document_id)
            
            print(f"\nDETAILED SUBSCORES FOR: {document_id}")
            print("-" * 50)
            print(f"LOF-Embeddings:    {subscores['lof_embeddings_score']:.4f}")
            print(f"LOF-Network:       {subscores['lof_network_score']:.4f}")
            print(f"LOF-Mixed:         {subscores['lof_mixed_score']:.4f}")
            print(f"Isolation Forest:  {subscores['isolation_forest_score']:.4f}")
            print(f"Ensemble Score:    {subscores['ensemble_score']:.4f}")
            print("-" * 50)
            
        except ValueError as e:
            print(f"Error: {e}")
    
    def print_thesis_analysis(self, document_id: str):
        """
        Print comprehensive thesis-level analysis showing detailed score decomposition.
        
        Args:
            document_id: OpenAlex ID of the document
        """
        try:
            analysis = self.get_comprehensive_document_analysis(document_id)
            
            print(f"\n" + "=" * 80)
            print(f"COMPREHENSIVE ENSEMBLE SCORE ANALYSIS FOR THESIS")
            print(f"Document: {document_id}")
            print("=" * 80)
            
            # Display ensemble weights
            print(f"\nENSEMBLE WEIGHTS USED:")
            print("-" * 40)
            weights = analysis['ensemble_weights']
            weight_sum = sum(weights.values()) if weights else 0
            
            if weights:
                for method, weight in weights.items():
                    percentage = (weight / weight_sum * 100) if weight_sum > 0 else 0
                    print(f"  {method:<20}: {weight:.4f} ({percentage:.1f}%)")
            else:
                print("  Using Robust Reciprocal Rank Fusion (RRF) - automatic weighting")
            
            # Display detailed method breakdown
            print(f"\nMETHOD-BY-METHOD BREAKDOWN:")
            print("-" * 80)
            print(f"{'Method':<20} {'Raw Score':<12} {'Rank':<8} {'Percentile':<12} {'Weight':<10}")
            print("-" * 80)
            
            total_docs = analysis['total_documents']
            
            # Order methods for logical presentation
            method_order = ['LOF-Embeddings', 'LOF-Network', 'LOF-Mixed', 'Isolation Forest', 'Ensemble']
            
            for method in method_order:
                if method in analysis['methods']:
                    method_data = analysis['methods'][method]
                    weight = weights.get(method.lower().replace('-', '_'), 0.0) if method != 'Ensemble' else 1.0
                    
                    print(f"{method:<20} "
                          f"{method_data['raw_score']:<12.4f} "
                          f"{method_data['rank']:<8} "
                          f"{method_data['percentile']:<12.1f}% "
                          f"{weight:<10.4f}")
            
            print("-" * 80)
            
            # Ensemble construction explanation
            print(f"\nENSEMBLE SCORE CONSTRUCTION:")
            print("-" * 40)
            if weights:
                ensemble_score = analysis['methods']['Ensemble']['raw_score']
                print(f"Final Ensemble Score: {ensemble_score:.4f}")
                print(f"Calculated as weighted combination of normalized individual scores")
                
                # Show contribution of each method
                print(f"\nEstimated Contributions:")
                for method in ['LOF-Embeddings', 'LOF-Network', 'LOF-Mixed', 'Isolation Forest']:
                    if method in analysis['methods']:
                        weight = weights.get(method.lower().replace('-', '_'), 0.0)
                        raw_score = analysis['methods'][method]['raw_score']
                        contribution = weight * 100  # Approximate contribution percentage
                        print(f"  {method:<20}: {contribution:.1f}% weight")
            else:
                print("Using Robust Reciprocal Rank Fusion (RRF)")
                print("Scores are combined using rank-based fusion with automatic weighting")
            
            # Performance summary
            print(f"\nPERFORMANCE SUMMARY:")
            print("-" * 40)
            ensemble_percentile = analysis['methods']['Ensemble']['percentile']
            ensemble_rank = analysis['methods']['Ensemble']['rank']
            
            print(f"Overall Rank: {ensemble_rank} out of {total_docs}")
            print(f"Overall Percentile: {ensemble_percentile:.1f}%")
            
            if ensemble_percentile >= 95:
                performance = "Excellent outlier detection"
            elif ensemble_percentile >= 90:
                performance = "Very good outlier detection"
            elif ensemble_percentile >= 80:
                performance = "Good outlier detection"
            else:
                performance = "Moderate outlier score"
            
            print(f"Performance Level: {performance}")
            
            print("=" * 80)
            
        except ValueError as e:
            print(f"Error: {e}")
    
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