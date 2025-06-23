"""
Outlier Detection Module

Handles core outlier detection algorithms including:
- LOF on embeddings for semantic outlier detection
- LOF on network features for structural outlier detection
- LOF on mixed features for hybrid outlier detection
- Isolation Forest for global anomaly detection
- Multi-LOF ensemble scoring methods
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from utils import normalize_scores, compute_ensemble_weights, robust_reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class OutlierDetector:
    """Handles outlier detection algorithms and ensemble methods."""
    
    def __init__(self, random_state: int = 42, use_rrf: bool = False):
        """
        Initialize outlier detector.
        
        Args:
            random_state: Random state for reproducibility
            use_rrf: Whether to use Robust Reciprocal Rank Fusion for ensemble scoring
        """
        self.random_state = random_state
        self.use_rrf = use_rrf
        self.scaler = StandardScaler()
    
    def apply_lof_to_embeddings(self, simulation_df: pd.DataFrame, 
                               embeddings: Optional[np.ndarray],
                               embeddings_metadata: Optional[Dict]) -> Dict[str, np.ndarray]:
        """
        Apply LOF directly to embeddings for semantic outlier detection.
        
        This method processes only simulation-eligible papers for outlier ranking,
        ensuring that background papers from the full dataset are not considered
        as potential outliers.
        
        Args:
            simulation_df: DataFrame with simulation documents (eligible for ranking)
            embeddings: SPECTER2 embeddings array 
            embeddings_metadata: Embeddings metadata
            
        Returns:
            Dictionary with LOF scores for simulation papers only
        """
        if embeddings is None or embeddings_metadata is None:
            logger.warning("No embeddings available for LOF analysis")
            return {'scores': np.zeros(len(simulation_df))}
        
        logger.info(f"Applying LOF to embeddings for {len(simulation_df)} simulation papers")
        
        # Create mapping from openalex_id to embedding index
        id_to_idx = self._create_id_to_index_mapping(embeddings_metadata)
        
        # Get embeddings for documents in simulation_df (eligible papers only)
        doc_embeddings, doc_indices = self._get_document_embeddings(
            simulation_df, id_to_idx, embeddings
        )
        
        if len(doc_embeddings) == 0:
            logger.warning("No embeddings found for simulation documents")
            return {'scores': np.zeros(len(simulation_df))}
        
        logger.info(f"Found embeddings for {len(doc_embeddings)} out of {len(simulation_df)} simulation papers")
        
        # Apply LOF to simulation papers only
        lof_scores = self._compute_lof_scores(doc_embeddings, metric='cosine', method_name='embeddings')
        
        # Map results back to full simulation dataframe
        return self._map_scores_to_dataframe(lof_scores, doc_indices, len(simulation_df))
    
    def apply_lof_to_network_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Apply LOF to network features for structural outlier detection.
        
        Args:
            feature_matrix: Scaled network feature matrix
            
        Returns:
            Array of structural anomaly scores
        """
        logger.info("Applying LOF to network features for structural outlier detection...")
        
        if feature_matrix.shape[0] == 0:
            logger.warning("Empty feature matrix for network LOF")
            return np.zeros(0)
        
        # Apply LOF to network features using euclidean distance (better for mixed numerical features)
        lof_scores = self._compute_lof_scores(feature_matrix, metric='euclidean', method_name='network_features')
        
        return lof_scores
    
    def apply_lof_to_mixed_features(self, simulation_df: pd.DataFrame,
                                  embeddings: Optional[np.ndarray],
                                  embeddings_metadata: Optional[Dict],
                                  feature_matrix: np.ndarray) -> np.ndarray:
        """
        Apply LOF to mixed features (embeddings + network features) for hybrid outlier detection.
        
        Args:
            simulation_df: DataFrame with simulation documents
            embeddings: SPECTER2 embeddings array
            embeddings_metadata: Embeddings metadata  
            feature_matrix: Scaled network feature matrix
            
        Returns:
            Array of hybrid anomaly scores
        """
        logger.info("Applying LOF to mixed features for hybrid outlier detection...")
        
        if embeddings is None or embeddings_metadata is None:
            logger.warning("No embeddings available for mixed features LOF, using network features only")
            return self.apply_lof_to_network_features(feature_matrix)
        
        # Get embeddings for simulation documents
        id_to_idx = self._create_id_to_index_mapping(embeddings_metadata)
        doc_embeddings, doc_indices = self._get_document_embeddings(
            simulation_df, id_to_idx, embeddings
        )
        
        if len(doc_embeddings) == 0:
            logger.warning("No embeddings found for mixed features LOF, using network features only")
            return self.apply_lof_to_network_features(feature_matrix)
        
        # Create mixed feature matrix: align embeddings with network features
        mixed_features = self._create_mixed_feature_matrix(
            doc_embeddings, doc_indices, feature_matrix
        )
        
        logger.info(f"Created mixed feature matrix: {mixed_features.shape[0]} samples, {mixed_features.shape[1]} features "
                   f"({doc_embeddings.shape[1]} embedding + {feature_matrix.shape[1]} network)")
        
        # Apply LOF to mixed features
        lof_scores = self._compute_lof_scores(mixed_features, metric='euclidean', method_name='mixed_features')
        
        return lof_scores

    def apply_isolation_forest(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Apply Isolation Forest to feature matrix.
        
        Args:
            feature_matrix: Scaled feature matrix
            
        Returns:
            Array of anomaly scores
        """
        logger.info("Applying Isolation Forest...")
        
        isolation_forest = IsolationForest(
            n_estimators=100,
            contamination='auto',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        isolation_forest.fit(feature_matrix)
        if_scores = isolation_forest.decision_function(feature_matrix)
        
        # Convert to anomaly scores (higher = more anomalous)
        return -if_scores
    
    def compute_multi_lof_ensemble_scores(self, results: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute ensemble scores from multiple LOF methods and Isolation Forest.
        
        Args:
            results: Dictionary with scores from different methods
            
        Returns:
            Array of ensemble scores
        """
        if self.use_rrf:
            logger.info("Computing Multi-LOF ensemble scores using Robust Reciprocal Rank Fusion...")
            
            # Extract raw scores for RRF (no normalization needed for RRF)
            score_arrays = {}
            for method_key in results.keys():
                if method_key.endswith('_scores') and method_key != 'ensemble_scores':
                    method_name = method_key.replace('_scores', '')
                    score_arrays[method_name] = results[method_key]
            
            # Debug: Check if methods produce different rankings
            logger.info("Score distribution analysis:")
            for method_name, scores in score_arrays.items():
                unique_scores = len(np.unique(scores))
                logger.info(f"  {method_name}: min={np.min(scores):.4f}, max={np.max(scores):.4f}, "
                           f"unique_values={unique_scores}")
            
            # Apply pure RRF (no weights - let RRF handle it automatically)
            ensemble_scores = robust_reciprocal_rank_fusion(score_arrays)
            
            return ensemble_scores
        else:
            logger.info("Computing Multi-LOF ensemble scores using variance-weighted combination...")
            
            # Normalize all scores
            normalized_scores = {}
            for method_key in results.keys():
                if method_key.endswith('_scores') and method_key != 'ensemble_scores':
                    normalized_scores[method_key] = normalize_scores(results[method_key])
            
            # Create score arrays for weight calculation
            score_arrays = {
                method_key.replace('_scores', ''): scores 
                for method_key, scores in normalized_scores.items()
            }
            
            # Calculate weights using variance-based method
            weights = compute_ensemble_weights(score_arrays, method='variance')
            
            # Apply method-specific adjustments for Multi-LOF
            #weights = self._adjust_multi_lof_weights(weights)
            
            logger.info("Multi-LOF ensemble weights:")
            for method, weight in weights.items():
                logger.info(f"  {method}: {weight:.3f}")
            
            # Compute weighted ensemble
            ensemble_scores = np.zeros(len(list(normalized_scores.values())[0]))
            
            for method_key, scores in normalized_scores.items():
                method_name = method_key.replace('_scores', '')
                if method_name in weights:
                    ensemble_scores += weights[method_name] * scores
            
            return ensemble_scores
    
    def _adjust_multi_lof_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Adjust weights for Multi-LOF ensemble with domain knowledge."""
        
        # Ensure semantic methods (embeddings-based) get reasonable weight
        semantic_methods = ['lof_embeddings', 'lof_mixed']
        structural_methods = ['lof_network', 'isolation_forest']
        
        # Boost semantic weight if it's too low (embeddings are high quality)
        semantic_weight = sum(weights.get(method, 0) for method in semantic_methods)
        structural_weight = sum(weights.get(method, 0) for method in structural_methods)
        
        if semantic_weight < 0.4 and 'lof_embeddings' in weights:
            # Redistribute some weight to semantic methods
            boost = min(0.15, (0.4 - semantic_weight) / 2)
            if 'lof_embeddings' in weights:
                weights['lof_embeddings'] += boost
            if 'lof_mixed' in weights:
                weights['lof_mixed'] += boost
            
            # Reduce other weights proportionally
            reduction_per_method = (2 * boost) / len([m for m in structural_methods if m in weights])
            for method in structural_methods:
                if method in weights:
                    weights[method] = max(0.05, weights[method] - reduction_per_method)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {method: w / total_weight for method, w in weights.items()}
        
        return weights
    
    def _create_mixed_feature_matrix(self, doc_embeddings: np.ndarray, 
                                   doc_indices: list, 
                                   feature_matrix: np.ndarray) -> np.ndarray:
        """Create mixed feature matrix combining embeddings and network features."""
        
        # For papers with embeddings, combine embedding + network features
        # For papers without embeddings, use network features + zero-padded embeddings
        
        n_samples = feature_matrix.shape[0]
        embedding_dim = doc_embeddings.shape[1] if len(doc_embeddings) > 0 else 0
        network_dim = feature_matrix.shape[1]
        
        # Initialize mixed feature matrix
        mixed_features = np.zeros((n_samples, embedding_dim + network_dim))
        
        # Add network features for all samples
        mixed_features[:, embedding_dim:] = feature_matrix
        
        # Add embeddings for samples that have them
        embedding_map = {doc_idx: emb_idx for emb_idx, doc_idx in enumerate(doc_indices)}
        
        for sample_idx in range(n_samples):
            if sample_idx in embedding_map:
                emb_idx = embedding_map[sample_idx]
                mixed_features[sample_idx, :embedding_dim] = doc_embeddings[emb_idx]
            # For samples without embeddings, embedding part remains zeros
        
        return mixed_features

    def _create_id_to_index_mapping(self, embeddings_metadata: Dict) -> Dict[str, int]:
        """Create mapping from document ID to embedding index."""
        if 'documents' in embeddings_metadata:
            return {
                doc.get('openalex_id', ''): idx 
                for idx, doc in enumerate(embeddings_metadata['documents'])
            }
        else:
            return {
                doc_id: idx 
                for idx, doc_id in enumerate(embeddings_metadata.get('openalex_id', []))
            }
    
    def _get_document_embeddings(self, simulation_df: pd.DataFrame, 
                               id_to_idx: Dict[str, int], 
                               embeddings: np.ndarray) -> tuple:
        """Get embeddings for documents in simulation DataFrame."""
        doc_embeddings = []
        doc_indices = []
        
        for idx, row in simulation_df.iterrows():
            doc_id = row['openalex_id']
            if doc_id in id_to_idx:
                embedding_idx = id_to_idx[doc_id]
                doc_embeddings.append(embeddings[embedding_idx])
                doc_indices.append(idx)
        
        return np.array(doc_embeddings) if doc_embeddings else [], doc_indices
    
    def _compute_lof_scores(self, feature_matrix: np.ndarray, 
                          metric: str = 'euclidean', 
                          method_name: str = 'features') -> np.ndarray:
        """Compute LOF scores for feature matrix."""
        # Dynamic n_neighbors calculation based on dataset size
        n_samples = len(feature_matrix)
        n_neighbors = max(3, min(int(np.sqrt(n_samples)), n_samples - 1))
        
        # Ensure we have enough neighbors for LOF to be meaningful
        if n_neighbors < 3:
            logger.warning(f"Very small dataset ({n_samples} samples), using minimum neighbors=3")
            n_neighbors = min(3, n_samples - 1)
        
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=False,
            metric=metric,
            n_jobs=-1
        )
        
        logger.info(f"Applying LOF to {method_name}: {feature_matrix.shape} "
                   f"with {n_neighbors} neighbors, metric={metric}")
        
        # Apply LOF to get scores
        lof.fit_predict(feature_matrix)
        return -lof.negative_outlier_factor_
    
    def _map_scores_to_dataframe(self, scores: np.ndarray, doc_indices: list, 
                                n_docs: int) -> Dict[str, np.ndarray]:
        """Map scores back to full simulation dataframe."""
        full_scores = np.zeros(n_docs)
        
        for i, sim_idx in enumerate(doc_indices):
            full_scores[sim_idx] = scores[i]
        
        return {'scores': full_scores}
    
    def scale_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Scale feature matrix using StandardScaler."""
        # Handle NaN and infinite values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.fit_transform(feature_matrix) 