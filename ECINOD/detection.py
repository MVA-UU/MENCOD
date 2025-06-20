"""
Outlier Detection Module

Handles core outlier detection algorithms including:
- LOF on embeddings for semantic outlier detection
- Isolation Forest for global anomaly detection
- Ensemble scoring methods
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from utils import normalize_scores, compute_ensemble_weights

logger = logging.getLogger(__name__)


class OutlierDetector:
    """Handles outlier detection algorithms and ensemble methods."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize outlier detector.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def apply_lof_to_embeddings(self, simulation_df: pd.DataFrame, 
                               embeddings: Optional[np.ndarray],
                               embeddings_metadata: Optional[Dict]) -> Dict[str, np.ndarray]:
        """
        Apply LOF directly to embeddings for semantic outlier detection.
        
        Args:
            simulation_df: DataFrame with simulation documents
            embeddings: SPECTER2 embeddings array
            embeddings_metadata: Embeddings metadata
            
        Returns:
            Dictionary with LOF scores
        """
        if embeddings is None or embeddings_metadata is None:
            logger.warning("No embeddings available for LOF analysis")
            return {'scores': np.zeros(len(simulation_df))}
        
        # Create mapping from openalex_id to embedding index
        id_to_idx = self._create_id_to_index_mapping(embeddings_metadata)
        
        # Get embeddings for documents in simulation_df
        doc_embeddings, doc_indices = self._get_document_embeddings(
            simulation_df, id_to_idx, embeddings
        )
        
        if len(doc_embeddings) == 0:
            logger.warning("No embeddings found for simulation documents")
            return {'scores': np.zeros(len(simulation_df))}
        
        # Apply LOF
        lof_scores = self._compute_lof_scores(doc_embeddings)
        
        # Map results back to full simulation dataframe
        return self._map_scores_to_dataframe(lof_scores, doc_indices, len(simulation_df))
    
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
    
    def compute_ensemble_scores(self, results: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute ensemble scores with data-driven weighting.
        
        Args:
            results: Dictionary with individual method scores
            
        Returns:
            Array of ensemble scores
        """
        logger.info("Computing ensemble scores...")
        
        # Normalize scores
        lof_scores_norm = normalize_scores(results['lof_scores'])
        if_scores_norm = normalize_scores(results['isolation_forest_scores'])
        
        # Calculate weights
        score_arrays = {
            'lof': lof_scores_norm,
            'isolation_forest': if_scores_norm,
        }
        
        weights = compute_ensemble_weights(score_arrays, method='variance')
        
        # Dynamically ensure LOF gets appropriate weight for semantic detection
        # If LOF weight is very low, gradually boost it while preserving relative importance
        if weights['lof'] < weights['isolation_forest']:
            # Boost LOF proportionally while maintaining total weight = 1
            lof_boost = min(0.15, (weights['isolation_forest'] - weights['lof']) / 2)
            weights['lof'] += lof_boost
            weights['isolation_forest'] -= lof_boost
        
        logger.info(f"Ensemble weights: LOF={weights['lof']:.3f}, IF={weights['isolation_forest']:.3f}")
        
        # Compute weighted ensemble
        ensemble_scores = (
            weights['lof'] * lof_scores_norm +
            weights['isolation_forest'] * if_scores_norm
        )
        
        return ensemble_scores
    
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
    
    def _compute_lof_scores(self, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute LOF scores for document embeddings."""
        # Dynamic n_neighbors calculation based on dataset size
        # Use sqrt rule with reasonable bounds for statistical validity
        n_docs = len(doc_embeddings)
        n_neighbors = max(3, min(int(np.sqrt(n_docs)), n_docs - 1))
        
        # Ensure we have enough neighbors for LOF to be meaningful
        if n_neighbors < 3:
            logger.warning(f"Very small dataset ({n_docs} docs), using minimum neighbors=3")
            n_neighbors = min(3, n_docs - 1)
        
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=False,
            metric='cosine',
            n_jobs=-1
        )
        
        logger.info(f"Applying LOF to embeddings: {doc_embeddings.shape} "
                   f"with {n_neighbors} neighbors (dynamic), metric=cosine")
        
        # Apply LOF to get scores
        lof.fit_predict(doc_embeddings)
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