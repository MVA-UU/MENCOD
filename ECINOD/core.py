"""
Core Citation Network Outlier Detection Module

Main orchestration class that coordinates network building, feature extraction,
outlier detection, and results analysis.
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Optional

from .network import NetworkBuilder
from .features import FeatureExtractor
from .detection import OutlierDetector
from .evaluation import ResultsAnalyzer
from utils import load_embeddings

logger = logging.getLogger(__name__)


class CitationNetworkOutlierDetector:
    """
    Main Citation Network Outlier Detection class.
    
    Streamlined outlier detection using:
    - LOF on embeddings for semantic outlier detection
    - Isolation Forest for global anomaly detection
    - NetworkX-based citation network analysis
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the outlier detector.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        
        # Initialize components
        self.network_builder = NetworkBuilder()
        self.outlier_detector = OutlierDetector(random_state=random_state)
        
        # State variables
        self.is_fitted = False
        self.embeddings = None
        self.embeddings_metadata = None
        self.feature_extractor = None
        self.results_analyzer = None
        
        # Results storage
        self.outlier_results = None
        self.feature_matrix = None
        self.features_df = None
        self.graph = None
        
    def fit_predict_outliers(self, 
                           simulation_df: pd.DataFrame,
                           dataset_name: str = None) -> Dict[str, np.ndarray]:
        """
        Main method: takes dataset and immediately returns outlier scores.
        
        Args:
            simulation_df: DataFrame with paper data and citation information
            dataset_name: Optional dataset name for loading embeddings and synergy data
            
        Returns:
            Dictionary with outlier scores and predictions from all methods
        """
        logger.info(f"Starting outlier detection for {len(simulation_df)} documents")
        start_time = time.time()
        
        # Build citation network
        self.graph = self.network_builder.build_citation_network(simulation_df, dataset_name)
        logger.info(f"Built citation network: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
        
        # Load embeddings if available
        if dataset_name:
            self._load_embeddings(dataset_name)
        
        # Initialize feature extractor with embeddings
        self.feature_extractor = FeatureExtractor(
            embeddings=self.embeddings,
            embeddings_metadata=self.embeddings_metadata
        )
        
        # Extract features
        self.features_df = self.feature_extractor.extract_network_features(
            self.graph, simulation_df
        )
        logger.info(f"Extracted features: {self.features_df.shape[1]-1} features "
                   f"for {self.features_df.shape[0]} documents")
        
        # Prepare feature matrix
        self.feature_matrix = self.features_df.drop('openalex_id', axis=1).values
        features_scaled = self.outlier_detector.scale_features(self.feature_matrix)
        
        # Apply outlier detection methods
        self.outlier_results = {}
        
        # 1. LOF on embeddings
        logger.info("Applying LOF on embeddings for semantic outlier detection...")
        lof_results = self.outlier_detector.apply_lof_to_embeddings(
            simulation_df, self.embeddings, self.embeddings_metadata
        )
        self.outlier_results['lof_scores'] = lof_results['scores']
        
        # 2. Isolation Forest
        if_scores = self.outlier_detector.apply_isolation_forest(features_scaled)
        self.outlier_results['isolation_forest_scores'] = if_scores
        
        # 3. Ensemble scoring
        ensemble_scores = self.outlier_detector.compute_ensemble_scores(self.outlier_results)
        self.outlier_results['ensemble_scores'] = ensemble_scores
        
        # Add document IDs to results
        self.outlier_results['openalex_ids'] = self.features_df['openalex_id'].values
        
        # Initialize results analyzer
        self.results_analyzer = ResultsAnalyzer(self.outlier_results, self.features_df)
        
        # Mark as fitted
        self.is_fitted = True
        
        # Log summary
        total_time = time.time() - start_time
        logger.info(f"Outlier scoring completed in {total_time:.2f}s")
        logger.info(f"Generated outlier scores for all {len(simulation_df)} documents")
        
        return self.outlier_results
    
    def get_outlier_documents(self, method: str = 'ensemble', 
                            top_k: int = None) -> pd.DataFrame:
        """Get outlier documents with detailed information."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting outlier documents")
        
        return self.results_analyzer.get_outlier_documents(method, top_k)
    
    def get_method_comparison(self) -> pd.DataFrame:
        """Compare results across all outlier detection methods."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before comparing methods")
        
        return self.results_analyzer.get_method_comparison()
    
    def get_detailed_outlier_breakdown(self, top_k: int = 10) -> pd.DataFrame:
        """Get detailed breakdown of outlier scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting outlier breakdown")
        
        return self.results_analyzer.get_detailed_outlier_breakdown(top_k)
    
    def print_outlier_score_summary(self, top_k: int = 10):
        """Print formatted summary of outlier scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before printing summary")
        
        self.results_analyzer.print_outlier_score_summary(top_k)
    
    def print_method_comparison(self):
        """Print formatted method comparison."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before printing comparison")
        
        self.results_analyzer.print_method_comparison()
    
    def _load_embeddings(self, dataset_name: str):
        """Load embeddings for semantic analysis."""
        self.embeddings, self.embeddings_metadata = load_embeddings(dataset_name)
        
        if self.embeddings is not None:
            logger.info(f"Loaded embeddings: {self.embeddings.shape}")
        else:
            logger.warning("No embeddings available for semantic analysis") 