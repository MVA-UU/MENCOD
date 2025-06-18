"""
Hybrid Outlier Detection Model for Systematic Reviews

This module combines multiple detection methods to identify relevant documents
that are missed by content-based ranking algorithms (outliers).

The hybrid model integrates:
1. Citation Network Analysis (with semantic embeddings)
2. Confidence Calibration Analysis
3. Content Similarity Analysis

All models use continuous scaling and adaptive weighting for generalizability.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm

# Import sub-models
from models.CitationNetwork import CitationNetworkModel
from models.ConfidenceCalibration import ConfidenceCalibrationModel
from models.ContentSimilarity import ContentSimilarityModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfiguration:
    """Configuration for individual model settings."""
    enable_citation_network: bool = True
    enable_confidence_calibration: bool = True
    enable_content_similarity: bool = True
    enable_gpu_acceleration: bool = True
    enable_semantic_embeddings: bool = True


@dataclass
class ModelWeights:
    """Weights for combining different models in the hybrid system."""
    citation_network: float = 0.4
    confidence_calibration: float = 0.3
    content_similarity: float = 0.3


class HybridOutlierDetector:
    """
    Hybrid system combining multiple approaches for outlier detection.
    
    Features:
    - GPU-accelerated citation network analysis with semantic embeddings
    - Confidence calibration for overconfidence detection
    - Content similarity analysis for text pattern recognition
    - Adaptive weighting based on dataset characteristics
    - Continuous scaling without hard thresholds
    - Individual model enable/disable capability
    """
    
    def __init__(self, 
                 dataset_name: Optional[str] = None,
                 model_config: Optional[ModelConfiguration] = None,
                 model_weights: Optional[ModelWeights] = None,
                 use_adaptive_weights: bool = True,
                 baseline_sample_size: Optional[int] = None):
        """
        Initialize the hybrid outlier detection system.
        
        Args:
            dataset_name: Name of dataset to use
            model_config: Configuration for individual models
            model_weights: Weights for combining models
            use_adaptive_weights: Whether to use adaptive weighting
            baseline_sample_size: Optional sample size for citation network baseline (None = use all)
        """
        self.dataset_name = dataset_name
        self.model_config = model_config or ModelConfiguration()
        self.model_weights = model_weights or ModelWeights()
        self.use_adaptive_weights = use_adaptive_weights
        
        # Initialize models based on configuration
        self.citation_model = None
        self.confidence_model = None
        self.content_model = None
        
        if self.model_config.enable_citation_network:
            self.citation_model = CitationNetworkModel(
                dataset_name=dataset_name,
                enable_gpu=self.model_config.enable_gpu_acceleration,
                enable_semantic=self.model_config.enable_semantic_embeddings,
                baseline_sample_size=baseline_sample_size
            )
        
        if self.model_config.enable_confidence_calibration:
            self.confidence_model = ConfidenceCalibrationModel()
        
        if self.model_config.enable_content_similarity:
            self.content_model = ContentSimilarityModel(
                enable_semantic_embeddings=self.model_config.enable_semantic_embeddings
            )
        
        # State variables
        self.is_fitted = False
        self.simulation_data = None
        self.dataset_stats = {}
        self.known_relevant_docs = set()
        
        # Load datasets configuration
        self.datasets_config = self._load_datasets_config()
        
        if dataset_name:
            logger.info(f"Initialized hybrid model for dataset: {dataset_name}")
        
        self._log_configuration()
    
    def _load_datasets_config(self) -> Dict[str, Any]:
        """Load datasets configuration from JSON file."""
        project_root = self._get_project_root()
        config_path = os.path.join(project_root, 'data', 'datasets.json')
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _get_project_root(self) -> str:
        """Get the project root directory."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def _log_configuration(self):
        """Log the current model configuration."""
        enabled_models = []
        if self.model_config.enable_citation_network:
            enabled_models.append("Citation Network")
        if self.model_config.enable_confidence_calibration:
            enabled_models.append("Confidence Calibration")
        if self.model_config.enable_content_similarity:
            enabled_models.append("Content Similarity")
        
        logger.info(f"Enabled models: {', '.join(enabled_models)}")
        logger.info(f"GPU acceleration: {self.model_config.enable_gpu_acceleration}")
        logger.info(f"Semantic embeddings: {self.model_config.enable_semantic_embeddings}")
        logger.info(f"Adaptive weighting: {self.use_adaptive_weights}")
    
    def _load_simulation_data(self, dataset_name: str) -> pd.DataFrame:
        """Load simulation data for the specified dataset."""
        project_root = self._get_project_root()
        simulation_path = os.path.join(project_root, 'data', 'simulations', f'{dataset_name}.csv')
        
        if not os.path.exists(simulation_path):
            raise FileNotFoundError(f"Simulation file not found: {simulation_path}")
        
        return pd.read_csv(simulation_path)
    
    def fit(self, simulation_df: Optional[pd.DataFrame] = None, 
            dataset_name: Optional[str] = None) -> 'HybridOutlierDetector':
        """
        Fit all enabled models on the simulation data.
        
        Args:
            simulation_df: Optional DataFrame with simulation results
            dataset_name: Optional dataset name
        
        Returns:
            self: Returns the fitted hybrid detector
        """
        start_time = time.time()
        
        # Resolve dataset name and load data
        if dataset_name:
            self.dataset_name = dataset_name
        
        if not self.dataset_name:
            self.dataset_name = self._prompt_dataset_selection()
        
        if simulation_df is None:
            simulation_df = self._load_simulation_data(self.dataset_name)
        
        self.simulation_data = simulation_df.copy()
        
        logger.info("="*60)
        logger.info("FITTING HYBRID OUTLIER DETECTION SYSTEM")
        logger.info("="*60)
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Documents: {len(simulation_df)}")
        
        # Identify known relevant documents
        self.known_relevant_docs = set(
            simulation_df[simulation_df['label_included'] == 1]['openalex_id'].tolist()
        )
        logger.info(f"Relevant documents: {len(self.known_relevant_docs)}")
        
        # Analyze dataset characteristics
        self._analyze_dataset_characteristics()
        
        # Apply adaptive weighting if enabled
        if self.use_adaptive_weights:
            self._optimize_model_weights()
        
        # Fit individual models
        fitted_models = []
        
        if self.citation_model:
            logger.info("\n" + "="*40)
            logger.info("FITTING CITATION NETWORK MODEL")
            logger.info("="*40)
            self.citation_model.fit(simulation_df, self.dataset_name)
            fitted_models.append("Citation Network")
        
        if self.confidence_model:
            logger.info("\n" + "="*40)
            logger.info("FITTING CONFIDENCE CALIBRATION MODEL")
            logger.info("="*40)
            self.confidence_model.fit(simulation_df, self.dataset_name)
            fitted_models.append("Confidence Calibration")
        
        if self.content_model:
            logger.info("\n" + "="*40)
            logger.info("FITTING CONTENT SIMILARITY MODEL")
            logger.info("="*40)
            self.content_model.fit(simulation_df, self.dataset_name)
            fitted_models.append("Content Similarity")
        
        self.is_fitted = True
        
        # Log completion
        total_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("HYBRID SYSTEM FITTING COMPLETE")
        logger.info("="*60)
        logger.info(f"Total fitting time: {total_time:.2f} seconds")
        logger.info(f"Fitted models: {', '.join(fitted_models)}")
        logger.info(f"Model weights: Citation={self.model_weights.citation_network:.3f}, "
                   f"Confidence={self.model_weights.confidence_calibration:.3f}, "
                   f"Content={self.model_weights.content_similarity:.3f}")
        
        return self
    
    def _prompt_dataset_selection(self) -> str:
        """Prompt user to select a dataset."""
        dataset_names = list(self.datasets_config.keys())
        
        print("\nAvailable datasets:")
        for i, name in enumerate(dataset_names, 1):
            print(f"{i}. {name}")
        
        while True:
            try:
                selection = int(input("\nSelect dataset (enter number): "))
                if 1 <= selection <= len(dataset_names):
                    return dataset_names[selection-1]
                else:
                    print(f"Please enter a number between 1 and {len(dataset_names)}")
            except ValueError:
                print("Please enter a valid number")
    
    def _analyze_dataset_characteristics(self):
        """Analyze dataset characteristics to guide adaptive weighting."""
        total_docs = len(self.simulation_data)
        num_relevant = len(self.known_relevant_docs)
        rel_ratio = num_relevant / total_docs if total_docs > 0 else 0
        
        # Calculate continuous metrics
        self.dataset_stats = {
            'total_documents': total_docs,
            'relevant_documents': num_relevant,
            'relevant_ratio': rel_ratio,
            'sparsity_factor': 1 - min(0.9, max(0.1, rel_ratio * 10)),
            'size_factor': min(1.0, total_docs / 5000)
        }
        
        # Analyze text characteristics
        if 'abstract' in self.simulation_data.columns:
            has_abstract = self.simulation_data['abstract'].notna()
            abstract_lengths = self.simulation_data.loc[has_abstract, 'abstract'].astype(str).apply(len)
            
            self.dataset_stats.update({
                'abstract_availability': has_abstract.mean(),
                'mean_abstract_length': abstract_lengths.mean() if not abstract_lengths.empty else 0,
                'text_richness_factor': min(1.0, (has_abstract.mean() * 
                                                (abstract_lengths.mean() / 1500 if not abstract_lengths.empty else 0)))
            })
        else:
            self.dataset_stats.update({
                'abstract_availability': 0.0,
                'mean_abstract_length': 0.0,
                'text_richness_factor': 0.0
            })
        
        logger.info(f"\nDataset Characteristics:")
        logger.info(f"  Documents: {self.dataset_stats['total_documents']}")
        logger.info(f"  Relevant: {self.dataset_stats['relevant_documents']} ({self.dataset_stats['relevant_ratio']:.4f})")
        logger.info(f"  Sparsity factor: {self.dataset_stats['sparsity_factor']:.4f}")
        logger.info(f"  Size factor: {self.dataset_stats['size_factor']:.4f}")
        logger.info(f"  Abstract availability: {self.dataset_stats['abstract_availability']:.4f}")
        logger.info(f"  Text richness: {self.dataset_stats['text_richness_factor']:.4f}")
    
    def _optimize_model_weights(self):
        """Optimize model weights using continuous scaling functions."""
        # Extract dataset characteristics
        sparsity = self.dataset_stats['sparsity_factor']
        text_richness = self.dataset_stats.get('text_richness_factor', 0.0)
        size_factor = self.dataset_stats['size_factor']
        
        # Start with equal weights for enabled models
        enabled_models = sum([
            self.model_config.enable_citation_network,
            self.model_config.enable_confidence_calibration,
            self.model_config.enable_content_similarity
        ])
        
        if enabled_models == 0:
            logger.warning("No models enabled!")
            return
        
        base_weight = 1.0 / enabled_models
        
        # Initialize weights
        citation_weight = base_weight if self.model_config.enable_citation_network else 0.0
        confidence_weight = base_weight if self.model_config.enable_confidence_calibration else 0.0
        content_weight = base_weight if self.model_config.enable_content_similarity else 0.0
        
        # Adaptive adjustments based on dataset characteristics
        if self.model_config.enable_citation_network:
            # Citation network more important for sparse datasets
            citation_adjustment = sparsity * 0.3
            citation_weight += citation_adjustment
        
        if self.model_config.enable_content_similarity:
            # Content similarity more important for text-rich datasets
            content_adjustment = text_richness * 0.3
            content_weight += content_adjustment
        
        if self.model_config.enable_confidence_calibration:
            # Confidence calibration more stable across different conditions
            confidence_adjustment = (1 - sparsity) * 0.1
            confidence_weight += confidence_adjustment
        
        # Normalize weights to sum to 1
        total_weight = citation_weight + confidence_weight + content_weight
        if total_weight > 0:
            self.model_weights.citation_network = citation_weight / total_weight
            self.model_weights.confidence_calibration = confidence_weight / total_weight
            self.model_weights.content_similarity = content_weight / total_weight
        
        logger.info(f"\nAdaptive Model Weights:")
        logger.info(f"  Citation Network: {self.model_weights.citation_network:.3f}")
        logger.info(f"  Confidence Calibration: {self.model_weights.confidence_calibration:.3f}")
        logger.info(f"  Content Similarity: {self.model_weights.content_similarity:.3f}")
    
    def extract_features(self, target_documents: List[str]) -> pd.DataFrame:
        """
        Extract features from all enabled models for target documents.
        
        Args:
            target_documents: List of document IDs to extract features for
        
        Returns:
            DataFrame with combined features from all models
        """
        if not self.is_fitted:
            raise ValueError("Hybrid model must be fitted before extracting features")
        
        logger.info(f"Extracting features for {len(target_documents)} documents from all models")
        
        # Initialize with document IDs
        combined_features = pd.DataFrame({'openalex_id': target_documents})
        
        # Extract features from each enabled model
        if self.citation_model:
            logger.info("Extracting citation network features...")
            citation_features = self.citation_model.extract_features(target_documents)
            combined_features = pd.merge(combined_features, citation_features, on='openalex_id', how='left')
        
        if self.confidence_model:
            logger.info("Extracting confidence calibration features...")
            confidence_features = self.confidence_model.extract_features(target_documents)
            combined_features = pd.merge(combined_features, confidence_features, on='openalex_id', how='left')
        
        if self.content_model:
            logger.info("Extracting content similarity features...")
            content_features = self.content_model.extract_features(target_documents)
            combined_features = pd.merge(combined_features, content_features, on='openalex_id', how='left')
        
        # Fill any missing values with zeros
        numeric_columns = combined_features.select_dtypes(include=[np.number]).columns
        combined_features[numeric_columns] = combined_features[numeric_columns].fillna(0)
        
        logger.info(f"Combined features shape: {combined_features.shape}")
        return combined_features
    
    def predict_relevance_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Generate hybrid relevance scores for target documents.
        
        Args:
            target_documents: List of document IDs to score
        
        Returns:
            Dictionary mapping document IDs to relevance scores (0-1)
        """
        if not self.is_fitted:
            return {doc_id: 0.0 for doc_id in target_documents}
        
        logger.info(f"Computing hybrid relevance scores for {len(target_documents)} documents")
        
        # Get scores from each enabled model
        model_scores = {}
        
        if self.citation_model:
            logger.info("Computing citation network scores...")
            model_scores['citation'] = self.citation_model.predict_relevance_scores(target_documents)
        
        if self.confidence_model:
            logger.info("Computing confidence calibration scores...")
            model_scores['confidence'] = self.confidence_model.predict_relevance_scores(target_documents)
        
        if self.content_model:
            logger.info("Computing content similarity scores...")
            model_scores['content'] = self.content_model.predict_relevance_scores(target_documents)
        
        # Combine scores using weighted average
        logger.info("Combining scores with adaptive weights...")
        combined_scores = {}
        
        for doc_id in target_documents:
            weighted_score = 0.0
            total_weight = 0.0
            
            if 'citation' in model_scores and self.model_weights.citation_network > 0:
                score = model_scores['citation'].get(doc_id, 0.0)
                weighted_score += self.model_weights.citation_network * score
                total_weight += self.model_weights.citation_network
            
            if 'confidence' in model_scores and self.model_weights.confidence_calibration > 0:
                score = model_scores['confidence'].get(doc_id, 0.0)
                weighted_score += self.model_weights.confidence_calibration * score
                total_weight += self.model_weights.confidence_calibration
            
            if 'content' in model_scores and self.model_weights.content_similarity > 0:
                score = model_scores['content'].get(doc_id, 0.0)
                weighted_score += self.model_weights.content_similarity * score
                total_weight += self.model_weights.content_similarity
            
            # Normalize by total weight
            if total_weight > 0:
                combined_scores[doc_id] = weighted_score / total_weight
            else:
                combined_scores[doc_id] = 0.0
        
        # Additional ensemble processing for improved performance
        combined_scores = self._apply_ensemble_post_processing(combined_scores, model_scores)
        
        return combined_scores
    
    def _apply_ensemble_post_processing(self, combined_scores: Dict[str, float], 
                                      model_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Apply post-processing to improve ensemble performance."""
        # Calculate score statistics for normalization
        if not combined_scores:
            return combined_scores
        
        score_values = list(combined_scores.values())
        score_mean = np.mean(score_values)
        score_std = np.std(score_values)
        
        # Apply consensus boosting: higher scores when models agree
        processed_scores = {}
        
        for doc_id, base_score in combined_scores.items():
            # Calculate model agreement
            individual_scores = []
            for model_type in model_scores:
                if doc_id in model_scores[model_type]:
                    individual_scores.append(model_scores[model_type][doc_id])
            
            if len(individual_scores) > 1:
                # Boost score when models agree (low variance)
                score_variance = np.var(individual_scores)
                agreement_factor = 1.0 + (0.2 * (1.0 - min(1.0, score_variance * 4)))
                
                # Apply consensus boosting
                boosted_score = base_score * agreement_factor
                
                # Normalize to maintain [0,1] range
                processed_scores[doc_id] = min(1.0, boosted_score)
            else:
                processed_scores[doc_id] = base_score
        
        return processed_scores
    
    def predict_outliers(self, target_documents: List[str], 
                        threshold: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """
        Predict outliers using dynamic thresholding.
        
        Args:
            target_documents: List of document IDs to analyze
            threshold: Optional fixed threshold (if None, uses dynamic thresholding)
        
        Returns:
            Dictionary with outlier predictions and detailed analysis
        """
        if not self.is_fitted:
            raise ValueError("Hybrid model must be fitted before predicting outliers")
        
        logger.info(f"Predicting outliers for {len(target_documents)} documents")
        
        # Get relevance scores
        relevance_scores = self.predict_relevance_scores(target_documents)
        
        # Calculate dynamic threshold if not provided
        if threshold is None:
            threshold = self._calculate_dynamic_threshold(relevance_scores)
            logger.info(f"Using dynamic threshold: {threshold:.4f}")
        else:
            logger.info(f"Using fixed threshold: {threshold:.4f}")
        
        # Identify outliers
        outliers = {}
        for doc_id, score in relevance_scores.items():
            if score >= threshold:
                # Get detailed analysis for outliers
                analysis = self.analyze_document(doc_id)
                outliers[doc_id] = {
                    'relevance_score': score,
                    'threshold': threshold,
                    'analysis': analysis
                }
        
        logger.info(f"Identified {len(outliers)} potential outliers (threshold: {threshold:.4f})")
        
        return outliers
    
    def _calculate_dynamic_threshold(self, scores: Dict[str, float]) -> float:
        """Calculate dynamic threshold based on score distribution."""
        if not scores:
            return 0.5
        
        score_values = np.array(list(scores.values()))
        
        # Use percentile-based threshold that adapts to score distribution
        # For sparse datasets, use lower percentile; for dense datasets, use higher
        sparsity = self.dataset_stats.get('sparsity_factor', 0.5)
        
        # Adaptive percentile: 75-90 based on sparsity
        target_percentile = 75 + (sparsity * 15)
        threshold = np.percentile(score_values, target_percentile)
        
        # Ensure threshold is reasonable
        threshold = max(0.1, min(0.9, threshold))
        
        return float(threshold)
    
    def analyze_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Provide comprehensive analysis of a specific document.
        
        Args:
            doc_id: Document ID to analyze
        
        Returns:
            Dictionary with detailed analysis from all models
        """
        if not self.is_fitted:
            raise ValueError("Hybrid model must be fitted before analyzing documents")
        
        analysis = {
            'document_id': doc_id,
            'model_analyses': {},
            'hybrid_score': 0.0
        }
        
        # Get analysis from each enabled model
        if self.citation_model:
            try:
                analysis['model_analyses']['citation_network'] = self.citation_model.analyze_document(doc_id)
            except Exception as e:
                logger.warning(f"Citation network analysis failed for {doc_id}: {e}")
        
        if self.confidence_model:
            try:
                analysis['model_analyses']['confidence_calibration'] = self.confidence_model.analyze_document(doc_id)
            except Exception as e:
                logger.warning(f"Confidence calibration analysis failed for {doc_id}: {e}")
        
        if self.content_model:
            try:
                analysis['model_analyses']['content_similarity'] = self.content_model.analyze_document(doc_id)
            except Exception as e:
                logger.warning(f"Content similarity analysis failed for {doc_id}: {e}")
        
        # Get hybrid score
        hybrid_scores = self.predict_relevance_scores([doc_id])
        analysis['hybrid_score'] = hybrid_scores.get(doc_id, 0.0)
        
        return analysis
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status information about all models."""
        return {
            'is_fitted': self.is_fitted,
            'dataset_name': self.dataset_name,
            'model_configuration': {
                'citation_network_enabled': self.model_config.enable_citation_network,
                'confidence_calibration_enabled': self.model_config.enable_confidence_calibration,
                'content_similarity_enabled': self.model_config.enable_content_similarity,
                'gpu_acceleration_enabled': self.model_config.enable_gpu_acceleration,
                'semantic_embeddings_enabled': self.model_config.enable_semantic_embeddings,
            },
            'model_weights': {
                'citation_network': self.model_weights.citation_network,
                'confidence_calibration': self.model_weights.confidence_calibration,
                'content_similarity': self.model_weights.content_similarity,
            },
            'dataset_stats': self.dataset_stats,
            'individual_model_status': {
                'citation_network': self.citation_model.is_fitted if self.citation_model else False,
                'confidence_calibration': self.confidence_model.is_fitted if self.confidence_model else False,
                'content_similarity': self.content_model.is_fitted if self.content_model else False,
            }
        }


def main():
    """Example usage of the HybridOutlierDetector."""
    try:
        # Initialize with different configurations
        print("=== Hybrid Outlier Detection Demo ===\n")
        
        # Configuration 1: All models enabled
        print("1. Full hybrid model (all components enabled)")
        config1 = ModelConfiguration(
            enable_citation_network=True,
            enable_confidence_calibration=True,
            enable_content_similarity=True,
            enable_gpu_acceleration=True,
            enable_semantic_embeddings=True
        )
        
        hybrid_model1 = HybridOutlierDetector(model_config=config1)
        
        # Fit the model (will prompt for dataset selection)
        hybrid_model1.fit()
        
        # Get some example documents from the simulation data
        example_docs = hybrid_model1.simulation_data['openalex_id'].head(20).tolist()
        
        # Extract features
        print("\nExtracting hybrid features...")
        features_df = hybrid_model1.extract_features(example_docs)
        print(f"Combined features shape: {features_df.shape}")
        
        # Compute hybrid scores
        print("\nComputing hybrid relevance scores...")
        scores = hybrid_model1.predict_relevance_scores(example_docs)
        
        # Show top scoring documents
        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 documents by hybrid score:")
        for doc_id, score in top_docs:
            print(f"  {doc_id}: {score:.4f}")
        
        # Predict outliers
        print("\nPredicting outliers with dynamic thresholding...")
        outliers = hybrid_model1.predict_outliers(example_docs)
        print(f"Found {len(outliers)} potential outliers")
        
        # Show model status
        print("\nModel Status:")
        status = hybrid_model1.get_model_status()
        print(f"  Dataset: {status['dataset_name']}")
        print(f"  Models fitted: {status['individual_model_status']}")
        print(f"  Weights: {status['model_weights']}")
        
        # Configuration 2: Citation network only
        print("\n" + "="*50)
        print("2. Citation network only")
        config2 = ModelConfiguration(
            enable_citation_network=True,
            enable_confidence_calibration=False,
            enable_content_similarity=False
        )
        
        # Use same dataset as previous model
        hybrid_model2 = HybridOutlierDetector(
            dataset_name=hybrid_model1.dataset_name,
            model_config=config2
        )
        hybrid_model2.fit(hybrid_model1.simulation_data)
        
        scores2 = hybrid_model2.predict_relevance_scores(example_docs[:5])
        print("Citation-only scores:", {k: f"{v:.4f}" for k, v in scores2.items()})
        
        print("\nDemo completed successfully!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 