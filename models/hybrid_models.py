"""
Hybrid Outlier Detection System

This module combines multiple detection methods to identify relevant documents
that are missed by content-based ranking algorithms (outliers).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .citation_network import CitationNetworkModel
from .content_similarity import ContentSimilarityModel
from .confidence_calibration import ConfidenceCalibrationModel


@dataclass
class ModelWeights:
    """Configuration for model weights in hybrid system."""
    citation_network: float = 0.2
    content_similarity: float = 0.6
    confidence_calibration: float = 0.2
    # metadata_analysis: float = 0.0
    # temporal_patterns: float = 0.0


class HybridOutlierDetector:
    """
    Hybrid system combining multiple approaches for outlier detection.
    
    Currently includes:
    - Citation network analysis
    - Content similarity analysis 
    - Confidence calibration analysis
    
    Future extensions:
    - Metadata pattern analysis
    - Temporal publication patterns
    - Author/venue analysis
    """
    
    def __init__(self, 
                 dataset_name: str = "Appenzeller-Herzog_2019",
                 model_weights: Optional[ModelWeights] = None):
        """
        Initialize the hybrid outlier detection system.
        
        Args:
            dataset_name: Name of the synergy dataset to use
            model_weights: Weights for combining different models
        """
        self.dataset_name = dataset_name
        self.model_weights = model_weights or ModelWeights()
        
        # Initialize individual models
        self.citation_model = CitationNetworkModel(dataset_name)
        self.content_model = ContentSimilarityModel(dataset_name)
        self.confidence_model = ConfidenceCalibrationModel()
        
        # Future models will be initialized here
        # self.metadata_model = MetadataAnalysisModel()
        
        self.is_fitted = False
        self.known_relevant_docs = set()
        self.simulation_data = None
    
    def fit(self, simulation_df: pd.DataFrame) -> 'HybridOutlierDetector':
        """
        Fit all models on the simulation data.
        
        Args:
            simulation_df: DataFrame with simulation results
        
        Returns:
            self: Returns the fitted hybrid detector
        """
        print("Fitting Hybrid Outlier Detection System...")
        print(f"Dataset: {self.dataset_name}")
        print(f"Simulation data: {len(simulation_df)} documents")
        
        # Store simulation data for analysis
        self.simulation_data = simulation_df.copy()
        
        # Identify known relevant documents
        self.known_relevant_docs = set(
            simulation_df[simulation_df['label_included'] == 1]['openalex_id'].tolist()
        )
        
        print(f"Known relevant documents: {len(self.known_relevant_docs)}")
        
        # Fit individual models
        print("\n1. Fitting Citation Network Model...")
        self.citation_model.fit(simulation_df)
        print("\n2. Fitting Content Similarity Model...")
        self.content_model.fit(simulation_df)
        print("\n3. Fitting Confidence Calibration Model...")
        self.confidence_model.fit(simulation_df)
        
        # Future model fitting will go here
        # print("4. Fitting Metadata Analysis Model...")
        # self.metadata_model.fit(simulation_df)
        
        self.is_fitted = True
        print("\nHybrid system successfully fitted!")
        
        return self
    
    def predict_outliers(self, 
                        candidate_documents: List[str],
                        threshold: float = 0.1) -> Dict[str, Dict[str, Any]]:
        """
        Identify potential outlier documents from candidates.
        
        Args:
            candidate_documents: List of OpenAlex IDs to evaluate
            threshold: Minimum combined score to consider as potential outlier
        
        Returns:
            Dictionary with outlier analysis for each candidate
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting outliers")
        
        print(f"Analyzing {len(candidate_documents)} candidate documents...")
        
        # Get combined relevance scores
        combined_scores = self.predict_relevance_scores(candidate_documents)
        
        # Get detailed features from all models
        detailed_features = self.get_combined_features(candidate_documents)
        
        # Identify outliers
        outliers = {}
        
        for doc_id in candidate_documents:
            score = combined_scores.get(doc_id, 0.0)
            
            if score >= threshold:
                # Get document info from simulation if available
                doc_info = self._get_document_info(doc_id)
                
                outliers[doc_id] = {
                    'combined_score': score,
                    'individual_scores': self._get_individual_scores(doc_id),
                    'features': detailed_features[detailed_features['openalex_id'] == doc_id].iloc[0].to_dict(),
                    'document_info': doc_info,
                    'outlier_reasons': self._analyze_outlier_reasons(doc_id, detailed_features)
                }
        
        print(f"Identified {len(outliers)} potential outliers (threshold: {threshold})")
        
        return outliers
    
    def predict_relevance_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Generate combined relevance scores from all models.
        
        Args:
            target_documents: List of OpenAlex IDs to score
        
        Returns:
            Dictionary mapping document IDs to combined relevance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting scores")
        
        # Get scores from individual models
        citation_scores = self.citation_model.predict_relevance_scores(target_documents)
        content_scores = self.content_model.predict_relevance_scores(target_documents)
        confidence_scores = self.confidence_model.predict_relevance_scores(target_documents)
        
        # Combine scores using weights
        combined_scores = {}
        total_weight = (self.model_weights.citation_network + 
                       self.model_weights.content_similarity + 
                       self.model_weights.confidence_calibration)
        
        for doc_id in target_documents:
            score = (
                self.model_weights.citation_network * citation_scores.get(doc_id, 0.0) +
                self.model_weights.content_similarity * content_scores.get(doc_id, 0.0) +
                self.model_weights.confidence_calibration * confidence_scores.get(doc_id, 0.0)
            )
            
            combined_scores[doc_id] = score / total_weight if total_weight > 0 else 0.0
        
        return combined_scores
    
    def get_combined_features(self, target_documents: List[str]) -> pd.DataFrame:
        """
        Extract features from all models for target documents.
        
        Args:
            target_documents: List of OpenAlex IDs to extract features for
        
        Returns:
            DataFrame with combined features from all models
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting features")
        
        # Get features from citation model
        citation_features = self.citation_model.get_citation_features(target_documents)
        
        # For now, just return citation features
        # Future: add content features when needed
        combined_features = citation_features.copy()
        
        return combined_features
    
    def analyze_known_outlier(self, outlier_id: str) -> Dict[str, Any]:
        """
        Detailed analysis of a known outlier document.
        
        Args:
            outlier_id: OpenAlex ID of the known outlier
        
        Returns:
            Comprehensive analysis of why this document is an outlier
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analysis")
        
        print(f"Analyzing known outlier: {outlier_id}")
        
        # Get comprehensive features and scores
        features = self.get_combined_features([outlier_id])
        scores = self.predict_relevance_scores([outlier_id])
        individual_scores = self._get_individual_scores(outlier_id)
        doc_info = self._get_document_info(outlier_id)
        
        # Compare with other relevant documents
        relevant_docs = list(self.known_relevant_docs - {outlier_id})
        if relevant_docs:
            relevant_features = self.get_combined_features(relevant_docs)
            comparison = self._compare_with_relevant_docs(features, relevant_features)
        else:
            comparison = {}
        
        analysis = {
            'document_info': doc_info,
            'combined_score': scores.get(outlier_id, 0.0),
            'individual_scores': individual_scores,
            'features': features.iloc[0].to_dict() if not features.empty else {},
            'comparison_with_relevant': comparison,
            'outlier_characteristics': self._identify_outlier_characteristics(outlier_id, features)
        }
        
        return analysis
    
    def _get_individual_scores(self, doc_id: str) -> Dict[str, float]:
        """Get scores from individual models."""
        citation_score = self.citation_model.predict_relevance_scores([doc_id]).get(doc_id, 0.0)
        content_score = self.content_model.predict_relevance_scores([doc_id]).get(doc_id, 0.0)
        confidence_score = self.confidence_model.predict_relevance_scores([doc_id]).get(doc_id, 0.0)
        
        scores = {
            'citation_network': citation_score,
            'content_similarity': content_score,
            'confidence_calibration': confidence_score
        }
        
        return scores
    
    def _get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """Get document information from simulation data."""
        if self.simulation_data is not None:
            doc_row = self.simulation_data[self.simulation_data['openalex_id'] == doc_id]
            if not doc_row.empty:
                return doc_row.iloc[0].to_dict()
        
        return {'openalex_id': doc_id, 'info': 'Not found in simulation data'}
    
    def _analyze_outlier_reasons(self, doc_id: str, features_df: pd.DataFrame) -> List[str]:
        """Analyze why a document might be an outlier."""
        reasons = []
        
        doc_features = features_df[features_df['openalex_id'] == doc_id]
        if doc_features.empty:
            return ['No features available']
        
        features = doc_features.iloc[0]
        
        # Citation-based analysis
        if features['total_citations'] == 0:
            reasons.append("No citations in dataset (isolated document)")
        
        if features['relevant_connections'] == 0:
            reasons.append("No direct citations to relevant documents")
        
        if features['max_coupling_strength'] < 0.1:
            reasons.append("Weak bibliographic coupling with relevant documents")
        
        if features['neighborhood_enrichment_1hop'] == 0:
            reasons.append("Citation neighborhood contains no relevant documents")
        
        # Future analysis will check other model features
        
        if not reasons:
            reasons.append("Standard outlier patterns not detected")
        
        return reasons
    
    def _compare_with_relevant_docs(self, outlier_features: pd.DataFrame, 
                                   relevant_features: pd.DataFrame) -> Dict[str, Any]:
        """Compare outlier features with other relevant documents."""
        if outlier_features.empty or relevant_features.empty:
            return {}
        
        outlier = outlier_features.iloc[0]
        relevant_stats = relevant_features.describe()
        
        comparison = {}
        
        # Compare key citation features
        citation_features = ['total_citations', 'relevant_connections_ratio', 
                           'max_coupling_strength', 'neighborhood_enrichment_1hop']
        
        for feature in citation_features:
            if feature in outlier and feature in relevant_stats.columns:
                outlier_val = outlier[feature]
                relevant_mean = relevant_stats.loc['mean', feature]
                relevant_min = relevant_stats.loc['min', feature]
                
                comparison[feature] = {
                    'outlier_value': outlier_val,
                    'relevant_mean': relevant_mean,
                    'relevant_min': relevant_min,
                    'percentile': self._calculate_percentile(outlier_val, relevant_features[feature])
                }
        
        return comparison
    
    def _calculate_percentile(self, value: float, distribution: pd.Series) -> float:
        """Calculate what percentile the value falls into within the distribution."""
        return (distribution <= value).mean() * 100
    
    def _identify_outlier_characteristics(self, doc_id: str, features_df: pd.DataFrame) -> List[str]:
        """Identify specific characteristics that make this an outlier."""
        characteristics = []
        
        if features_df.empty:
            return ['No feature data available']
        
        features = features_df.iloc[0]
        
        # Citation isolation
        if features['total_citations'] == 0:
            characteristics.append("Citation isolated: No citations within dataset")
        elif features['relevant_connections'] == 0:
            characteristics.append("Relevance isolated: No connections to relevant documents")
        
        # Weak coupling
        if features['max_coupling_strength'] < 0.05:
            characteristics.append("Bibliographically distant: Very weak coupling with relevant docs")
        
        # Neighborhood analysis
        if features['neighborhood_enrichment_1hop'] == 0:
            characteristics.append("Poor citation neighborhood: No relevant documents in vicinity")
        
        # Future characteristics from other models will be added here
        
        return characteristics


def main():
    """Example usage of the hybrid outlier detection system."""
    # Load simulation data
    simulation_df = pd.read_csv('data/simulation.csv')
    
    # Initialize hybrid detector with optimal weights
    optimal_weights = ModelWeights(
        citation_network=0.4,
        content_similarity=0.6,
        confidence_calibration=0.2
    )
    detector = HybridOutlierDetector(model_weights=optimal_weights)
    
    # Fit the system
    detector.fit(simulation_df)
    
    # Analyze the known outlier (record_id=497)
    outlier_row = simulation_df[simulation_df['record_id'] == 497]
    if not outlier_row.empty:
        outlier_id = outlier_row.iloc[0]['openalex_id']
        
        print(f"\n=== Analysis of Known Outlier (record_id=497) ===")
        analysis = detector.analyze_known_outlier(outlier_id)
        
        print(f"Combined Score: {analysis['combined_score']:.4f}")
        print(f"Citation Score: {analysis['individual_scores']['citation_network']:.4f}")
        print(f"Content Similarity Score: {analysis['individual_scores']['content_similarity']:.4f}")
        print(f"Confidence Calibration Score: {analysis['individual_scores']['confidence_calibration']:.4f}")
        print(f"ASReview Ranking: {analysis['document_info'].get('asreview_ranking', 'N/A')}")
        
        print("\nOutlier Characteristics:")
        for char in analysis['outlier_characteristics']:
            print(f"  - {char}")
    
    # Test outlier detection on non-relevant documents
    non_relevant = simulation_df[simulation_df['label_included'] == 0]['openalex_id'].head(20).tolist()
    outliers = detector.predict_outliers(non_relevant, threshold=0.05)
    
    print(f"\n=== Outlier Detection Results ===")
    print(f"Candidates analyzed: {len(non_relevant)}")
    print(f"Potential outliers found: {len(outliers)}")


if __name__ == "__main__":
    main() 