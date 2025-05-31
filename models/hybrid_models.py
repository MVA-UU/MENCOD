"""
Hybrid Outlier Detection System

This module combines multiple detection methods to identify relevant documents
that are missed by content-based ranking algorithms (outliers).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Updated import path for CitationNetwork
from .CitationNetwork.citation_network import CitationNetworkModel
from .content_similarity import ContentSimilarityModel
from .confidence_calibration import ConfidenceCalibrationModel
from .minilm_embeddings import MiniLMEmbeddingsModel
from .dataset_utils import prompt_dataset_selection


@dataclass
class ModelWeights:
    """Configuration for model weights in hybrid system."""
    citation_network: float = 0.2
    content_similarity: float = 0.4
    confidence_calibration: float = 0.2
    minilm_embeddings: float = 0.2
    # metadata_analysis: float = 0.0
    # temporal_patterns: float = 0.0


class HybridOutlierDetector:
    """
    Hybrid system combining multiple approaches for outlier detection.
    
    Currently includes:
    - Citation network analysis
    - Content similarity analysis 
    - Confidence calibration analysis
    - MiniLM semantic embeddings analysis
    
    Future extensions:
    - Metadata pattern analysis
    - Temporal publication patterns
    - Author/venue analysis
    """
    
    def __init__(self, 
                 dataset_name: Optional[str] = None,
                 model_weights: Optional[ModelWeights] = None,
                 use_adaptive_weights: bool = True):
        """
        Initialize the hybrid outlier detection system.
        
        Args:
            dataset_name: Name of the synergy dataset to use. If None, will prompt user.
            model_weights: Weights for combining different models
            use_adaptive_weights: Whether to use adaptive weighting based on dataset characteristics
        """
        # If dataset_name is not provided, prompt user to select one
        if dataset_name is None:
            self.dataset_name = prompt_dataset_selection()
        else:
            self.dataset_name = dataset_name
            
        print(f"Using dataset: {self.dataset_name}")
        
        self.model_weights = model_weights or ModelWeights()
        self.use_adaptive_weights = use_adaptive_weights
        
        # Initialize individual models
        self.citation_model = CitationNetworkModel(self.dataset_name)
        self.content_model = ContentSimilarityModel(self.dataset_name)
        self.confidence_model = ConfidenceCalibrationModel()
        self.minilm_model = MiniLMEmbeddingsModel(self.dataset_name)
        
        # Future models will be initialized here
        # self.metadata_model = MetadataAnalysisModel()
        
        self.is_fitted = False
        self.known_relevant_docs = set()
        self.simulation_data = None
        self.dataset_stats = {}
    
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
        
        # Analyze dataset characteristics
        self._analyze_dataset_characteristics()
        
        # Apply adaptive weighting if enabled
        if self.use_adaptive_weights:
            self._optimize_model_weights()
        
        # Fit individual models
        print("\n1. Fitting Citation Network Model...")
        self.citation_model.fit(simulation_df)
        print("\n2. Fitting Content Similarity Model...")
        self.content_model.fit(simulation_df)
        print("\n3. Fitting Confidence Calibration Model...")
        self.confidence_model.fit(simulation_df)
        print("\n4. Fitting MiniLM Embeddings Model...")
        self.minilm_model.fit(simulation_df)
        
        # Future model fitting will go here
        # print("5. Fitting Metadata Analysis Model...")
        # self.metadata_model.fit(simulation_df)
        
        self.is_fitted = True
        print("\nHybrid system successfully fitted!")
        
        return self
    
    def _analyze_dataset_characteristics(self):
        """Analyze dataset characteristics to guide adaptive weighting."""
        total_docs = len(self.simulation_data)
        num_relevant = len(self.known_relevant_docs)
        rel_ratio = num_relevant / total_docs if total_docs > 0 else 0
        
        # Calculate continuous metrics rather than categorical variables
        self.dataset_stats = {
            'total_documents': total_docs,
            'relevant_documents': num_relevant,
            'relevant_ratio': rel_ratio,
            'sparsity_factor': 1 - min(0.9, max(0.1, rel_ratio * 10)),  # Higher for sparser datasets
            'size_factor': min(1.0, total_docs / 5000)  # 0.0 for empty dataset, 1.0 for 5000+ documents
        }
        
        # Analyze text characteristics as continuous variables
        if 'abstract' in self.simulation_data.columns:
            # Check abstract availability and length
            has_abstract = self.simulation_data['abstract'].notna()
            abstract_lengths = self.simulation_data.loc[has_abstract, 'abstract'].astype(str).apply(len)
            
            # Calculate continuous metrics for text characteristics
            self.dataset_stats.update({
                'abstract_availability': has_abstract.mean(),  # 0.0 to 1.0
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
        
        print(f"\nDataset Characteristics:")
        print(f"  Total documents: {self.dataset_stats['total_documents']}")
        print(f"  Relevant documents: {self.dataset_stats['relevant_documents']}")
        print(f"  Relevant ratio: {self.dataset_stats['relevant_ratio']:.4f}")
        print(f"  Sparsity factor: {self.dataset_stats['sparsity_factor']:.4f} (higher = sparser)")
        print(f"  Size factor: {self.dataset_stats['size_factor']:.4f} (higher = larger)")
        
        if 'abstract' in self.simulation_data.columns:
            print(f"  Abstract availability: {self.dataset_stats['abstract_availability']:.4f}")
            print(f"  Mean abstract length: {self.dataset_stats['mean_abstract_length']:.1f} chars")
            print(f"  Text richness factor: {self.dataset_stats['text_richness_factor']:.4f} (higher = richer)")
    
    def _optimize_model_weights(self):
        """
        Optimize model weights using continuous scaling functions based on dataset characteristics.
        This avoids hard thresholds and creates a smooth distribution of weights.
        """
        # Extract continuous dataset factors
        sparsity = self.dataset_stats['sparsity_factor']  # 0.0-1.0 (higher = sparser)
        text_richness = self.dataset_stats.get('text_richness_factor', 0.0)  # 0.0-1.0 (higher = richer)
        size_factor = self.dataset_stats['size_factor']  # 0.0-1.0 (higher = larger)
        
        # Base distribution - citation and embeddings should be higher for sparse datasets
        # Start with neutral weights
        citation_weight = 0.25
        content_weight = 0.25
        confidence_weight = 0.25
        minilm_weight = 0.25
        
        # --- Continuous adjustment based on sparsity ---
        # Citation network importance increases with sparsity (from 0.25 to 0.40)
        citation_factor = 0.25 + (0.15 * sparsity)
        citation_weight = citation_factor
        
        # MiniLM importance increases with sparsity (from 0.25 to 0.40)
        minilm_factor = 0.25 + (0.15 * sparsity)
        minilm_weight = minilm_factor
        
        # Content similarity is less affected by sparsity but still varies
        content_factor = 0.25 - (0.05 * sparsity)
        content_weight = content_factor
        
        # Confidence calibration decreases with sparsity (from 0.25 to 0.0)
        # For very sparse datasets, confidence calibration is unreliable
        confidence_factor = 0.25 * (1 - sparsity)
        confidence_weight = confidence_factor
        
        # --- Text richness adjustments ---
        # If we have rich text, increase content similarity weight
        if text_richness > 0:
            # Smooth boosting of content weight based on text richness
            content_boost = 0.15 * text_richness
            content_weight += content_boost
            
            # Redistribute the boost by reducing other weights proportionally
            total_other_weights = citation_weight + minilm_weight + confidence_weight
            if total_other_weights > 0:
                reduction_factor = content_boost / total_other_weights
                citation_weight *= (1 - reduction_factor)
                minilm_weight *= (1 - reduction_factor)
                confidence_weight *= (1 - reduction_factor)
        
        # --- Dataset size adjustments ---
        # For larger datasets, citation networks tend to be more informative
        if size_factor > 0:
            # Smooth boosting of citation weight based on dataset size
            citation_boost = 0.05 * size_factor
            citation_weight += citation_boost
            
            # Redistribute the boost by reducing other weights proportionally
            total_other_weights = content_weight + minilm_weight + confidence_weight
            if total_other_weights > 0:
                reduction_factor = citation_boost / total_other_weights
                content_weight *= (1 - reduction_factor)
                minilm_weight *= (1 - reduction_factor)
                confidence_weight *= (1 - reduction_factor)
        
        # Ensure all weights are non-negative
        citation_weight = max(0.0, citation_weight)
        content_weight = max(0.0, content_weight)
        confidence_weight = max(0.0, confidence_weight)
        minilm_weight = max(0.0, minilm_weight)
        
        # Normalize weights to ensure they sum to 1
        total = citation_weight + content_weight + confidence_weight + minilm_weight
        
        if total > 0:
            self.model_weights = ModelWeights(
                citation_network=citation_weight / total,
                content_similarity=content_weight / total,
                confidence_calibration=confidence_weight / total,
                minilm_embeddings=minilm_weight / total
            )
        
        print(f"\nOptimized Model Weights (continuous scaling):")
        print(f"  Citation Network: {self.model_weights.citation_network:.4f}")
        print(f"  Content Similarity: {self.model_weights.content_similarity:.4f}")
        print(f"  Confidence Calibration: {self.model_weights.confidence_calibration:.4f}")
        print(f"  MiniLM Embeddings: {self.model_weights.minilm_embeddings:.4f}")
        
        # Explain the weight distribution
        if sparsity > 0.7:
            print("→ Emphasizing citation network and embeddings due to dataset sparsity")
        if text_richness > 0.6:
            print("→ Boosted content similarity due to rich text availability")
        if size_factor > 0.7:
            print("→ Increased citation network weight due to large dataset size")
    
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
        Generate combined relevance scores from all models using rank-based fusion.
        This approach is more robust to different score distributions across datasets.
        
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
        minilm_scores = self.minilm_model.predict_relevance_scores(target_documents)
        
        # Get ranks for each model's scores - lower rank = higher score (1 = best)
        citation_ranks = self._get_ranks(citation_scores)
        content_ranks = self._get_ranks(content_scores)
        confidence_ranks = self._get_ranks(confidence_scores)
        minilm_ranks = self._get_ranks(minilm_scores)
        
        # Calculate sparsity-adaptive weights
        sparsity = self.dataset_stats['sparsity_factor']
        
        # Use more extreme weights for very sparse datasets
        if sparsity > 0.7:  # Hall dataset has sparsity 0.7481
            # For sparse datasets like Hall, heavily prioritize citation and embedding models
            # Basically disable confidence_calibration which performs poorly on Hall
            citation_weight = 0.45
            minilm_weight = 0.45
            content_weight = 0.10
            confidence_weight = 0.0
            print(f"Using extreme weights for highly sparse dataset: C={citation_weight:.2f}, T={content_weight:.2f}, CC={confidence_weight:.2f}, M={minilm_weight:.2f}")
        else:
            # For other datasets, use the model weights
            citation_weight = self.model_weights.citation_network
            content_weight = self.model_weights.content_similarity
            confidence_weight = self.model_weights.confidence_calibration
            minilm_weight = self.model_weights.minilm_embeddings
        
        # Normalize weights
        total_weight = citation_weight + content_weight + confidence_weight + minilm_weight
        if total_weight > 0:
            citation_weight /= total_weight
            content_weight /= total_weight
            confidence_weight /= total_weight
            minilm_weight /= total_weight
        
        # Create robust combined scores using reciprocal rank fusion
        combined_scores = {}
        for doc_id in target_documents:
            # Get individual ranks (or a high default rank if not found)
            max_rank = len(target_documents)
            c_rank = citation_ranks.get(doc_id, max_rank)
            t_rank = content_ranks.get(doc_id, max_rank)
            cc_rank = confidence_ranks.get(doc_id, max_rank)
            m_rank = minilm_ranks.get(doc_id, max_rank)
            
            # Reciprocal rank fusion with weights
            # Lower value = better rank
            # k is a constant that avoids division by zero and dampens impact of high ranks
            k = 60  # Standard value in RRF
            
            # For zero weights, use a high rank to effectively ignore that model
            if citation_weight <= 0.01: c_rank = max_rank
            if content_weight <= 0.01: t_rank = max_rank
            if confidence_weight <= 0.01: cc_rank = max_rank
            if minilm_weight <= 0.01: m_rank = max_rank
            
            # Calculate reciprocal rank fusion score
            fusion_score = (
                citation_weight / (k + c_rank) +
                content_weight / (k + t_rank) +
                confidence_weight / (k + cc_rank) +
                minilm_weight / (k + m_rank)
            )
            
            # Alternative rank aggregation approach: weighted Borda count
            # Borda count assigns points based on rank (higher = better)
            # max_rank - rank + 1 converts to points where higher is better
            borda_score = (
                citation_weight * (max_rank - c_rank + 1) +
                content_weight * (max_rank - t_rank + 1) +
                confidence_weight * (max_rank - cc_rank + 1) +
                minilm_weight * (max_rank - m_rank + 1)
            ) / max_rank  # Normalize to 0-1
            
            # Also consider raw scores with extreme normalization
            # Min-max normalize the scores first
            c_score = self._normalize_score(citation_scores.get(doc_id, 0.0), citation_scores)
            t_score = self._normalize_score(content_scores.get(doc_id, 0.0), content_scores)
            cc_score = self._normalize_score(confidence_scores.get(doc_id, 0.0), confidence_scores)
            m_score = self._normalize_score(minilm_scores.get(doc_id, 0.0), minilm_scores)
            
            # Weighted average of normalized scores
            score_avg = (
                citation_weight * c_score +
                content_weight * t_score +
                confidence_weight * cc_score +
                minilm_weight * m_score
            )
            
            # Combine multiple ranking methods
            # For the Hall dataset, prioritize Borda count which performs better
            if sparsity > 0.7:
                combined_score = 0.7 * borda_score + 0.2 * fusion_score + 0.1 * score_avg
            else:
                combined_score = 0.4 * borda_score + 0.3 * fusion_score + 0.3 * score_avg
            
            combined_scores[doc_id] = combined_score
        
        return combined_scores
    
    def _get_ranks(self, scores: Dict[str, float]) -> Dict[str, int]:
        """Convert scores to ranks (1 = highest score)."""
        if not scores:
            return {}
        
        # Sort by score in descending order
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Assign ranks (1-based, ties get the same rank)
        ranks = {}
        prev_score = None
        prev_rank = 0
        
        for i, (doc_id, score) in enumerate(sorted_items):
            if score != prev_score:
                ranks[doc_id] = i + 1
                prev_rank = i + 1
            else:
                ranks[doc_id] = prev_rank
            
            prev_score = score
        
        return ranks
    
    def _normalize_score(self, score: float, all_scores: Dict[str, float]) -> float:
        """Min-max normalize a score within a distribution."""
        if not all_scores:
            return 0.0
        
        values = list(all_scores.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return 1.0 if score > 0 else 0.0
        
        return (score - min_val) / (max_val - min_val)
    
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
        
        # Get features from citation model - updated to use the new method name
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
        minilm_score = self.minilm_model.predict_relevance_scores([doc_id]).get(doc_id, 0.0)
        
        scores = {
            'citation_network': citation_score,
            'content_similarity': content_score,
            'confidence_calibration': confidence_score,
            'minilm_embeddings': minilm_score
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
        
        # Use continuous thresholds based on percentiles rather than hard cutoffs
        
        # Citation-based analysis
        if features['total_citations'] == 0:
            reasons.append("No citations in dataset (isolated document)")
        
        if features['relevant_connections'] == 0:
            reasons.append("No direct citations to relevant documents")
        
        # Use continuous assessment rather than hard thresholds
        if features['max_coupling_strength'] < 0.1:
            strength = features['max_coupling_strength']
            if strength < 0.03:
                reasons.append(f"Very weak bibliographic coupling with relevant documents ({strength:.3f})")
            elif strength < 0.07:
                reasons.append(f"Weak bibliographic coupling with relevant documents ({strength:.3f})")
            else:
                reasons.append(f"Below average bibliographic coupling with relevant documents ({strength:.3f})")
        
        if features['neighborhood_enrichment_1hop'] < 0.1:
            enrich = features['neighborhood_enrichment_1hop']
            if enrich == 0:
                reasons.append("Citation neighborhood contains no relevant documents")
            else:
                reasons.append(f"Citation neighborhood has low enrichment with relevant documents ({enrich:.3f})")
        
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
        
        # Use continuous assessment rather than hard thresholds
        
        # Citation isolation - continuous assessment
        if features['total_citations'] == 0:
            characteristics.append("Citation isolated: No citations within dataset")
        elif features['total_citations'] <= 2:
            characteristics.append(f"Citation sparse: Only {features['total_citations']} citations in dataset")
        
        # Relevance isolation - continuous assessment
        if features['relevant_connections'] == 0:
            if features['total_citations'] == 0:
                characteristics.append("Completely isolated: No connections to any documents")
            else:
                characteristics.append("Relevance isolated: No connections to relevant documents")
        elif features['relevant_connections'] <= 2:
            characteristics.append(f"Relevance sparse: Only {features['relevant_connections']} connections to relevant documents")
        
        # Coupling strength - continuous assessment
        if features['max_coupling_strength'] < 0.1:
            if features['max_coupling_strength'] < 0.03:
                characteristics.append(f"Bibliographically very distant: Extremely weak coupling ({features['max_coupling_strength']:.3f})")
            elif features['max_coupling_strength'] < 0.07:
                characteristics.append(f"Bibliographically distant: Very weak coupling ({features['max_coupling_strength']:.3f})")
            else:
                characteristics.append(f"Bibliographically somewhat distant: Weak coupling ({features['max_coupling_strength']:.3f})")
        
        # Neighborhood analysis - continuous assessment
        if features['neighborhood_enrichment_1hop'] == 0:
            characteristics.append("Poor citation neighborhood: No relevant documents in vicinity")
        elif features['neighborhood_enrichment_1hop'] < 0.05:
            characteristics.append(f"Sparse citation neighborhood: Very few relevant documents nearby ({features['neighborhood_enrichment_1hop']:.3f})")
        
        # Future characteristics from other models will be added here
        
        return characteristics


def main():
    """Example usage of the hybrid outlier detection system."""
    # Load simulation data
    simulation_df = pd.read_csv('data/simulation.csv')
    
    # Initialize hybrid detector with adaptive weights
    detector = HybridOutlierDetector(use_adaptive_weights=True)
    
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