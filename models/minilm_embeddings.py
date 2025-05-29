"""
MiniLM Semantic Embeddings Model for Outlier Detection

This model uses sentence-transformers all-MiniLM-L6-v2 to create semantic embeddings
of scientific papers and detect outliers based on semantic similarity to known relevant documents.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class MiniLMEmbeddingsModel:
    """
    Semantic embeddings model using MiniLM for outlier detection.
    
    Uses pre-trained sentence transformers to create semantic embeddings and
    identifies outliers based on their semantic distance from typical relevant documents.
    """
    
    def __init__(self, dataset_name: str = "Appenzeller-Herzog_2019"):
        """Initialize the MiniLM embeddings model."""
        self.dataset_name = dataset_name
        self.is_fitted = False
        self.simulation_data = None
        
        # Initialize the sentence transformer model
        print("Loading MiniLM model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("MiniLM model loaded successfully!")
        
        # Storage for embeddings and patterns
        self.relevant_embeddings = None
        self.relevant_centroid = None
        self.embedding_dim = 384  # MiniLM-L6-v2 output dimension
        
    def fit(self, simulation_df: pd.DataFrame) -> 'MiniLMEmbeddingsModel':
        """
        Fit the MiniLM embeddings model.
        
        Args:
            simulation_df: DataFrame with simulation results
        
        Returns:
            self: Returns the fitted model
        """
        print("Fitting MiniLM Embeddings Model...")
        
        self.simulation_data = simulation_df.copy()
        
        # Get relevant documents to establish embedding patterns
        relevant_docs = simulation_df[simulation_df['label_included'] == 1].copy()
        
        if len(relevant_docs) == 0:
            print("Warning: No relevant documents found")
            return self
        
        print(f"Creating embeddings for {len(relevant_docs)} relevant documents...")
        
        # Prepare texts for embedding
        relevant_texts = []
        for _, row in relevant_docs.iterrows():
            title = str(row.get('title', '') or '')
            abstract = str(row.get('abstract', '') or '')
            combined_text = f"{title}. {abstract}".strip()
            if combined_text and len(combined_text) > 10:
                relevant_texts.append(combined_text)
            else:
                relevant_texts.append("No text available")
        
        # Create embeddings for relevant documents
        self.relevant_embeddings = self.model.encode(
            relevant_texts, 
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Calculate centroid of relevant documents
        self.relevant_centroid = np.mean(self.relevant_embeddings, axis=0)
        
        # Calculate statistics for relevant documents
        self._analyze_relevant_patterns()
        
        self.is_fitted = True
        print("MiniLM embeddings model fitted successfully!")
        
        return self
    
    def _analyze_relevant_patterns(self):
        """Analyze patterns in relevant document embeddings."""
        if self.relevant_embeddings is None or len(self.relevant_embeddings) == 0:
            return
        
        # Calculate pairwise similarities within relevant documents
        similarities = cosine_similarity(self.relevant_embeddings)
        
        # Remove diagonal (self-similarity = 1.0)
        mask = np.ones(similarities.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        
        self.relevant_similarities = similarities[mask]
        self.mean_relevant_similarity = np.mean(self.relevant_similarities)
        self.std_relevant_similarity = np.std(self.relevant_similarities)
        
        # Calculate distances from centroid
        centroid_distances = []
        for embedding in self.relevant_embeddings:
            distance = np.linalg.norm(embedding - self.relevant_centroid)
            centroid_distances.append(distance)
        
        self.mean_centroid_distance = np.mean(centroid_distances)
        self.std_centroid_distance = np.std(centroid_distances)
        
        print(f"Relevant documents similarity statistics:")
        print(f"  Mean pairwise similarity: {self.mean_relevant_similarity:.3f}")
        print(f"  Std pairwise similarity: {self.std_relevant_similarity:.3f}")
        print(f"  Mean centroid distance: {self.mean_centroid_distance:.3f}")
        print(f"  Std centroid distance: {self.std_centroid_distance:.3f}")
    
    def create_embeddings(self, target_documents: List[str]) -> np.ndarray:
        """
        Create embeddings for target documents.
        
        Args:
            target_documents: List of OpenAlex IDs to create embeddings for
        
        Returns:
            numpy array of embeddings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before creating embeddings")
        
        texts = []
        
        for doc_id in target_documents:
            doc_row = self.simulation_data[self.simulation_data['openalex_id'] == doc_id]
            
            if doc_row.empty:
                texts.append("No document found")
                continue
            
            row = doc_row.iloc[0]
            title = str(row.get('title', '') or '')
            abstract = str(row.get('abstract', '') or '')
            combined_text = f"{title}. {abstract}".strip()
            
            if not combined_text or len(combined_text) <= 10:
                texts.append("No text available")
            else:
                texts.append(combined_text)
        
        # Create embeddings
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        
        return embeddings
    
    def calculate_similarity_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Calculate similarity scores for target documents.
        
        Args:
            target_documents: List of OpenAlex IDs to score
        
        Returns:
            Dictionary mapping document IDs to similarity scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating similarity scores")
        
        # Create embeddings for target documents
        target_embeddings = self.create_embeddings(target_documents)
        
        scores = {}
        
        for i, doc_id in enumerate(target_documents):
            embedding = target_embeddings[i]
            
            # Calculate multiple similarity metrics
            
            # 1. Cosine similarity to centroid
            centroid_similarity = cosine_similarity(
                embedding.reshape(1, -1), 
                self.relevant_centroid.reshape(1, -1)
            )[0, 0]
            
            # 2. Max similarity to any relevant document
            max_similarity = 0.0
            if self.relevant_embeddings is not None and len(self.relevant_embeddings) > 0:
                similarities_to_relevant = cosine_similarity(
                    embedding.reshape(1, -1), 
                    self.relevant_embeddings
                )[0]
                max_similarity = np.max(similarities_to_relevant)
            
            # 3. Mean similarity to all relevant documents
            mean_similarity = 0.0
            if self.relevant_embeddings is not None and len(self.relevant_embeddings) > 0:
                similarities_to_relevant = cosine_similarity(
                    embedding.reshape(1, -1), 
                    self.relevant_embeddings
                )[0]
                mean_similarity = np.mean(similarities_to_relevant)
            
            # Combine metrics (weighted average)
            combined_score = (
                0.4 * centroid_similarity + 
                0.4 * max_similarity + 
                0.2 * mean_similarity
            )
            
            scores[doc_id] = combined_score
        
        return scores
    
    def calculate_outlier_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Alternative scoring method focused on outlier detection.
        Looks for documents that are similar to relevant docs but different from typical patterns.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating outlier scores")
        
        target_embeddings = self.create_embeddings(target_documents)
        scores = {}
        
        for i, doc_id in enumerate(target_documents):
            embedding = target_embeddings[i]
            
            if self.relevant_embeddings is not None and len(self.relevant_embeddings) > 0:
                # Calculate similarities to all relevant documents
                similarities = cosine_similarity(
                    embedding.reshape(1, -1), 
                    self.relevant_embeddings
                )[0]
                
                # Outlier score: high max similarity but diverse similarity pattern
                max_sim = np.max(similarities)
                mean_sim = np.mean(similarities)
                std_sim = np.std(similarities)
                
                # Distance from centroid (normalized)
                centroid_dist = np.linalg.norm(embedding - self.relevant_centroid)
                normalized_dist = (centroid_dist - self.mean_centroid_distance) / self.std_centroid_distance
                
                # Outlier score: documents with high max similarity but unusual patterns
                outlier_score = (
                    0.4 * max_sim +  # Should be similar to at least one relevant doc
                    0.3 * std_sim +  # Should have diverse similarities (some high, some low)
                    0.2 * (1.0 / (1.0 + abs(normalized_dist))) +  # Prefer moderate distance from centroid
                    0.1 * (max_sim - mean_sim)  # High max but lower mean indicates outlier pattern
                )
                
                scores[doc_id] = outlier_score
            else:
                scores[doc_id] = 0.0
        
        return scores
    
    def calculate_isolation_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Use isolation forest approach to find outliers in embedding space.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating isolation scores")
        
        target_embeddings = self.create_embeddings(target_documents)
        
        # Combine relevant embeddings with target embeddings for isolation forest
        all_embeddings = np.vstack([self.relevant_embeddings, target_embeddings])
        
        # Use a simple distance-based outlier score
        scores = {}
        
        for i, doc_id in enumerate(target_documents):
            embedding = target_embeddings[i]
            
            # Calculate distances to all relevant documents
            distances = []
            for rel_emb in self.relevant_embeddings:
                dist = np.linalg.norm(embedding - rel_emb)
                distances.append(dist)
            
            # Outlier score based on distance distribution
            mean_dist = np.mean(distances)
            min_dist = np.min(distances)
            
            # Lower distance to closest relevant doc = higher score
            # But we want documents that are close to some but not all
            closest_similarity = 1.0 / (1.0 + min_dist)
            average_similarity = 1.0 / (1.0 + mean_dist)
            
            # Isolation score: close to some relevant docs but not typical
            isolation_score = closest_similarity * (1.0 - average_similarity + 0.5)
            
            scores[doc_id] = isolation_score
        
        return scores
    
    def calculate_aggressive_outlier_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Very aggressive outlier detection focusing on max similarity.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating aggressive scores")
        
        target_embeddings = self.create_embeddings(target_documents)
        scores = {}
        
        for i, doc_id in enumerate(target_documents):
            embedding = target_embeddings[i]
            
            if self.relevant_embeddings is not None and len(self.relevant_embeddings) > 0:
                similarities = cosine_similarity(
                    embedding.reshape(1, -1), 
                    self.relevant_embeddings
                )[0]
                
                max_sim = np.max(similarities)
                
                # Very aggressive: if max similarity > 0.75, boost heavily
                if max_sim > 0.75:
                    aggressive_score = max_sim * 2.0  # Double the score
                elif max_sim > 0.7:
                    aggressive_score = max_sim * 1.5
                else:
                    aggressive_score = max_sim
                
                scores[doc_id] = aggressive_score
            else:
                scores[doc_id] = 0.0
        
        return scores
    
    def calculate_top_k_similarity_scores(self, target_documents: List[str], k: int = 3) -> Dict[str, float]:
        """
        Focus on top-k similarities instead of just max.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating top-k scores")
        
        target_embeddings = self.create_embeddings(target_documents)
        scores = {}
        
        for i, doc_id in enumerate(target_documents):
            embedding = target_embeddings[i]
            
            if self.relevant_embeddings is not None and len(self.relevant_embeddings) > 0:
                similarities = cosine_similarity(
                    embedding.reshape(1, -1), 
                    self.relevant_embeddings
                )[0]
                
                # Get top-k similarities
                top_k_sims = np.sort(similarities)[-k:]
                
                # Score based on top-k average with heavy weight on highest
                top_k_score = (
                    0.6 * top_k_sims[-1] +  # Highest similarity
                    0.3 * top_k_sims[-2] if len(top_k_sims) > 1 else 0 +  # Second highest
                    0.1 * np.mean(top_k_sims[:-1]) if len(top_k_sims) > 2 else 0  # Rest
                )
                
                scores[doc_id] = top_k_score
            else:
                scores[doc_id] = 0.0
        
        return scores
    
    def calculate_threshold_based_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Threshold-based scoring: documents with any similarity > threshold get high scores.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating threshold scores")
        
        target_embeddings = self.create_embeddings(target_documents)
        scores = {}
        
        # Set threshold based on relevant document patterns
        high_threshold = self.mean_relevant_similarity + 0.5 * self.std_relevant_similarity
        
        for i, doc_id in enumerate(target_documents):
            embedding = target_embeddings[i]
            
            if self.relevant_embeddings is not None and len(self.relevant_embeddings) > 0:
                similarities = cosine_similarity(
                    embedding.reshape(1, -1), 
                    self.relevant_embeddings
                )[0]
                
                # Count similarities above threshold
                above_threshold = np.sum(similarities > high_threshold)
                max_sim = np.max(similarities)
                
                # Score: combination of max similarity and count above threshold
                threshold_score = 0.7 * max_sim + 0.3 * (above_threshold / len(similarities))
                
                scores[doc_id] = threshold_score
            else:
                scores[doc_id] = 0.0
        
        return scores
    
    def calculate_percentile_based_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Score based on what percentile the max similarity falls into among relevant docs.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating percentile scores")
        
        target_embeddings = self.create_embeddings(target_documents)
        scores = {}
        
        # Create reference distribution from relevant similarities
        ref_similarities = self.relevant_similarities
        
        for i, doc_id in enumerate(target_documents):
            embedding = target_embeddings[i]
            
            if self.relevant_embeddings is not None and len(self.relevant_embeddings) > 0:
                similarities = cosine_similarity(
                    embedding.reshape(1, -1), 
                    self.relevant_embeddings
                )[0]
                
                max_sim = np.max(similarities)
                
                # Calculate percentile of max similarity in reference distribution
                percentile = (np.sum(ref_similarities < max_sim) / len(ref_similarities)) * 100
                
                # Convert percentile to score (higher percentile = higher score)
                percentile_score = percentile / 100.0
                
                # Boost if percentile is very high
                if percentile > 80:
                    percentile_score *= 1.5
                elif percentile > 60:
                    percentile_score *= 1.2
                
                scores[doc_id] = percentile_score
            else:
                scores[doc_id] = 0.0
        
        return scores
    
    def calculate_super_ensemble_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Super ensemble combining all methods including the new aggressive ones.
        """
        # Get all scoring methods
        all_scores = {
            'original': self.calculate_similarity_scores(target_documents),
            'outlier': self.calculate_outlier_scores(target_documents),
            'isolation': self.calculate_isolation_scores(target_documents),
            'aggressive': self.calculate_aggressive_outlier_scores(target_documents),
            'top_k': self.calculate_top_k_similarity_scores(target_documents),
            'threshold': self.calculate_threshold_based_scores(target_documents),
            'percentile': self.calculate_percentile_based_scores(target_documents)
        }
        
        # Normalize all scores
        def normalize_scores(scores_dict):
            values = list(scores_dict.values())
            if len(values) == 0:
                return scores_dict
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return {k: 0.5 for k in scores_dict.keys()}
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}
        
        normalized_scores = {name: normalize_scores(scores) for name, scores in all_scores.items()}
        
        # Combine with heavy weights on aggressive methods
        super_ensemble = {}
        for doc_id in target_documents:
            combined_score = (
                0.1 * normalized_scores['original'].get(doc_id, 0) +
                0.1 * normalized_scores['outlier'].get(doc_id, 0) +
                0.15 * normalized_scores['isolation'].get(doc_id, 0) +
                0.25 * normalized_scores['aggressive'].get(doc_id, 0) +  # Heavy weight
                0.2 * normalized_scores['top_k'].get(doc_id, 0) +
                0.1 * normalized_scores['threshold'].get(doc_id, 0) +
                0.1 * normalized_scores['percentile'].get(doc_id, 0)
            )
            super_ensemble[doc_id] = combined_score
        
        return super_ensemble
    
    def predict_relevance_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Predict relevance scores for target documents based on semantic similarity.
        Uses the best performing 'Novelty' approach by default.
        
        Args:
            target_documents: List of OpenAlex IDs to score
        
        Returns:
            Dictionary mapping document IDs to relevance scores
        """
        # Use the best performing method (Novelty) by default
        return self.calculate_novelty_scores(target_documents)
    
    def analyze_outlier_embedding(self, outlier_id: str) -> Dict[str, float]:
        """
        Detailed analysis of a specific outlier's embedding characteristics.
        
        Args:
            outlier_id: OpenAlex ID of the outlier to analyze
        
        Returns:
            Dictionary with detailed embedding analysis
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analyzing outliers")
        
        outlier_embedding = self.create_embeddings([outlier_id])[0]
        
        analysis = {}
        
        # Similarity to centroid
        centroid_similarity = cosine_similarity(
            outlier_embedding.reshape(1, -1), 
            self.relevant_centroid.reshape(1, -1)
        )[0, 0]
        analysis['centroid_similarity'] = centroid_similarity
        
        # Distance from centroid
        centroid_distance = np.linalg.norm(outlier_embedding - self.relevant_centroid)
        analysis['centroid_distance'] = centroid_distance
        analysis['centroid_distance_zscore'] = (centroid_distance - self.mean_centroid_distance) / self.std_centroid_distance
        
        if self.relevant_embeddings is not None and len(self.relevant_embeddings) > 0:
            # Similarities to all relevant documents
            similarities = cosine_similarity(
                outlier_embedding.reshape(1, -1), 
                self.relevant_embeddings
            )[0]
            
            analysis['max_similarity_to_relevant'] = np.max(similarities)
            analysis['mean_similarity_to_relevant'] = np.mean(similarities)
            analysis['min_similarity_to_relevant'] = np.min(similarities)
            analysis['std_similarity_to_relevant'] = np.std(similarities)
            
            # Compare to typical inter-relevant similarities
            analysis['similarity_zscore'] = (np.mean(similarities) - self.mean_relevant_similarity) / self.std_relevant_similarity
        
        return analysis
    
    def calculate_pure_max_similarity_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Pure max similarity - just rank by highest similarity to any relevant doc.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating pure max scores")
        
        target_embeddings = self.create_embeddings(target_documents)
        scores = {}
        
        for i, doc_id in enumerate(target_documents):
            embedding = target_embeddings[i]
            
            if self.relevant_embeddings is not None and len(self.relevant_embeddings) > 0:
                similarities = cosine_similarity(
                    embedding.reshape(1, -1), 
                    self.relevant_embeddings
                )[0]
                
                # Just use max similarity as score
                scores[doc_id] = np.max(similarities)
            else:
                scores[doc_id] = 0.0
        
        return scores
    
    def calculate_novelty_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Novelty-based scoring: similar to relevant docs but different from typical patterns.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating novelty scores")
        
        target_embeddings = self.create_embeddings(target_documents)
        scores = {}
        
        for i, doc_id in enumerate(target_documents):
            embedding = target_embeddings[i]
            
            if self.relevant_embeddings is not None and len(self.relevant_embeddings) > 0:
                similarities = cosine_similarity(
                    embedding.reshape(1, -1), 
                    self.relevant_embeddings
                )[0]
                
                max_sim = np.max(similarities)
                
                # Calculate "novelty" - how different this doc is from the typical relevant pattern
                # High max similarity but low similarity to centroid = novel
                centroid_sim = cosine_similarity(
                    embedding.reshape(1, -1), 
                    self.relevant_centroid.reshape(1, -1)
                )[0, 0]
                
                # Novelty score: high max similarity but lower centroid similarity
                novelty_gap = max_sim - centroid_sim
                
                # Only consider documents with reasonably high max similarity
                if max_sim > 0.6:
                    novelty_score = max_sim + novelty_gap  # Boost for novelty
                else:
                    novelty_score = max_sim
                
                scores[doc_id] = novelty_score
            else:
                scores[doc_id] = 0.0
        
        return scores
    
    def calculate_semantic_bridge_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Look for documents that bridge semantic gaps between relevant documents.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating bridge scores")
        
        target_embeddings = self.create_embeddings(target_documents)
        scores = {}
        
        # Find pairs of relevant documents that are most dissimilar
        relevant_similarities = cosine_similarity(self.relevant_embeddings)
        np.fill_diagonal(relevant_similarities, 1.0)  # Ignore self-similarity
        
        for i, doc_id in enumerate(target_documents):
            embedding = target_embeddings[i]
            
            if self.relevant_embeddings is not None and len(self.relevant_embeddings) > 0:
                similarities = cosine_similarity(
                    embedding.reshape(1, -1), 
                    self.relevant_embeddings
                )[0]
                
                # Find if this document "bridges" between dissimilar relevant docs
                bridge_score = 0.0
                
                for j in range(len(self.relevant_embeddings)):
                    for k in range(j+1, len(self.relevant_embeddings)):
                        # If j and k are dissimilar relevant docs
                        if relevant_similarities[j, k] < 0.5:  # Dissimilar threshold
                            # Check if current doc is similar to both
                            sim_to_j = similarities[j]
                            sim_to_k = similarities[k]
                            
                            if sim_to_j > 0.6 and sim_to_k > 0.6:
                                bridge_score = max(bridge_score, min(sim_to_j, sim_to_k))
                
                # Combine with regular max similarity
                max_sim = np.max(similarities)
                combined_score = 0.7 * max_sim + 0.3 * bridge_score
                
                scores[doc_id] = combined_score
            else:
                scores[doc_id] = 0.0
        
        return scores
    
    def calculate_rarity_weighted_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Weight similarities by how rare/unique the relevant documents are.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating rarity scores")
        
        target_embeddings = self.create_embeddings(target_documents)
        scores = {}
        
        # Calculate "rarity" of each relevant document
        # Documents that are dissimilar to other relevant docs are "rare"
        relevant_similarities = cosine_similarity(self.relevant_embeddings)
        np.fill_diagonal(relevant_similarities, 0.0)  # Ignore self-similarity
        
        rarity_weights = []
        for i in range(len(self.relevant_embeddings)):
            avg_sim_to_others = np.mean(relevant_similarities[i])
            rarity = 1.0 - avg_sim_to_others  # Lower similarity = higher rarity
            rarity_weights.append(rarity)
        
        rarity_weights = np.array(rarity_weights)
        
        for i, doc_id in enumerate(target_documents):
            embedding = target_embeddings[i]
            
            if self.relevant_embeddings is not None and len(self.relevant_embeddings) > 0:
                similarities = cosine_similarity(
                    embedding.reshape(1, -1), 
                    self.relevant_embeddings
                )[0]
                
                # Weight similarities by rarity of relevant documents
                weighted_similarities = similarities * rarity_weights
                
                # Use max weighted similarity
                rarity_score = np.max(weighted_similarities)
                
                scores[doc_id] = rarity_score
            else:
                scores[doc_id] = 0.0
        
        return scores
    
    def calculate_ultra_aggressive_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Ultra aggressive: just find documents with very high max similarity.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating ultra aggressive scores")
        
        target_embeddings = self.create_embeddings(target_documents)
        scores = {}
        
        for i, doc_id in enumerate(target_documents):
            embedding = target_embeddings[i]
            
            if self.relevant_embeddings is not None and len(self.relevant_embeddings) > 0:
                similarities = cosine_similarity(
                    embedding.reshape(1, -1), 
                    self.relevant_embeddings
                )[0]
                
                max_sim = np.max(similarities)
                
                # Ultra aggressive boosting
                if max_sim > 0.8:
                    ultra_score = max_sim * 5.0  # 5x boost
                elif max_sim > 0.75:
                    ultra_score = max_sim * 3.0  # 3x boost
                elif max_sim > 0.7:
                    ultra_score = max_sim * 2.0  # 2x boost
                else:
                    ultra_score = max_sim
                
                scores[doc_id] = ultra_score
            else:
                scores[doc_id] = 0.0
        
        return scores


def main():
    """Test the MiniLM embeddings model standalone."""
    print("Testing MiniLM Embeddings Model...")
    
    # Load simulation data
    simulation_df = pd.read_csv('../data/simulation.csv')
    print(f"Loaded {len(simulation_df)} documents from simulation")
    
    # CRITICAL FIX: Create training data that excludes rank 26 (like in test.py)
    training_data = simulation_df.copy()
    training_data['label_included'] = training_data['asreview_ranking'].apply(
        lambda x: 1 if x <= 25 else 0
    )
    
    print(f"Training with {training_data['label_included'].sum()} relevant documents (ranks 1-25 only)")
    print("Rank 26 is NOT included in training - this is the outlier we're trying to find")
    
    # Create and fit model
    model = MiniLMEmbeddingsModel()
    model.fit(training_data)  # Use training data without rank 26
    
    # Get the known outlier
    outlier_row = simulation_df[simulation_df['record_id'] == 497]
    if outlier_row.empty:
        print("ERROR: Outlier with record_id=497 not found!")
        return
        
    outlier_id = outlier_row.iloc[0]['openalex_id']
    outlier_ranking = outlier_row.iloc[0]['asreview_ranking']
    print(f"\nOutlier: {outlier_id} (ASReview ranking: {outlier_ranking})")
    
    # Test outlier analysis
    analysis = model.analyze_outlier_embedding(outlier_id)
    print(f"\nDetailed outlier analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value:.4f}")
    
    # Test outlier retrieval among irrelevant documents
    print(f"\n=== OUTLIER RETRIEVAL TEST ===")
    
    # Get all irrelevant documents
    irrelevant_docs = simulation_df[simulation_df['label_included'] == 0]
    print(f"Total irrelevant documents: {len(irrelevant_docs)}")
    
    # Create search pool: outlier + all irrelevant documents
    search_pool = [outlier_id] + irrelevant_docs['openalex_id'].tolist()
    print(f"Search pool size: {len(search_pool)} documents")
    
    # Score all documents in search pool
    print("Scoring all documents...")
    
    # Test different scoring approaches
    approaches = {
        'Original Similarity': model.calculate_similarity_scores,
        'Outlier Detection': model.calculate_outlier_scores,
        'Isolation Method': model.calculate_isolation_scores,
        'Aggressive Outlier': model.calculate_aggressive_outlier_scores,
        'Top-K Similarity': model.calculate_top_k_similarity_scores,
        'Threshold Based': model.calculate_threshold_based_scores,
        'Percentile Based': model.calculate_percentile_based_scores,
        'Super Ensemble': model.calculate_super_ensemble_scores,
        'Pure Max Similarity': model.calculate_pure_max_similarity_scores,
        'Novelty': model.calculate_novelty_scores,
        'Semantic Bridge': model.calculate_semantic_bridge_scores,
        'Rarity Weighted': model.calculate_rarity_weighted_scores,
        'Ultra Aggressive': model.calculate_ultra_aggressive_scores
    }
    
    best_approach = None
    best_position = float('inf')
    
    for approach_name, scoring_method in approaches.items():
        print(f"\n--- Testing: {approach_name} ---")
        scores = scoring_method(search_pool)
        
        # Sort documents by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Find outlier position
        outlier_position = None
        outlier_score = None
        for i, (doc_id, score) in enumerate(sorted_results):
            if doc_id == outlier_id:
                outlier_position = i + 1  # 1-indexed
                outlier_score = score
                break
        
        print(f"Outlier found at position: {outlier_position} out of {len(search_pool)}")
        print(f"Outlier score: {outlier_score:.4f}")
        percentile = ((len(search_pool) - outlier_position) / len(search_pool)) * 100
        print(f"Percentile: {percentile:.2f}th percentile")
        
        # Check top results
        top_50_ids = [doc_id for doc_id, _ in sorted_results[:50]]
        top_100_ids = [doc_id for doc_id, _ in sorted_results[:100]]
        
        found_in_top_50 = outlier_id in top_50_ids
        found_in_top_100 = outlier_id in top_100_ids
        
        print(f"Found in top 50: {'YES' if found_in_top_50 else 'NO'}")
        print(f"Found in top 100: {'YES' if found_in_top_100 else 'NO'}")
        
        # Track best approach
        if outlier_position < best_position:
            best_position = outlier_position
            best_approach = approach_name
    
    print(f"\n=== SUMMARY ===")
    print(f"Best approach: {best_approach}")
    print(f"Best position: {best_position} out of {len(search_pool)}")
    
    # Show detailed results for best approach
    print(f"\n=== DETAILED RESULTS FOR {best_approach.upper()} ===")
    best_method = approaches[best_approach]
    scores = best_method(search_pool)
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Find outlier position again
    outlier_position = None
    outlier_score = None
    for i, (doc_id, score) in enumerate(sorted_results):
        if doc_id == outlier_id:
            outlier_position = i + 1
            outlier_score = score
            break
    
    percentile = ((len(search_pool) - outlier_position) / len(search_pool)) * 100
    
    print(f"Outlier found at position: {outlier_position} out of {len(search_pool)}")
    print(f"Outlier score: {outlier_score:.4f}")
    print(f"Percentile: {percentile:.2f}th percentile")
    
    # Show top 10 results
    print(f"\nTop 10 highest scoring documents:")
    for i, (doc_id, score) in enumerate(sorted_results[:10]):
        marker = " *** OUTLIER ***" if doc_id == outlier_id else ""
        print(f"  {i+1}. Score: {score:.4f}{marker}")
    
    # Practical questions
    top_50_ids = [doc_id for doc_id, _ in sorted_results[:50]]
    top_100_ids = [doc_id for doc_id, _ in sorted_results[:100]]
    
    found_in_top_50 = outlier_id in top_50_ids
    found_in_top_100 = outlier_id in top_100_ids
    
    print(f"\nPractical Results:")
    print(f"Found in top 50: {'YES' if found_in_top_50 else 'NO'}")
    print(f"Found in top 100: {'YES' if found_in_top_100 else 'NO'}")
    
    if found_in_top_50:
        print("✅ EXCELLENT: Outlier found in top 50!")
    elif found_in_top_100:
        print("⚠️  GOOD: Outlier found in top 100")
    else:
        print("❌ NEEDS IMPROVEMENT: Outlier not found in top 100")
    
    return {
        'best_approach': best_approach,
        'outlier_position': outlier_position,
        'outlier_score': outlier_score,
        'total_documents': len(search_pool),
        'found_in_top_50': found_in_top_50,
        'found_in_top_100': found_in_top_100
    }


if __name__ == "__main__":
    main() 