"""
MiniLM Semantic Embeddings Model for Outlier Detection

This model uses sentence-transformers all-MiniLM-L6-v2 to create semantic embeddings
of scientific papers and detect outliers based on semantic similarity to known relevant documents.
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Fix import for direct script execution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import dataset utilities
try:
    from models.dataset_utils import (
        load_datasets_config,
        prompt_dataset_selection,
        load_dataset,
        identify_outlier_in_simulation,
        create_training_data,
        get_search_pool
    )
except ModuleNotFoundError:
    from dataset_utils import (
        load_datasets_config,
        prompt_dataset_selection,
        load_dataset,
        identify_outlier_in_simulation,
        create_training_data,
        get_search_pool
    )


class MiniLMEmbeddingsModel:
    """
    Semantic embeddings model using MiniLM for outlier detection.
    
    Uses pre-trained sentence transformers to create semantic embeddings and
    identifies outliers based on their semantic distance from typical relevant documents.
    """
    
    def __init__(self, dataset_name: Optional[str] = None):
        """
        Initialize the MiniLM embeddings model.
        
        Args:
            dataset_name: Optional name of dataset to use. If None, will prompt user.
        """
        # If dataset_name is not provided, prompt user to select one
        if dataset_name is None:
            self.dataset_name = prompt_dataset_selection()
        else:
            self.dataset_name = dataset_name
            
        print(f"Using dataset: {self.dataset_name}")
        
        # Load dataset configuration
        self.datasets_config = load_datasets_config()
        if self.dataset_name not in self.datasets_config:
            raise ValueError(f"Dataset '{self.dataset_name}' not found in configuration")
        
        self.dataset_config = self.datasets_config[self.dataset_name]
        
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
        
    def fit(self, simulation_df: Optional[pd.DataFrame] = None) -> 'MiniLMEmbeddingsModel':
        """
        Fit the MiniLM embeddings model.
        
        Args:
            simulation_df: Optional DataFrame with simulation results.
                           If None, will load from dataset configuration.
        
        Returns:
            self: Returns the fitted model
        """
        print("Fitting MiniLM Embeddings Model...")
        
        # Load simulation data if not provided
        if simulation_df is None:
            simulation_df, _ = load_dataset(self.dataset_name)
        
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
    
    def calculate_novelty_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Novelty-based scoring: similar to relevant docs but different from typical patterns.
        Uses dynamically calculated thresholds based on dataset statistics.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating novelty scores")
        
        target_embeddings = self.create_embeddings(target_documents)
        scores = {}
        
        # Calculate dynamic threshold based on relevant document patterns
        min_sim_threshold = self.mean_relevant_similarity - 0.5 * self.std_relevant_similarity
        
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
                if max_sim > min_sim_threshold:
                    novelty_score = max_sim + novelty_gap  # Boost for novelty
                else:
                    novelty_score = max_sim
                
                scores[doc_id] = novelty_score
            else:
                scores[doc_id] = 0.0
        
        return scores
    
    def predict_relevance_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """
        Predict relevance scores for target documents based on semantic similarity.
        Uses the novelty approach.
        
        Args:
            target_documents: List of OpenAlex IDs to score
        
        Returns:
            Dictionary mapping document IDs to relevance scores
        """
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


def main():
    """Test the MiniLM embeddings model with a selected dataset."""
    print("Testing MiniLM Embeddings Model...")
    
    # Create model (will prompt user to select dataset)
    model = MiniLMEmbeddingsModel()
    
    # Load dataset
    simulation_df, dataset_config = load_dataset(model.dataset_name)
    print(f"Loaded {len(simulation_df)} documents from simulation")
    
    # Find the outlier record
    outlier_row = identify_outlier_in_simulation(simulation_df, dataset_config)
    outlier_id = outlier_row['openalex_id']
    print(f"\nOutlier: {outlier_id} (Record ID: {outlier_row['record_id']})")
    
    # Create training data that excludes the outlier
    training_data = create_training_data(simulation_df, outlier_id)
    
    # Count relevant documents for reporting
    num_relevant = training_data['label_included'].sum()
    print(f"Training with {num_relevant} relevant documents (excluding outlier)")
    print("Outlier is NOT included in training - this is what we're trying to find")
    
    # Fit model
    model.fit(training_data)
    
    # Test outlier analysis
    analysis = model.analyze_outlier_embedding(outlier_id)
    print(f"\nDetailed outlier analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value:.4f}")
    
    # Test outlier retrieval among irrelevant documents
    print(f"\n=== OUTLIER RETRIEVAL TEST ===")
    
    # Get search pool: outlier + all irrelevant documents
    search_pool = get_search_pool(simulation_df, outlier_id)
    print(f"Search pool size: {len(search_pool)} documents")
    
    # Score all documents in search pool
    print("Scoring all documents using novelty approach...")
    
    scores = model.calculate_novelty_scores(search_pool)
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
        'outlier_position': outlier_position,
        'outlier_score': outlier_score,
        'total_documents': len(search_pool),
        'found_in_top_50': found_in_top_50,
        'found_in_top_100': found_in_top_100
    }


if __name__ == "__main__":
    main() 