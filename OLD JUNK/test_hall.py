"""
Test script for the Hall dataset using the updated Hybrid Outlier Detection System.

This script specifically tests the Hall dataset which was previously challenging
for the hybrid model to reach >95th percentile performance.
"""

import pandas as pd
import numpy as np
from models.hybrid_models import HybridOutlierDetector, ModelWeights
from models.CitationNetwork.citation_network import CitationNetworkModel
from models.dataset_utils import (
    load_dataset,
    identify_outlier_in_simulation,
    create_training_data,
    get_search_pool
)


def test_hall_dataset():
    """
    Test the hybrid model specifically on the Hall dataset,
    which requires special handling due to its characteristics.
    """
    print("=== TESTING HALL DATASET WITH RANK-BASED FUSION ===")
    
    # Create the hybrid model with adaptive weighting and specify Hall dataset
    detector = HybridOutlierDetector(dataset_name="hall", use_adaptive_weights=True)
    
    # Load dataset
    simulation_df, dataset_config = load_dataset(detector.dataset_name)
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
    
    # Fit model
    detector.fit(training_data)
    
    # Get search pool
    search_pool = get_search_pool(simulation_df, outlier_id)
    print(f"Search pool size: {len(search_pool)} documents")
    
    # Score all documents in search pool using rank-based fusion
    scores = detector.predict_relevance_scores(search_pool)
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
    print(f"Percentile: {percentile:.1f}th")
    
    # Top 10 scores
    print(f"\nTop 10 scores:")
    for i, (doc_id, score) in enumerate(sorted_results[:10], 1):
        marker = " *** OUTLIER ***" if doc_id == outlier_id else ""
        print(f"  {i:2d}. Score: {score:.4f}{marker}")
    
    # Assessment of results
    if percentile >= 95:
        print("\n✅ EXCELLENT: Outlier found above 95th percentile!")
    elif percentile >= 90:
        print("\n⚠️ GOOD: Outlier found above 90th percentile")
    else:
        print("\n❌ NEEDS IMPROVEMENT: Outlier below 90th percentile")


if __name__ == "__main__":
    test_hall_dataset() 