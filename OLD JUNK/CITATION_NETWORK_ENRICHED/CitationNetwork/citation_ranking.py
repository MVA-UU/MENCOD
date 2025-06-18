"""
Citation network ranking utilities.

This module contains functions for specialized ranking adjustments in citation networks,
particularly for extremely sparse datasets where standard scoring may not be sufficient.
"""

import numpy as np
from typing import Dict, Set, List, Any


def apply_sparse_dataset_ranking_adjustments(
    scores: Dict[str, float], 
    feature_cache: Dict[str, Dict[str, Any]], 
    relevant_documents: Set[str],
    sparsity_factor: float,
    baseline_stats: Dict[str, float]
) -> Dict[str, float]:
    """
    Apply specialized ranking adjustments for extremely sparse datasets.
    
    For sparse datasets like Appenzeller, we use a rank-based approach with adaptive
    boosting for documents with outlier-like characteristics.
    
    Args:
        scores: Original document scores
        feature_cache: Cached document features
        relevant_documents: Set of known relevant document IDs
        sparsity_factor: Dataset sparsity factor (0-1)
        baseline_stats: Baseline statistics from relevant documents
        
    Returns:
        Adjusted document scores
    """
    # First, identify documents with strong outlier characteristics for special boosting
    # Adaptively determine what constitutes an outlier based on dataset statistics
    special_boost_docs = []
    
    # Get relevant document statistics to adapt thresholds
    rel_conns_stats = []
    div_stats = []
    anomaly_stats = []
    
    # Collect feature statistics for relevant documents
    for doc_id in relevant_documents:
        if doc_id in feature_cache:
            features = feature_cache[doc_id]
            rel_conns_stats.append(features['relevant_connections'])
            div_stats.append(features['citation_diversity'])
            anomaly_stats.append(features['structural_anomaly'])
    
    # Calculate adaptive thresholds based on relevant document statistics
    if rel_conns_stats and div_stats and anomaly_stats:
        # Use percentiles rather than fixed thresholds
        rel_conns_thresh = np.percentile(rel_conns_stats, 60)  # Above 60th percentile
        div_thresh = np.percentile(div_stats, 60)
        anomaly_thresh = np.percentile(anomaly_stats, 60)
        
        print(f"Adaptive thresholds based on dataset statistics: rel_conns={rel_conns_thresh:.1f}, "
              f"div={div_thresh:.2f}, anomaly={anomaly_thresh:.2f}")
    else:
        # Fallback to reasonable defaults if no statistics available
        rel_conns_thresh = 2.0
        div_thresh = 0.3
        anomaly_thresh = 0.5
    
    # Score all documents using a continuous approach rather than binary thresholds
    boost_scores = {}
    
    # Calculate continuous boost score for each document
    for doc_id in scores:
        if doc_id not in feature_cache:
            continue
            
        features = feature_cache[doc_id]
        base_score = scores[doc_id]
        
        # Skip documents with very low scores
        if base_score < 0.3:
            continue
        
        # Calculate normalized distance from median for each feature
        rel_conns = features['relevant_connections']
        diversity = features['citation_diversity'] 
        anomaly = features['structural_anomaly']
        
        # Boosting factors based on how far each feature is from the threshold
        rel_conns_factor = min(1.0, rel_conns / max(1, rel_conns_thresh))
        div_factor = min(1.0, diversity / max(0.1, div_thresh))
        anomaly_factor = min(1.0, anomaly / max(0.1, anomaly_thresh))
        
        # Weighted boost score - all features contribute proportionally
        boost_score = (rel_conns_factor * 0.4 + div_factor * 0.3 + anomaly_factor * 0.3)
        
        # Apply boost only to documents with reasonable base scores
        if boost_score > 0.5 and base_score > 0.3:
            special_boost_docs.append(doc_id)
            boost_scores[doc_id] = boost_score
    
    print(f"Identified {len(special_boost_docs)} documents for adaptive boosting")
    
    # Convert scores to rank-based percentiles
    score_items = list(scores.items())
    
    # Apply special boosting - amount of boost depends on the boost score
    for doc_id in special_boost_docs:
        # Find the item in score_items
        for i, (id, score) in enumerate(score_items):
            if id == doc_id:
                # Apply data-driven boost based on boost score
                boost_amount = boost_scores[doc_id] * 0.4  # Scale boost by feature score
                boosted_score = min(0.95, score + boost_amount)
                score_items[i] = (id, boosted_score)
                break
    
    sorted_scores = sorted(score_items, key=lambda x: x[1], reverse=True)
    
    # Apply rank-based score normalization 
    max_rank = len(sorted_scores)
    
    # Create a new dictionary with rank-normalized scores
    ranked_scores = {}
    for rank, (doc_id, _) in enumerate(sorted_scores):
        # Linear rank scaling with boosting for top results
        if rank == 0:
            # Top document gets highest possible score
            ranked_scores[doc_id] = 0.999
        elif rank < 10:
            # Top 10 documents get scores from 0.99 to 0.90
            ranked_scores[doc_id] = 0.999 - (rank * 0.01)
        elif rank < 100:
            # Next 90 documents get scores from 0.89 to 0.80
            ranked_scores[doc_id] = 0.90 - ((rank - 10) * 0.001)
        else:
            # Remaining documents get scores from 0.79 to 0
            remaining_ranks = max_rank - 100
            if remaining_ranks > 0:
                ranked_scores[doc_id] = max(0.0, 0.80 - ((rank - 100) / remaining_ranks) * 0.80)
            else:
                ranked_scores[doc_id] = 0.0
    
    print("Applied rank-based score normalization for extremely sparse dataset")
    return ranked_scores 