"""
Citation network scoring utilities.

This module contains functions for scoring documents based on citation network features,
used for outlier detection in scientific document collections.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Set


def calculate_isolation_deviation(features: pd.Series, baseline: Dict[str, float], 
                                 G: nx.Graph, relevant_documents: Set[str]) -> float:
    """Calculate how isolated this document is from the relevant paper ecosystem."""
    doc_id = features['openalex_id']
    
    # Calculate shortest path distance to relevant documents
    dist = float('inf')
    for rel_doc in relevant_documents:
        if rel_doc in G.nodes:
            try:
                d = nx.shortest_path_length(G, doc_id, rel_doc)
                dist = min(dist, d)
            except nx.NetworkXNoPath:
                continue
    
    # Adaptive distance-based scoring with better generalization
    if dist == float('inf'):
        dist_score = 1.0  # Completely disconnected
    else:
        # Get distance statistics from baseline
        mean_dist = baseline.get('mean_distance', 3.0)
        std_dist = baseline.get('std_distance', 1.5)
        
        # Calculate z-score and map to isolation score
        z_score = (dist - mean_dist) / max(std_dist, 0.01)
        
        # Map z-score to isolation score (higher z = more isolated)
        if z_score <= -1:     dist_score = 0.1
        elif z_score <= 0:    dist_score = 0.1 + (z_score + 1) * 0.2
        elif z_score <= 1:    dist_score = 0.3 + z_score * 0.35
        elif z_score <= 2:    dist_score = 0.65 + (z_score - 1) * 0.2
        else:                 dist_score = min(0.95, 0.85 + (z_score - 2) * 0.1)
    
    # Citation penalty - adjust based on dataset
    cit_count = features['total_citations']
    mean_cit = baseline.get('mean_total_citations', 5.0)
    
    # Calculate citation penalty factor
    if cit_count == 0:                      cit_penalty = 1.0  # Max penalty
    elif cit_count < 0.2 * mean_cit:        cit_penalty = 0.9  # Very few citations  
    elif cit_count < 0.5 * mean_cit:        cit_penalty = 0.7  # Below average
    elif cit_count < mean_cit:              cit_penalty = 0.5  # Near average
    else:                                   cit_penalty = 0.3  # Above average
    
    # Apply citation penalty
    dist_score = min(1.0, dist_score * cit_penalty + (1 - cit_penalty) * 0.5)
    
    # Direct connection score
    rel_ratio = features['relevant_connections_ratio']
    mean_rel_ratio = baseline.get('mean_relevant_connections_ratio', 0.3)
    
    if rel_ratio >= mean_rel_ratio:
        direct_iso = 0.1  # Well connected
    elif rel_ratio > 0:
        direct_iso = 0.5 * (1 - rel_ratio / max(mean_rel_ratio, 0.01))  # Partial isolation
    else:
        direct_iso = min(0.9, 0.6 + 0.3 * (1 - min(1, cit_count / max(1, mean_cit))))
    
    # Neighborhood enrichment scores
    neigh_enrichment = features['neighborhood_enrichment_1hop']
    mean_neigh = baseline.get('mean_neighborhood_enrichment_1hop', 0.2)
    
    # Calculate neighborhood isolation score
    if neigh_enrichment <= 0.01:               neigh_iso = 0.9  # No relevant neighbors
    elif neigh_enrichment < 0.25 * mean_neigh: neigh_iso = 0.7  # Far below average
    elif neigh_enrichment < 0.5 * mean_neigh:  neigh_iso = 0.5  # Below average
    elif neigh_enrichment < mean_neigh:        neigh_iso = 0.3  # Near average
    else:                                      neigh_iso = 0.1  # Above average
    
    # Second-hop enrichment score
    hop2_enrichment = features['neighborhood_enrichment_2hop']
    
    if hop2_enrichment <= 0.01:   hop2_iso = 0.8
    elif hop2_enrichment < 0.05:  hop2_iso = 0.6
    elif hop2_enrichment < 0.1:   hop2_iso = 0.4
    elif hop2_enrichment < 0.2:   hop2_iso = 0.2
    else:                         hop2_iso = 0.1
    
    # Dataset-adaptive weights
    if baseline.get('mean_total_citations', 5.0) < 3:
        weights = [0.5, 0.3, 0.15, 0.05]  # Sparse network
    elif baseline.get('mean_total_citations', 5.0) > 10:
        weights = [0.4, 0.25, 0.2, 0.15]  # Dense network
    else:
        weights = [0.45, 0.25, 0.2, 0.1]  # Medium density
    
    # Combine scores with weights
    iso_score = (
        weights[0] * dist_score + 
        weights[1] * direct_iso + 
        weights[2] * neigh_iso + 
        weights[3] * hop2_iso
    )
    
    return min(1.0, max(0.0, iso_score))


def calculate_coupling_deviation(features: pd.Series, baseline: Dict[str, float]) -> float:
    """Calculate how different coupling patterns are from baseline."""
    max_coupling = features['max_coupling_strength']
    above_threshold = features['coupling_above_threshold']
    mean_max_coupling = baseline.get('mean_max_coupling_strength', 0.2)
    
    # Is this a sparse coupling network?
    is_sparse_coupling = mean_max_coupling < 0.1
    
    # No coupling with any relevant document
    if max_coupling == 0:
        return 0.9
    
    # Calculate relative difference for better generalization
    if mean_max_coupling > 0:
        rel_diff = (mean_max_coupling - max_coupling) / mean_max_coupling
    else:
        # If mean is 0, any non-zero coupling is better
        rel_diff = -1 if max_coupling > 0 else 0
    
    # Adjust thresholds for sparse coupling networks
    if is_sparse_coupling:
        # For datasets with very low coupling values, use more sensitive thresholds
        if rel_diff <= -0.5:      coupling_score = 0.05  # Much better than average
        elif rel_diff <= 0:       coupling_score = 0.1   # Better than average
        elif rel_diff < 0.4:      coupling_score = 0.3   # Slightly below average
        elif rel_diff < 0.7:      coupling_score = 0.5   # Moderately below average
        elif rel_diff < 0.9:      coupling_score = 0.8   # Significantly below average
        else:                     coupling_score = 0.9   # Very weak coupling
    else:
        # Standard thresholds
        if rel_diff <= 0:       coupling_score = 0.1  # Better than average
        elif rel_diff < 0.3:    coupling_score = 0.3  # Slightly below average
        elif rel_diff < 0.6:    coupling_score = 0.5  # Moderately below average
        elif rel_diff < 0.9:    coupling_score = 0.7  # Significantly below average
        else:                   coupling_score = 0.9  # Very weak coupling
    
    # Adjust for documents with connections above threshold
    if above_threshold > 0:
        coupling_score = max(0.1, coupling_score * 0.7)
    
    return coupling_score


def calculate_neighborhood_deviation(features: pd.Series, baseline: Dict[str, float]) -> float:
    """Calculate how different neighborhood patterns are from baseline."""
    neigh_enrichment_1hop = features['neighborhood_enrichment_1hop']
    neigh_enrichment_2hop = features['neighborhood_enrichment_2hop']
    neigh_size = features['neighborhood_size_1hop']
    mean_1hop = baseline.get('mean_neighborhood_enrichment_1hop', 0.2)
    
    # Calculate relative difference
    rel_diff_1hop = (mean_1hop - neigh_enrichment_1hop) / mean_1hop if mean_1hop > 0 else (1.0 if neigh_enrichment_1hop == 0 else 0.0)
    
    # Score for 1-hop neighborhood
    if neigh_enrichment_1hop == 0:      score_1hop = 0.9  # No relevant neighbors
    elif rel_diff_1hop <= 0:            score_1hop = 0.1  # Better than average
    elif rel_diff_1hop < 0.3:           score_1hop = 0.3  # Slightly below average
    elif rel_diff_1hop < 0.6:           score_1hop = 0.5  # Moderately below average
    elif rel_diff_1hop < 0.9:           score_1hop = 0.7  # Significantly below average
    else:                               score_1hop = 0.9  # Very poor enrichment
    
    # Score for 2-hop neighborhood
    if neigh_enrichment_2hop == 0:      score_2hop = 0.9
    elif neigh_enrichment_2hop < 0.05:  score_2hop = 0.7
    elif neigh_enrichment_2hop < 0.1:   score_2hop = 0.5
    elif neigh_enrichment_2hop < 0.2:   score_2hop = 0.3
    else:                               score_2hop = 0.1
    
    # Weight based on neighborhood size
    if neigh_size <= 2:      weight_1hop = 0.9  # Very small neighborhood
    elif neigh_size <= 5:    weight_1hop = 0.8  # Small neighborhood
    elif neigh_size <= 10:   weight_1hop = 0.7  # Medium neighborhood
    else:                    weight_1hop = 0.6  # Large neighborhood
    
    # Combine scores
    return weight_1hop * score_1hop + (1.0 - weight_1hop) * score_2hop


def calculate_advanced_score(features: pd.Series, relevant_ratio: float) -> float:
    """Calculate advanced score from advanced network features."""
    # Extract advanced features
    citation_diversity = features.get('citation_diversity', 0.0)
    relevant_betweenness = features.get('relevant_betweenness', 0.0)
    structural_anomaly = features.get('structural_anomaly', 0.0)
    semantic_isolation = features.get('semantic_isolation', 0.0)
    
    # Calculate diversity score - higher diversity is more likely to be an outlier
    diversity_score = min(1.0, citation_diversity * 1.2)
    
    # Calculate betweenness score - outliers often have low betweenness
    # or paradoxically very high betweenness if they're bridges between clusters
    if relevant_betweenness < 0.01:
        betweenness_score = 0.8  # Very low betweenness suggests isolation
    elif relevant_betweenness > 0.3:
        betweenness_score = 0.7  # High betweenness can indicate a bridge node
    else:
        betweenness_score = 0.3  # Normal betweenness
        
    # Calculate structural anomaly score
    anomaly_score = min(1.0, structural_anomaly * 1.5)
    
    # Calculate semantic isolation score (new for sparse datasets)
    isolation_score = semantic_isolation if semantic_isolation > 0 else 0.0
    
    # Combine scores based on dataset characteristics
    if relevant_ratio < 0.02:
        # For extremely sparse datasets like Appenzeller, emphasize semantic isolation
        weights = [0.25, 0.15, 0.30, 0.30]  # [diversity, betweenness, anomaly, isolation]
    elif relevant_ratio < 0.05:
        weights = [0.35, 0.20, 0.45, 0.0]  # [diversity, betweenness, anomaly, isolation]
    else:
        weights = [0.3, 0.4, 0.3, 0.0]
        
    advanced_score = (
        weights[0] * diversity_score +
        weights[1] * betweenness_score +
        weights[2] * anomaly_score +
        weights[3] * isolation_score
    )
    
    return advanced_score


def get_adaptive_weights(sparsity_factor: float) -> Dict[str, float]:
    """Get adaptive weights for combining scores based on dataset sparsity."""
    # Adaptive weights based on dataset characteristics
    # Use continuous scaling based on sparsity rather than discrete categories
    weights = {
        'isolation': 0.35 * (1 + sparsity_factor * 0.3),  # More weight to isolation in sparse datasets
        'coupling': 0.25 * (1 + sparsity_factor * 0.2),
        'neighborhood': 0.25 * (1 - sparsity_factor * 0.2),
        'advanced': 0.15 * (1 + sparsity_factor * 0.4)  # More weight to advanced features in sparse datasets
    }
    
    # Normalize weights to sum to 1.0
    weight_sum = sum(weights.values())
    weights = {k: v / weight_sum for k, v in weights.items()}
    
    return weights 