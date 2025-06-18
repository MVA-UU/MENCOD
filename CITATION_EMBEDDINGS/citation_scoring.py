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
    """Calculate relevance score based on connection strength to relevant documents.
    
    Optimized version that avoids expensive shortest path computations.
    """
    doc_id = features['openalex_id']
    
    # 1. Direct connectivity analysis (much faster than shortest paths)
    direct_connections = 0
    semantic_connections = 0
    citation_connections = 0
    connection_weights = []
    
    if doc_id in G.nodes:
        # Check direct connections to relevant documents
        neighbors = set(G.neighbors(doc_id))
        relevant_neighbors = neighbors.intersection(relevant_documents)
        
        for rel_doc in relevant_neighbors:
            if G.has_edge(doc_id, rel_doc):
                edge_data = G.get_edge_data(doc_id, rel_doc)
                if edge_data:
                    edge_type = edge_data.get('edge_type', 'unknown')
                    weight = edge_data.get('weight', 1.0)
                    connection_weights.append(weight)
                    
                    if edge_type == 'semantic':
                        semantic_connections += 1
                    elif edge_type == 'citation':
                        citation_connections += 1
                    
                    direct_connections += 1
        
        # 2-hop connectivity (more efficient than general shortest path)
        twohop_connections = 0
        if direct_connections == 0:  # Only check if no direct connections
            for neighbor in neighbors:
                if neighbor in G.nodes:
                    neighbor_neighbors = set(G.neighbors(neighbor))
                    twohop_relevant = neighbor_neighbors.intersection(relevant_documents)
                    twohop_connections += len(twohop_relevant)
    
    # Score based on connectivity patterns
    if direct_connections >= 3:
        connectivity_score = 1.0
    elif direct_connections >= 1:
        avg_weight = np.mean(connection_weights) if connection_weights else 1.0
        connectivity_score = 0.7 + 0.2 * min(1.0, avg_weight / 2.0)
    elif twohop_connections >= 2:
        connectivity_score = 0.5
    elif twohop_connections >= 1:
        connectivity_score = 0.3
    else:
        connectivity_score = 0.1
    
    # 2. Enhanced citation analysis with relative metrics
    cit_count = features.get('citation_in_degree', 0)
    
    # Calculate citation percentile within dataset
    baseline_cit_std = baseline.get('std_citation_in_degree', 1.0)
    baseline_cit_mean = baseline.get('mean_citation_in_degree', 1.0)
    
    # Z-score based citation assessment
    if baseline_cit_std > 0:
        cit_zscore = (cit_count - baseline_cit_mean) / baseline_cit_std
        cit_score = max(0.1, min(1.0, 0.5 + cit_zscore * 0.15))
    else:
        cit_score = 0.5 if cit_count > 0 else 0.1
    
    # 3. Local network centrality (efficient degree-based measure)
    if doc_id in G.nodes:
        degree = G.degree(doc_id)
        weighted_degree = G.degree(doc_id, weight='weight') if G.is_multigraph() else degree
        
        # Normalize by baseline
        baseline_degree_mean = baseline.get('mean_total_degree', 1.0)
        baseline_degree_std = baseline.get('std_total_degree', 1.0)
        
        if baseline_degree_std > 0:
            degree_zscore = (degree - baseline_degree_mean) / baseline_degree_std
            centrality_score = max(0.1, min(1.0, 0.5 + degree_zscore * 0.1))
        else:
            centrality_score = 0.5 if degree > 0 else 0.1
    else:
        centrality_score = 0.0
    
    # 4. Semantic connectivity score (use precomputed features)
    semantic_score = 0.2  # Default
    if semantic_connections >= 3:
        semantic_score = 1.0
    elif semantic_connections >= 1:
        avg_weight = np.mean([w for w, t in zip(connection_weights, 
                            [G.get_edge_data(doc_id, rel_doc, {}).get('edge_type', '') 
                             for rel_doc in relevant_documents if G.has_edge(doc_id, rel_doc)])
                            if t == 'semantic']) if connection_weights else 1.0
        semantic_score = 0.6 + 0.3 * min(1.0, avg_weight / 2.0)
    
    # 5. Coupling-based relevance (use precomputed features)
    coupling_strength = features.get('max_coupling_strength', 0)
    rel_coupling = features.get('relevant_connections', 0)
    
    if coupling_strength >= 0.15:
        coupling_score = 1.0
    elif coupling_strength >= 0.08:
        coupling_score = 0.8
    elif coupling_strength >= 0.03:
        coupling_score = 0.6
    elif rel_coupling > 0:
        coupling_score = 0.4
    else:
        coupling_score = 0.1
    
    # 6. Adaptive weighting based on connection patterns
    if semantic_connections > citation_connections:
        # Semantic-dominant connections
        weights = [0.35, 0.15, 0.20, 0.25, 0.05]
    elif citation_connections > 0:
        # Citation-based connections
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    else:
        # Sparse connections - rely more on coupling and centrality
        weights = [0.25, 0.20, 0.25, 0.20, 0.10]
    
    # [connectivity, citation, centrality, semantic, coupling]
    relevance_score = (
        weights[0] * connectivity_score + 
        weights[1] * cit_score + 
        weights[2] * centrality_score +
        weights[3] * semantic_score +
        weights[4] * coupling_score
    )
    
    return min(1.0, max(0.0, relevance_score))


def calculate_coupling_deviation(features: pd.Series, baseline: Dict[str, float]) -> float:
    """Calculate coupling-based relevance score - REWARD strong coupling."""
    coupling_score = features.get('coupling_score', 0)
    cocitation_score = features.get('cocitation_score', 0)
    mean_coupling = baseline.get('mean_coupling_score', 0.2)
    mean_cocitation = baseline.get('mean_cocitation_score', 0.2)
    
    # Coupling relevance - REWARD strong coupling
    if coupling_score == 0:
        coup_relevance = 0.2  # No coupling = low relevance
    elif coupling_score >= 2 * mean_coupling:
        coup_relevance = 1.0  # Very strong coupling = excellent
    elif coupling_score >= mean_coupling:
        coup_relevance = 0.8  # Above average coupling = good
    elif coupling_score >= 0.5 * mean_coupling:
        coup_relevance = 0.6  # Moderate coupling = fair
    else:
        coup_relevance = 0.4  # Weak coupling = poor
    
    # Co-citation relevance - REWARD being co-cited with relevant papers
    if cocitation_score >= 3:
        cocit_relevance = 1.0  # Strong co-citation = excellent
    elif cocitation_score >= 1:
        cocit_relevance = 0.8  # Some co-citation = good
    else:
        cocit_relevance = 0.4  # No co-citation = moderate
    
    # Combine coupling measures
    return 0.7 * coup_relevance + 0.3 * cocit_relevance


def calculate_neighborhood_deviation(features: pd.Series, baseline: Dict[str, float]) -> float:
    """Calculate neighborhood-based relevance score - REWARD relevant neighborhoods."""
    neigh_enrichment_1hop = features.get('neighborhood_enrichment_1hop', 0)
    neigh_enrichment_2hop = features.get('neighborhood_enrichment_2hop', 0)
    neigh_size = features.get('neighborhood_size_1hop', 0)
    mean_1hop = baseline.get('mean_neighborhood_enrichment_1hop', 0.2)
    
    # 1-hop neighborhood relevance
    if neigh_enrichment_1hop >= mean_1hop:
        score_1hop = 1.0  # Above average = excellent
    elif neigh_enrichment_1hop >= 0.5 * mean_1hop:
        score_1hop = 0.8  # Decent = good
    elif neigh_enrichment_1hop > 0:
        score_1hop = 0.6  # Some = moderate
    else:
        score_1hop = 0.3  # None = poor
    
    # 2-hop neighborhood relevance
    if neigh_enrichment_2hop >= 0.1:
        score_2hop = 1.0
    elif neigh_enrichment_2hop >= 0.05:
        score_2hop = 0.8
    elif neigh_enrichment_2hop > 0:
        score_2hop = 0.6
    else:
        score_2hop = 0.4
    
    # Weight based on neighborhood size - larger neighborhoods are more informative
    if neigh_size >= 10:
        weight_1hop = 0.8  # Large neighborhood
    elif neigh_size >= 5:
        weight_1hop = 0.7  # Medium neighborhood
    elif neigh_size >= 2:
        weight_1hop = 0.6  # Small neighborhood
    else:
        weight_1hop = 0.5  # Very small neighborhood
    
    # Combine scores
    return weight_1hop * score_1hop + (1.0 - weight_1hop) * score_2hop


def calculate_temporal_score(features: pd.Series, baseline: Dict[str, float]) -> float:
    """Calculate temporal-based relevance score - REWARD temporal patterns indicating relevance."""
    # Extract temporal features
    citation_velocity = features.get('citation_velocity', 0.0)
    age_normalized_impact = features.get('age_normalized_impact', 0.0)
    citation_burst_score = features.get('citation_burst_score', 0.0)
    temporal_isolation = features.get('temporal_isolation', 0.0)
    recent_citation_ratio = features.get('recent_citation_ratio', 0.0)
    citation_acceleration = features.get('citation_acceleration', 0.0)
    
    # Citation velocity scoring - REWARD papers with high citation velocity
    mean_velocity = baseline.get('mean_citation_velocity', 0.5)
    if citation_velocity >= 2 * mean_velocity:
        velocity_score = 1.0  # Very high velocity = excellent
    elif citation_velocity >= mean_velocity:
        velocity_score = 0.8  # Above average velocity = good
    elif citation_velocity >= 0.5 * mean_velocity:
        velocity_score = 0.6  # Moderate velocity = fair
    elif citation_velocity > 0:
        velocity_score = 0.4  # Some velocity = poor
    else:
        velocity_score = 0.2  # No velocity = very poor
    
    # Age-normalized impact scoring
    mean_impact = baseline.get('mean_age_normalized_impact', 1.0)
    if age_normalized_impact >= 2 * mean_impact:
        impact_score = 1.0
    elif age_normalized_impact >= mean_impact:
        impact_score = 0.8
    elif age_normalized_impact >= 0.5 * mean_impact:
        impact_score = 0.6
    elif age_normalized_impact > 0:
        impact_score = 0.4
    else:
        impact_score = 0.2
    
    # Citation burst scoring - REWARD burst patterns (indicates sudden relevance)
    if citation_burst_score >= 0.8:
        burst_score = 1.0  # Strong burst = excellent
    elif citation_burst_score >= 0.5:
        burst_score = 0.8  # Moderate burst = good
    elif citation_burst_score >= 0.2:
        burst_score = 0.6  # Weak burst = fair
    else:
        burst_score = 0.4  # No burst = moderate
    
    # Temporal isolation scoring - REWARD low isolation (connected to active papers)
    if temporal_isolation <= 0.2:
        isolation_score = 1.0  # Well connected = excellent
    elif temporal_isolation <= 0.5:
        isolation_score = 0.8  # Moderately connected = good
    elif temporal_isolation <= 0.8:
        isolation_score = 0.6  # Somewhat isolated = fair
    else:
        isolation_score = 0.3  # Highly isolated = poor
    
    # Recent citation ratio scoring - REWARD citing recent/active papers
    if recent_citation_ratio >= 0.7:
        recent_score = 1.0  # Cites many recent papers = excellent
    elif recent_citation_ratio >= 0.5:
        recent_score = 0.8  # Cites some recent papers = good
    elif recent_citation_ratio >= 0.3:
        recent_score = 0.6  # Cites few recent papers = fair
    else:
        recent_score = 0.4  # Cites mostly old papers = moderate
    
    # Citation acceleration scoring - REWARD fast citation accumulation
    if citation_acceleration >= 1.5:
        accel_score = 1.0  # Very fast = excellent
    elif citation_acceleration >= 1.0:
        accel_score = 0.8  # Fast = good
    elif citation_acceleration >= 0.5:
        accel_score = 0.6  # Moderate = fair
    else:
        accel_score = 0.4  # Slow = moderate
    
    # Combine temporal scores with adaptive weights
    # For sparse datasets, emphasize burst and isolation patterns
    temporal_score = (
        0.25 * velocity_score +
        0.20 * impact_score +
        0.20 * burst_score +
        0.15 * isolation_score +
        0.10 * recent_score +
        0.10 * accel_score
    )
    
    return min(1.0, max(0.0, temporal_score))


def calculate_efficiency_score(features: pd.Series, baseline: Dict[str, float]) -> float:
    """Calculate efficiency-based relevance score - REWARD network efficiency patterns."""
    # Extract efficiency features
    local_clustering = features.get('local_clustering', 0.0)
    edge_type_diversity = features.get('edge_type_diversity', 0.0)
    relevant_path_efficiency = features.get('relevant_path_efficiency', 0.0)
    citation_authority = features.get('citation_authority', 0.0)
    semantic_coherence = features.get('semantic_coherence', 0.0)
    network_position_score = features.get('network_position_score', 0.0)
    
    # 1. Local clustering scoring - moderate clustering is optimal
    if local_clustering >= 0.3 and local_clustering <= 0.7:
        clustering_score = 1.0  # Optimal clustering = excellent
    elif local_clustering >= 0.1 and local_clustering <= 0.9:
        clustering_score = 0.8  # Good clustering = good
    elif local_clustering > 0:
        clustering_score = 0.6  # Some clustering = moderate
    else:
        clustering_score = 0.3  # No clustering = poor
    
    # 2. Edge type diversity scoring - diverse connections are valuable
    if edge_type_diversity >= 1.5:
        diversity_score = 1.0  # High diversity = excellent
    elif edge_type_diversity >= 1.0:
        diversity_score = 0.8  # Good diversity = good
    elif edge_type_diversity >= 0.5:
        diversity_score = 0.6  # Some diversity = moderate
    else:
        diversity_score = 0.4  # Low diversity = poor
    
    # 3. Relevant path efficiency scoring - MOST IMPORTANT for outlier detection
    if relevant_path_efficiency >= 0.8:
        path_score = 1.0  # Very efficient paths to relevant docs = excellent
    elif relevant_path_efficiency >= 0.5:
        path_score = 0.9  # Good paths = very good
    elif relevant_path_efficiency >= 0.2:
        path_score = 0.7  # Some paths = good
    elif relevant_path_efficiency > 0:
        path_score = 0.5  # Few paths = moderate
    else:
        path_score = 0.2  # No paths = poor
    
    # 4. Citation authority scoring - being cited indicates importance
    mean_authority = baseline.get('mean_citation_authority', 5.0)
    if citation_authority >= 2 * mean_authority:
        authority_score = 1.0  # High authority = excellent
    elif citation_authority >= mean_authority:
        authority_score = 0.8  # Above average authority = good
    elif citation_authority >= 0.5 * mean_authority:
        authority_score = 0.6  # Moderate authority = fair
    elif citation_authority > 0:
        authority_score = 0.4  # Some authority = poor
    else:
        authority_score = 0.2  # No authority = very poor
    
    # 5. Semantic coherence scoring - consistent semantic connections
    if semantic_coherence >= 0.7:
        coherence_score = 1.0  # High coherence = excellent
    elif semantic_coherence >= 0.5:
        coherence_score = 0.8  # Good coherence = good
    elif semantic_coherence >= 0.3:
        coherence_score = 0.6  # Moderate coherence = fair
    elif semantic_coherence > 0:
        coherence_score = 0.4  # Low coherence = poor
    else:
        coherence_score = 0.3  # No coherence = very poor
    
    # 6. Network position scoring - centrality measures
    if network_position_score >= 0.7:
        position_score = 1.0  # High centrality = excellent
    elif network_position_score >= 0.5:
        position_score = 0.8  # Good centrality = good
    elif network_position_score >= 0.3:
        position_score = 0.6  # Moderate centrality = fair
    elif network_position_score > 0:
        position_score = 0.4  # Low centrality = poor
    else:
        position_score = 0.3  # No centrality = very poor
    
    # Combine efficiency scores with adaptive weights
    # Emphasize path efficiency for outlier detection
    efficiency_score = (
        0.15 * clustering_score +
        0.15 * diversity_score +
        0.35 * path_score +        # Most important for outlier detection
        0.20 * authority_score +
        0.10 * coherence_score +
        0.05 * position_score
    )
    
    return min(1.0, max(0.0, efficiency_score))


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
        'isolation': 0.25 * (1 + sparsity_factor * 0.3),  # More weight to isolation in sparse datasets
        'coupling': 0.18 * (1 + sparsity_factor * 0.2),
        'neighborhood': 0.17 * (1 - sparsity_factor * 0.2),
        'advanced': 0.12 * (1 + sparsity_factor * 0.4),  # More weight to advanced features in sparse datasets
        'temporal': 0.13 * (1 + sparsity_factor * 0.5),   # Temporal features especially important for sparse datasets
        'efficiency': 0.15 * (1 + sparsity_factor * 0.6)  # Efficiency features very important for sparse datasets
    }
    
    # Normalize weights to sum to 1.0
    weight_sum = sum(weights.values())
    weights = {k: v / weight_sum for k, v in weights.items()}
    
    return weights 