"""
Citation network feature extraction utilities.

This module contains functions for extracting various features from citation networks,
used for outlier detection in scientific document collections.
"""

import networkx as nx
from typing import Dict, Set, List


def get_connectivity_features(G: nx.Graph, doc_id: str, relevant_documents: Set[str]) -> Dict[str, float]:
    """Calculate direct citation connectivity features."""
    neighbors = list(G.neighbors(doc_id))
    if not neighbors:
        return {'total_citations': 0, 'relevant_connections': 0, 'relevant_connections_ratio': 0.0}
    
    # Use set operations for better performance
    neighbors_set = set(neighbors)
    rel_conn = len(neighbors_set.intersection(relevant_documents))
    
    return {
        'total_citations': len(neighbors),
        'relevant_connections': rel_conn,
        'relevant_connections_ratio': rel_conn / len(neighbors)
    }


def get_coupling_features(G: nx.Graph, doc_id: str, relevant_documents: Set[str]) -> Dict[str, float]:
    """Calculate bibliographic coupling features."""
    doc_refs = set(G.neighbors(doc_id))
    if not doc_refs:
        return {'max_coupling_strength': 0.0, 'avg_coupling_relevant': 0.0, 'coupling_above_threshold': 0}
    
    # Cache the document's connections
    doc_ref_count = len(doc_refs)
    
    # Calculate coupling with all relevant documents in one pass
    couplings = []
    coupling_sum = 0.0
    above_threshold = 0
    
    for rel_doc in relevant_documents:
        if rel_doc == doc_id:
            continue
            
        # Get relevant document connections
        rel_refs = set(G.neighbors(rel_doc))
        rel_ref_count = len(rel_refs)
        
        # Skip documents with no connections
        if not rel_refs:
            continue
            
        # Calculate Jaccard similarity
        shared_count = len(doc_refs.intersection(rel_refs))
        if shared_count > 0:
            jaccard = shared_count / (doc_ref_count + rel_ref_count - shared_count)
            couplings.append(jaccard)
            coupling_sum += jaccard
            if jaccard > 0.1:
                above_threshold += 1
    
    if not couplings:
        return {'max_coupling_strength': 0.0, 'avg_coupling_relevant': 0.0, 'coupling_above_threshold': 0}
    
    return {
        'max_coupling_strength': max(couplings),
        'avg_coupling_relevant': coupling_sum / len(couplings),
        'coupling_above_threshold': above_threshold
    }


def get_neighborhood_features(G: nx.Graph, doc_id: str, relevant_documents: Set[str]) -> Dict[str, float]:
    """Calculate citation neighborhood enrichment features."""
    # Get 1-hop neighbors efficiently
    hop1_neighbors = set(G.neighbors(doc_id))
    if not hop1_neighbors:
        return {
            'neighborhood_size_1hop': 0,
            'neighborhood_size_2hop': 0,
            'neighborhood_enrichment_1hop': 0.0,
            'neighborhood_enrichment_2hop': 0.0
        }
        
    # Count relevant neighbors in 1-hop
    hop1_size = len(hop1_neighbors)
    hop1_rel_count = sum(1 for n in hop1_neighbors if n in relevant_documents)
    
    # Get 2-hop neighbors efficiently
    hop2_neighbors = set()
    for neighbor in hop1_neighbors:
        hop2_neighbors.update(G.neighbors(neighbor))
    
    # Remove 1-hop and self from 2-hop
    hop2_neighbors.difference_update(hop1_neighbors)
    hop2_neighbors.discard(doc_id)
    
    # Count relevant neighbors in 2-hop
    hop2_size = len(hop2_neighbors)
    hop2_rel_count = sum(1 for n in hop2_neighbors if n in relevant_documents)
    
    # Calculate enrichment ratios with safe division
    hop1_enrichment = hop1_rel_count / max(hop1_size, 1)
    hop2_enrichment = hop2_rel_count / max(hop2_size, 1)
    
    return {
        'neighborhood_size_1hop': hop1_size,
        'neighborhood_size_2hop': hop2_size,
        'neighborhood_enrichment_1hop': hop1_enrichment,
        'neighborhood_enrichment_2hop': hop2_enrichment
    }


def get_advanced_features(G: nx.Graph, doc_id: str, relevant_documents: Set[str]) -> Dict[str, float]:
    """Calculate advanced citation network features for better outlier detection."""
    # Initialize empty results
    results = {
        'citation_diversity': 0.0,
        'relevant_betweenness': 0.0,
        'structural_anomaly': 0.0
    }
    
    # Skip if no neighbors
    if G.degree(doc_id) == 0:
        return results
    
    # 1. Citation diversity - how diverse are the document's connections
    neighbors = set(G.neighbors(doc_id))
    neighbor_connections = []
    
    # Calculate how connected the neighbors are to each other
    for n1 in neighbors:
        n1_neighbors = set(G.neighbors(n1))
        connections = sum(1 for n2 in neighbors if n2 in n1_neighbors)
        neighbor_connections.append(connections)
    
    if neighbor_connections:
        # Higher diversity means neighbors are less connected to each other
        avg_connections = sum(neighbor_connections) / len(neighbor_connections)
        max_possible = len(neighbors) - 1
        if max_possible > 0:
            results['citation_diversity'] = 1.0 - (avg_connections / max_possible)
    
    # 2. Relevant betweenness - is the document a bridge to relevant docs?
    # Optimized version - only check immediate relevant neighbors
    if not relevant_documents:
        return results
        
    # Only consider relevant documents that are 1 or 2 hops away
    relevant_neighbors = neighbors.intersection(relevant_documents)
    
    # If this document directly connects to multiple relevant docs,
    # check if it serves as a bridge between them
    if len(relevant_neighbors) >= 2:
        # Check if this document bridges relevant neighbors that aren't directly connected
        bridged_pairs = 0
        rel_list = list(relevant_neighbors)
        
        for i in range(len(rel_list)):
            for j in range(i+1, len(rel_list)):
                rel1, rel2 = rel_list[i], rel_list[j]
                # Check if rel1 and rel2 are directly connected
                if rel2 not in set(G.neighbors(rel1)):
                    bridged_pairs += 1
        
        # Normalize by number of potential pairs
        max_pairs = (len(relevant_neighbors) * (len(relevant_neighbors) - 1)) // 2
        if max_pairs > 0:
            results['relevant_betweenness'] = bridged_pairs / max_pairs
    elif len(relevant_neighbors) == 1:
        # If connected to exactly one relevant doc, check 2-hop connections
        rel_doc = next(iter(relevant_neighbors))
        rel_neighbors = set(G.neighbors(rel_doc)).intersection(relevant_documents)
        
        # If the relevant neighbor connects to other relevant docs,
        # this document might be an outlier bridge
        if rel_neighbors:
            results['relevant_betweenness'] = 0.3  # Moderate betweenness score
    
    # 3. Structural anomaly - how different is this doc's connection pattern
    neighbors_degree = [G.degree(n) for n in neighbors]
    if neighbors_degree:
        doc_degree = G.degree(doc_id)
        avg_neighbor_degree = sum(neighbors_degree) / len(neighbors_degree)
        # If doc has many connections but connects to isolated nodes = anomalous
        if avg_neighbor_degree > 0:
            ratio = doc_degree / avg_neighbor_degree
            if ratio > 2.0:  # Doc is hub connecting to less-connected nodes
                results['structural_anomaly'] = min(1.0, (ratio - 2.0) / 3.0)
            elif ratio < 0.5:  # Doc is peripheral but connects to hubs
                results['structural_anomaly'] = min(1.0, (0.5 - ratio) / 0.4)
    
    # 4. Semantic isolation - for extremely sparse datasets, detect semantic outliers
    relevant_ratio = len(relevant_documents) / len(G.nodes) if G.nodes else 0
    if relevant_ratio < 0.02:  # Only calculate for very sparse datasets
        # Check if this document has any semantic connections to relevant documents
        has_connections = any(rel_doc in G.neighbors(doc_id) for rel_doc in relevant_documents 
                            if rel_doc in G.nodes and doc_id != rel_doc)
        
        # Calculate semantic isolation by looking at second-order connections
        if not has_connections and neighbors:
            # How many of this document's neighbors are connected to relevant documents?
            neighbors_with_relevant_conn = 0
            for neighbor in neighbors:
                if any(rel_doc in G.neighbors(neighbor) for rel_doc in relevant_documents 
                       if rel_doc in G.nodes and neighbor != rel_doc):
                    neighbors_with_relevant_conn += 1
            
            semantic_isolation = 1.0 - (neighbors_with_relevant_conn / len(neighbors))
            
            # If this document has low semantic similarity to relevant docs
            # it's more likely to be an outlier
            results['semantic_isolation'] = semantic_isolation
            
            # Boost structural anomaly for semantically isolated documents
            if semantic_isolation > 0.7:
                results['structural_anomaly'] = max(results['structural_anomaly'], 0.5)
    
    return results


def get_zero_features(doc_id: str) -> Dict[str, float]:
    """Return zero features for documents not in citation network."""
    return {
        'openalex_id': doc_id, 
        'total_citations': 0, 
        'relevant_connections': 0,
        'relevant_connections_ratio': 0.0, 
        'max_coupling_strength': 0.0,
        'avg_coupling_relevant': 0.0, 
        'coupling_above_threshold': 0,
        'neighborhood_size_1hop': 0, 
        'neighborhood_size_2hop': 0,
        'neighborhood_enrichment_1hop': 0.0, 
        'neighborhood_enrichment_2hop': 0.0,
        'citation_diversity': 0.0,
        'relevant_betweenness': 0.0,
        'structural_anomaly': 0.0,
        'semantic_isolation': 0.0
    } 