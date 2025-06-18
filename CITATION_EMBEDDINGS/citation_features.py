"""
Citation network feature extraction utilities.

This module contains functions for extracting various features from citation networks,
used for outlier detection in scientific document collections.
"""

import networkx as nx
from typing import Dict, Set, List
import numpy as np
import re
from datetime import datetime
from collections import defaultdict


def get_connectivity_features(G: nx.Graph, doc_id: str, relevant_documents: Set[str]) -> Dict[str, float]:
    """Calculate connectivity-based features considering different edge types."""
    if doc_id not in G.nodes:
        return get_zero_connectivity_features(doc_id)
    
    # Basic connectivity measures
    out_degree = G.out_degree(doc_id) if hasattr(G, 'out_degree') else G.degree(doc_id)
    in_degree = G.in_degree(doc_id) if hasattr(G, 'in_degree') else G.degree(doc_id)
    
    # Edge type specific connectivity
    citation_out = 0  # Papers this document cites
    citation_in = 0   # Papers that cite this document
    semantic_edges = 0
    cocitation_edges = 0
    coupling_edges = 0
    
    # Outgoing edges (what this paper cites or is similar to)
    if hasattr(G, 'successors'):
        for neighbor in G.successors(doc_id):
            edge_data = G[doc_id][neighbor]
            edge_type = edge_data.get('edge_type', 'unknown')
            if edge_type == 'citation':
                citation_out += 1
            elif edge_type == 'semantic':
                semantic_edges += 1
            elif edge_type == 'cocitation':
                cocitation_edges += 1
            elif edge_type == 'coupling':
                coupling_edges += 1
    
    # Incoming edges (what cites this paper or is similar to it)
    if hasattr(G, 'predecessors'):
        for neighbor in G.predecessors(doc_id):
            edge_data = G[neighbor][doc_id]
            edge_type = edge_data.get('edge_type', 'unknown')
            if edge_type == 'citation':
                citation_in += 1
    
    # Calculate weighted connectivity based on academic research
    # Direct citations have higher importance for influence
    weighted_influence = citation_in * 2.0 + cocitation_edges * 1.5 + coupling_edges * 1.8
    weighted_connections = citation_out * 2.0 + semantic_edges * 1.0 + coupling_edges * 1.8
    
    # Connection to relevant documents (more important for outlier detection)
    relevant_connections = 0
    relevant_citation_out = 0
    relevant_citation_in = 0
    
    for neighbor in G.neighbors(doc_id):
        if neighbor in relevant_documents:
            relevant_connections += 1
            # Check edge types to relevant documents
            if G.has_edge(doc_id, neighbor):
                edge_data = G[doc_id][neighbor]
                if edge_data.get('edge_type') == 'citation':
                    relevant_citation_out += 1
            if G.has_edge(neighbor, doc_id):
                edge_data = G[neighbor][doc_id]
                if edge_data.get('edge_type') == 'citation':
                    relevant_citation_in += 1
    
    relevant_ratio = relevant_connections / max(1, out_degree + in_degree)
    
    return {
        'total_degree': out_degree + in_degree,
        'out_degree': out_degree,
        'in_degree': in_degree,
        'citation_out_degree': citation_out,
        'citation_in_degree': citation_in,
        'semantic_degree': semantic_edges,
        'cocitation_degree': cocitation_edges,
        'coupling_degree': coupling_edges,
        'weighted_influence': weighted_influence,
        'weighted_connections': weighted_connections,
        'relevant_connections': relevant_connections,
        'relevant_citation_out': relevant_citation_out,
        'relevant_citation_in': relevant_citation_in,
        'relevant_ratio': relevant_ratio,
        'citation_ratio': citation_in / max(1, citation_out + citation_in),
        'semantic_ratio': semantic_edges / max(1, out_degree + in_degree)
    }


def get_zero_connectivity_features(doc_id: str) -> Dict[str, float]:
    """Return zero connectivity features for isolated nodes."""
    return {
        'total_degree': 0.0,
        'out_degree': 0.0,
        'in_degree': 0.0,
        'citation_out_degree': 0.0,
        'citation_in_degree': 0.0,
        'semantic_degree': 0.0,
        'cocitation_degree': 0.0,
        'coupling_degree': 0.0,
        'weighted_influence': 0.0,
        'weighted_connections': 0.0,
        'relevant_connections': 0.0,
        'relevant_citation_out': 0.0,
        'relevant_citation_in': 0.0,
        'relevant_ratio': 0.0,
        'citation_ratio': 0.0,
        'semantic_ratio': 0.0
    }


def get_coupling_features(G: nx.Graph, doc_id: str, relevant_documents: Set[str]) -> Dict[str, float]:
    """Calculate bibliographic coupling and co-citation features."""
    if doc_id not in G.nodes:
        return {
            'coupling_score': 0.0,
            'cocitation_score': 0.0,
            'relevant_coupling': 0.0,
            'relevant_cocitation': 0.0,
            'coupling_diversity': 0.0,
            'cocitation_diversity': 0.0
        }
    
    coupling_scores = []
    cocitation_scores = []
    relevant_coupling_count = 0
    relevant_cocitation_count = 0
    
    # Calculate coupling and co-citation scores
    for neighbor in G.neighbors(doc_id):
        if G.has_edge(doc_id, neighbor):
            edge_data = G[doc_id][neighbor]
            
            # Bibliographic coupling
            if edge_data.get('edge_type') == 'coupling':
                coupling_count = edge_data.get('coupling_count', 0)
                coupling_scores.append(coupling_count)
                if neighbor in relevant_documents:
                    relevant_coupling_count += coupling_count
            
            # Co-citation
            elif edge_data.get('edge_type') == 'cocitation':
                cocitation_count = edge_data.get('cocitation_count', 0)
                cocitation_scores.append(cocitation_count)
                if neighbor in relevant_documents:
                    relevant_cocitation_count += cocitation_count
    
    # Aggregate coupling measures
    coupling_score = sum(coupling_scores) if coupling_scores else 0.0
    cocitation_score = sum(cocitation_scores) if cocitation_scores else 0.0
    
    # Diversity measures (based on distribution of coupling/co-citation strengths)
    coupling_diversity = np.std(coupling_scores) if len(coupling_scores) > 1 else 0.0
    cocitation_diversity = np.std(cocitation_scores) if len(cocitation_scores) > 1 else 0.0
    
    return {
        'coupling_score': coupling_score,
        'cocitation_score': cocitation_score,
        'relevant_coupling': relevant_coupling_count,
        'relevant_cocitation': relevant_cocitation_count,
        'coupling_diversity': coupling_diversity,
        'cocitation_diversity': cocitation_diversity
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


def get_efficiency_features(G: nx.Graph, doc_id: str, relevant_documents: Set[str]) -> Dict[str, float]:
    """Extract computationally efficient features for sparse datasets."""
    if doc_id not in G.nodes:
        return {
            'local_clustering': 0.0,
            'edge_type_diversity': 0.0,
            'relevant_path_efficiency': 0.0,
            'citation_authority': 0.0,
            'semantic_coherence': 0.0,
            'network_position_score': 0.0
        }
    
    neighbors = list(G.neighbors(doc_id))
    if not neighbors:
        return {
            'local_clustering': 0.0,
            'edge_type_diversity': 0.0,
            'relevant_path_efficiency': 0.0,
            'citation_authority': 0.0,
            'semantic_coherence': 0.0,
            'network_position_score': 0.0
        }
    
    # 1. Local clustering coefficient (O(k^2) where k is degree)
    local_clustering = 0.0
    if len(neighbors) > 1:
        possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
        actual_edges = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                if G.has_edge(n1, n2):
                    actual_edges += 1
        local_clustering = actual_edges / possible_edges if possible_edges > 0 else 0.0
    
    # 2. Edge type diversity (Shannon entropy of edge types)
    edge_types = []
    semantic_weight_sum = 0.0
    citation_count = 0
    
    for neighbor in neighbors:
        if G.has_edge(doc_id, neighbor):
            edge_data = G[doc_id][neighbor]
            edge_type = edge_data.get('edge_type', 'unknown')
            edge_types.append(edge_type)
            
            # Track semantic weights for coherence
            if edge_type == 'semantic':
                semantic_weight_sum += edge_data.get('weight', 0.0)
            elif edge_type == 'citation':
                citation_count += 1
    
    # Calculate edge type diversity
    edge_type_diversity = 0.0
    if edge_types:
        from collections import Counter
        type_counts = Counter(edge_types)
        total = len(edge_types)
        for count in type_counts.values():
            p = count / total
            if p > 0:
                edge_type_diversity -= p * np.log2(p)
    
    # 3. Relevant path efficiency (average shortest path to relevant docs)
    relevant_neighbors = [n for n in neighbors if n in relevant_documents]
    relevant_path_efficiency = 0.0
    if relevant_neighbors:
        # Direct connections to relevant docs are most efficient
        relevant_path_efficiency = len(relevant_neighbors) / len(neighbors)
        
        # Bonus for multiple relevant connections (network effect)
        if len(relevant_neighbors) > 1:
            relevant_path_efficiency *= (1 + 0.1 * (len(relevant_neighbors) - 1))
    
    # 4. Citation authority (weighted by edge types)
    citation_authority = 0.0
    if hasattr(G, 'in_degree'):
        in_degree = G.in_degree(doc_id)
        # Weight citation edges more heavily
        for pred in G.predecessors(doc_id):
            edge_data = G[pred][doc_id]
            if edge_data.get('edge_type') == 'citation':
                citation_authority += 2.0  # Direct citation
            elif edge_data.get('edge_type') == 'cocitation':
                citation_authority += 1.5  # Co-citation
            else:
                citation_authority += 1.0  # Other connections
    
    # 5. Semantic coherence (consistency of semantic connections)
    semantic_coherence = 0.0
    if semantic_weight_sum > 0:
        semantic_count = sum(1 for et in edge_types if et == 'semantic')
        if semantic_count > 0:
            semantic_coherence = semantic_weight_sum / semantic_count
    
    # 6. Network position score (combination of centrality measures)
    degree_centrality = len(neighbors) / (len(G.nodes) - 1) if len(G.nodes) > 1 else 0.0
    
    # Estimate betweenness efficiently using ego network
    ego_betweenness = 0.0
    if len(neighbors) > 2:
        # Count paths through this node in its ego network
        ego_graph = G.subgraph([doc_id] + neighbors)
        for n1 in neighbors:
            for n2 in neighbors:
                if n1 != n2 and not ego_graph.has_edge(n1, n2):
                    # This node is on the shortest path between n1 and n2
                    ego_betweenness += 1.0
        ego_betweenness /= (len(neighbors) * (len(neighbors) - 1) / 2)
    
    network_position_score = (degree_centrality + ego_betweenness) / 2.0
    
    return {
        'local_clustering': local_clustering,
        'edge_type_diversity': edge_type_diversity,
        'relevant_path_efficiency': relevant_path_efficiency,
        'citation_authority': citation_authority,
        'semantic_coherence': semantic_coherence,
        'network_position_score': network_position_score
    }


def get_temporal_features(G: nx.Graph, doc_id: str, relevant_documents: Set[str], simulation_data=None) -> Dict[str, float]:
    """Extract temporal citation patterns for improved outlier detection."""
    features = {
        'citation_velocity': 0.0,
        'age_normalized_impact': 0.0,
        'citation_burst_score': 0.0,
        'temporal_isolation': 0.0,
        'recent_citation_ratio': 0.0,
        'citation_acceleration': 0.0
    }
    
    if doc_id not in G.nodes:
        return features
    
    # Try to extract publication year from various sources
    pub_year = None
    current_year = datetime.now().year
    
    # Method 1: From node attributes
    if 'publication_year' in G.nodes[doc_id]:
        pub_year = G.nodes[doc_id]['publication_year']
    
    # Method 2: From simulation data if provided
    elif simulation_data is not None:
        doc_row = simulation_data[simulation_data['openalex_id'] == doc_id]
        if not doc_row.empty:
            # Try to extract year from DOI or other fields
            if 'doi' in doc_row.columns:
                doi = doc_row['doi'].iloc[0]
                if isinstance(doi, str):
                    # Extract year from DOI patterns
                    year_match = re.search(r'(19|20)\d{2}', doi)
                    if year_match:
                        pub_year = int(year_match.group())
    
    # Method 3: Extract from OpenAlex ID (sometimes contains year info)
    if pub_year is None:
        # OpenAlex IDs sometimes have patterns that indicate year
        year_match = re.search(r'W(\d{4})', doc_id)
        if year_match:
            potential_year = int(year_match.group(1))
            if 1950 <= potential_year <= current_year:
                pub_year = potential_year
    
    # If we have publication year, calculate temporal features
    if pub_year is not None and pub_year > 1950:
        age = max(1, current_year - pub_year)
        
        # Citation velocity (citations per year)
        citation_count = G.in_degree(doc_id) if hasattr(G, 'in_degree') else 0
        features['citation_velocity'] = citation_count / age
        
        # Age-normalized impact (accounts for citation accumulation over time)
        features['age_normalized_impact'] = citation_count / np.log(age + 1)
        
        # Citation acceleration (higher for papers that get cited quickly)
        if age > 1:
            # Estimate if this is a "fast" citation pattern
            expected_citations_for_age = age * 0.5  # Rough baseline
            if citation_count > expected_citations_for_age:
                features['citation_acceleration'] = min(2.0, citation_count / expected_citations_for_age)
    
    # Citation burst detection (independent of publication year)
    features['citation_burst_score'] = calculate_citation_burst(G, doc_id)
    
    # Temporal isolation - how connected is this doc to recently active papers
    features['temporal_isolation'] = calculate_temporal_isolation(G, doc_id, relevant_documents)
    
    # Recent citation ratio - what fraction of citations are to recent papers
    if hasattr(G, 'successors'):
        total_citations_out = 0
        recent_citations_out = 0
        
        for cited_doc in G.successors(doc_id):
            edge_data = G[doc_id][cited_doc]
            if edge_data.get('edge_type') == 'citation':
                total_citations_out += 1
                
                # Check if cited document is "recent" (has high activity)
                cited_degree = G.in_degree(cited_doc) if hasattr(G, 'in_degree') else G.degree(cited_doc)
                if cited_degree > 5:  # Threshold for "active" papers
                    recent_citations_out += 1
        
        if total_citations_out > 0:
            features['recent_citation_ratio'] = recent_citations_out / total_citations_out
    
    return features


def calculate_citation_burst(G: nx.Graph, doc_id: str) -> float:
    """Calculate citation burst score - detects papers with unusual citation patterns."""
    if not hasattr(G, 'in_degree'):
        return 0.0
    
    # Get citing papers
    citing_papers = list(G.predecessors(doc_id)) if hasattr(G, 'predecessors') else []
    if len(citing_papers) < 2:
        return 0.0
    
    # Calculate citation concentration
    # If most citations come from a small number of highly-cited papers, it's a burst
    citing_degrees = [G.in_degree(paper) for paper in citing_papers]
    
    if not citing_degrees:
        return 0.0
    
    # Measure concentration using coefficient of variation
    mean_degree = np.mean(citing_degrees)
    std_degree = np.std(citing_degrees)
    
    if mean_degree > 0:
        concentration = std_degree / mean_degree
        # Higher concentration indicates burst pattern
        return min(1.0, concentration / 2.0)
    
    return 0.0


def calculate_temporal_isolation(G: nx.Graph, doc_id: str, relevant_documents: Set[str]) -> float:
    """Calculate how temporally isolated this document is from relevant documents."""
    if not relevant_documents or doc_id not in G.nodes:
        return 1.0  # Completely isolated
    
    # Check connections to relevant documents
    neighbors = set(G.neighbors(doc_id))
    relevant_neighbors = neighbors.intersection(relevant_documents)
    
    if relevant_neighbors:
        return 0.0  # Not isolated - directly connected to relevant docs
    
    # Check 2-hop connections to relevant documents
    second_hop_relevant = 0
    total_second_hop = 0
    
    for neighbor in neighbors:
        neighbor_neighbors = set(G.neighbors(neighbor))
        total_second_hop += len(neighbor_neighbors)
        second_hop_relevant += len(neighbor_neighbors.intersection(relevant_documents))
    
    if total_second_hop == 0:
        return 1.0  # Completely isolated
    
    # Calculate isolation as inverse of connection ratio
    connection_ratio = second_hop_relevant / total_second_hop
    return 1.0 - connection_ratio


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
        'semantic_isolation': 0.0,
        # Temporal features
        'citation_velocity': 0.0,
        'age_normalized_impact': 0.0,
        'citation_burst_score': 0.0,
        'temporal_isolation': 0.0,
        'recent_citation_ratio': 0.0,
        'citation_acceleration': 0.0,
        # Efficiency features
        'local_clustering': 0.0,
        'edge_type_diversity': 0.0,
        'relevant_path_efficiency': 0.0,
        'citation_authority': 0.0,
        'semantic_coherence': 0.0,
        'network_position_score': 0.0
    } 