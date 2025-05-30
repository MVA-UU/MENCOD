"""
Network utilities for citation analysis.

This module contains functions for building, analyzing, and manipulating citation networks
for outlier detection in scientific document collections.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_network_from_simulation(simulation_df: pd.DataFrame) -> nx.Graph:
    """Build a citation network from simulation data."""
    G = nx.Graph()
    
    # Add all documents as nodes
    for _, row in simulation_df.iterrows():
        doc_id = row['openalex_id']
        title = str(row.get('title', '') or '')
        abstract = str(row.get('abstract', '') or '')
        
        # Add the node with title and abstract
        G.add_node(doc_id, title=title, abstract=abstract)
        
        # Add label information
        G.nodes[doc_id]['label_included'] = row['label_included']
        G.nodes[doc_id]['record_id'] = row['record_id']
        
        # If asreview columns exist, add them
        if 'asreview_ranking' in row:
            G.nodes[doc_id]['asreview_ranking'] = row['asreview_ranking']
        if 'asreview_prior' in row:
            G.nodes[doc_id]['asreview_prior'] = row['asreview_prior']
    
    print(f"Created graph with {len(G.nodes)} nodes")
    return G


def create_similarity_edges(G: nx.Graph, simulation_df: pd.DataFrame) -> None:
    """Create edges between documents based on text similarity."""
    print(f"Creating edges based on text similarity for {len(simulation_df)} documents...")
    
    # Create text representations for TF-IDF
    documents = []
    doc_ids = []
    
    for _, row in simulation_df.iterrows():
        doc_id = row['openalex_id']
        title = str(row.get('title', '') or '')
        abstract = str(row.get('abstract', '') or '')
        
        if title or abstract:
            text = f"{title}. {abstract}".strip()
            documents.append(text)
            doc_ids.append(doc_id)
    
    if not documents:
        print("Warning: No documents with text found")
        return
    
    # Calculate TF-IDF vectors
    print("Calculating TF-IDF vectors...")
    
    # Identify relevant document count for adaptive max_features
    relevant_count = len([
        row['openalex_id'] for _, row in simulation_df.iterrows() 
        if row.get('label_included', 0) == 1
    ])
    total_count = len(simulation_df)
    
    # Calculate adaptive max_features based on dataset characteristics
    base_features = min(250, max(100, relevant_count * 8))
    sparsity_factor = np.sqrt(relevant_count / total_count)
    
    # More aggressive feature reduction for extremely sparse datasets
    if relevant_count < 30 and (relevant_count / total_count) < 0.02:
        max_features = min(250, relevant_count * 10)
        print(f"Using reduced feature set for extremely sparse dataset")
    else:
        max_features = min(1000, int(base_features / sparsity_factor))
    
    print(f"Using adaptive TF-IDF with {max_features} features")
    
    try:
        # Fit TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Identify relevant document indices
        relevant_doc_ids = set(
            row['openalex_id'] for _, row in simulation_df.iterrows() 
            if row.get('label_included', 0) == 1
        )
        
        # Map to indices in a single pass
        relevant_indices = [
            i for i, doc_id in enumerate(doc_ids) 
            if doc_id in relevant_doc_ids
        ]
        
        # Calculate threshold using vector norms and statistics
        doc_norms = np.sqrt(np.array(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)).flatten())
        
        # Calculate mean dot product between relevant documents
        rel_dots = []
        if len(relevant_indices) > 1:
            # Use only a subset of relevant pairs if there are many
            max_pairs = min(500, len(relevant_indices) * (len(relevant_indices) - 1) // 2)
            pairs_sampled = 0
            
            for i in range(len(relevant_indices)):
                for j in range(i+1, len(relevant_indices)):
                    if pairs_sampled >= max_pairs:
                        break
                    idx1, idx2 = relevant_indices[i], relevant_indices[j]
                    # Efficient dot product calculation
                    dot = tfidf_matrix[idx1].dot(tfidf_matrix[idx2].T)[0, 0]
                    # Normalize by norms to get cosine similarity
                    sim = dot / (doc_norms[idx1] * doc_norms[idx2])
                    rel_dots.append(sim)
                    pairs_sampled += 1
        
        # Estimate a good threshold based on statistics
        if rel_dots:
            mean_rel_sim = np.mean(rel_dots)
            std_rel_sim = np.std(rel_dots)
            # Threshold based on relevant document similarities
            similarity_threshold = max(0.15, mean_rel_sim - std_rel_sim)
        else:
            # Fallback if no relevant pairs
            similarity_threshold = 0.15
        
        # Adaptive threshold with bounds
        similarity_threshold = max(min(similarity_threshold, 0.5), 0.15)
        
        # Use a lower threshold for connecting to relevant documents
        relevant_threshold = max(similarity_threshold * 0.8, 0.1)
        
        print(f"Using similarity thresholds: general={similarity_threshold:.4f}, relevant={relevant_threshold:.4f}")
        
        # Process documents in batches for efficiency
        _create_edges_in_batches(G, tfidf_matrix, doc_ids, relevant_indices, 
                               similarity_threshold, relevant_threshold)
        
    except Exception as e:
        print(f"Error creating similarity edges: {e}")
        import traceback
        traceback.print_exc()


def _create_edges_in_batches(G: nx.Graph, tfidf_matrix, doc_ids: List[str], 
                           relevant_indices: List[int], similarity_threshold: float,
                           relevant_threshold: float) -> None:
    """Create similarity edges efficiently using batched processing."""
    n_docs = tfidf_matrix.shape[0]
    edge_count = 0
    
    # 1. First connect relevant documents to each other
    if relevant_indices:
        rel_matrix = tfidf_matrix[relevant_indices]
        # Get pairwise similarities between relevant docs
        rel_sims = rel_matrix.dot(rel_matrix.T).toarray()
        np.fill_diagonal(rel_sims, 0)  # Remove self-similarities
        
        # Add edges between relevant docs
        rel_edges = np.where(rel_sims >= relevant_threshold)
        for i, j in zip(rel_edges[0], rel_edges[1]):
            if i < j:  # Avoid duplicates
                G.add_edge(doc_ids[relevant_indices[i]], doc_ids[relevant_indices[j]])
                edge_count += 1
    
    # 2. Efficiently connect non-relevant docs
    # Process in manageable batch sizes
    batch_size = 200
    for batch_start in range(0, n_docs, batch_size):
        batch_end = min(batch_start + batch_size, n_docs)
        batch = list(range(batch_start, batch_end))
        
        # a. Connect batch docs to relevant docs
        if relevant_indices:
            # Compute similarities between batch and relevant docs
            batch_matrix = tfidf_matrix[batch]
            rel_matrix = tfidf_matrix[relevant_indices]
            cross_sims = batch_matrix.dot(rel_matrix.T).toarray()
            
            # Find pairs above threshold
            batch_rel_edges = np.where(cross_sims >= relevant_threshold)
            for i, j in zip(batch_rel_edges[0], batch_rel_edges[1]):
                batch_idx = batch[i]
                rel_idx = relevant_indices[j]
                if batch_idx != rel_idx:
                    G.add_edge(doc_ids[batch_idx], doc_ids[rel_idx])
                    edge_count += 1
        
        # b. Connect batch docs to their top neighbors
        # For each doc in batch, find top-k neighbors
        k = min(10, n_docs - 1)  # Number of neighbors to find
        batch_matrix = tfidf_matrix[batch]
        
        for i, idx in enumerate(batch):
            # Skip relevant docs (already processed)
            if idx in relevant_indices:
                continue
            
            # Compute similarities with all docs
            sims = tfidf_matrix.dot(batch_matrix[i].T).toarray().flatten()
            sims[idx] = 0  # Remove self-similarity
            
            # Find top-k neighbors
            top_k_indices = np.argsort(sims)[-k:]
            top_k_sims = sims[top_k_indices]
            
            # Add edges to neighbors above threshold
            for j, sim in zip(top_k_indices, top_k_sims):
                threshold = relevant_threshold if j in relevant_indices else similarity_threshold
                if sim >= threshold:
                    G.add_edge(doc_ids[idx], doc_ids[j])
                    edge_count += 1
        
        print(f"Processed batch {batch_start+1}-{batch_end} of {n_docs}, edges so far: {edge_count}")
    
    print(f"Created {edge_count} edges based on text similarity")


def calculate_distance_baselines(G: nx.Graph, relevant_documents: Set[str]) -> Dict[str, float]:
    """Calculate distance statistics from relevant documents for adaptive scoring."""
    if not relevant_documents:
        return {}
    
    # Calculate pairwise distances between relevant documents
    relevant_list = list(relevant_documents)
    distances = []
    
    for i, doc1 in enumerate(relevant_list):
        for doc2 in relevant_list[i+1:]:
            if doc1 in G.nodes and doc2 in G.nodes:
                try:
                    dist = nx.shortest_path_length(G, doc1, doc2)
                    distances.append(dist)
                except nx.NetworkXNoPath:
                    # If no path exists, consider this as max observable distance + 1
                    distances.append(10)
    
    if not distances:
        # Fallback values if no distances can be calculated
        return {
            'mean_distance': 3.0,
            'std_distance': 1.5,
            'median_distance': 3.0,
            'p25_distance': 2.0,
            'p75_distance': 4.0,
            'max_distance': 6.0
        }
    
    distances = np.array(distances)
    return {
        'mean_distance': float(np.mean(distances)),
        'std_distance': max(float(np.std(distances)), 0.5),
        'median_distance': float(np.median(distances)),
        'p25_distance': float(np.percentile(distances, 25)),
        'p75_distance': float(np.percentile(distances, 75)),
        'max_distance': float(np.max(distances))
    } 