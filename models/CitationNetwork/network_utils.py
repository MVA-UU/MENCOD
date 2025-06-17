"""
Network utilities for citation analysis.

This module contains functions for building, analyzing, and manipulating citation networks
for outlier detection in scientific document collections.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import json


def build_network_from_simulation(simulation_df: pd.DataFrame) -> nx.DiGraph:
    """Build a citation network from simulation data with both semantic and citation edges."""
    # Use directed graph to properly represent citation relationships
    G = nx.DiGraph()
    
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
    
    # Add direct citation edges
    add_citation_edges(G, simulation_df)
    
    # Add semantic similarity edges (undirected edges for content similarity)
    add_semantic_edges(G, simulation_df)
    
    # Add co-citation and bibliographic coupling edges
    add_cocitation_edges(G, simulation_df)
    add_bibliographic_coupling_edges(G, simulation_df)
    
    return G


def add_citation_edges(G: nx.DiGraph, simulation_df: pd.DataFrame) -> None:
    """Add direct citation edges with weights based on citation frequency."""
    print("Adding direct citation edges...")
    
    # Create mapping from URL to openalex_id for efficient lookup
    url_to_id = {}
    for _, row in simulation_df.iterrows():
        openalex_id = row['openalex_id']
        # Handle both full URLs and just IDs
        if openalex_id.startswith('https://openalex.org/'):
            url_to_id[openalex_id] = openalex_id
            url_to_id[openalex_id.split('/')[-1]] = openalex_id
        else:
            url_to_id[f"https://openalex.org/{openalex_id}"] = openalex_id
            url_to_id[openalex_id] = openalex_id
    
    citation_count = 0
    total_papers = len(simulation_df)
    
    for idx, row in simulation_df.iterrows():
        citing_paper = row['openalex_id']
        referenced_works = row.get('referenced_works', [])
        
        if referenced_works and isinstance(referenced_works, list):
            for ref_url in referenced_works:
                # Clean and normalize the reference URL/ID
                ref_clean = ref_url.strip()
                
                # Try to find the referenced paper in our dataset
                cited_paper = None
                if ref_clean in url_to_id:
                    cited_paper = url_to_id[ref_clean]
                elif ref_clean.startswith('https://openalex.org/'):
                    ref_id = ref_clean.split('/')[-1]
                    if ref_id in url_to_id:
                        cited_paper = url_to_id[ref_id]
                
                if cited_paper and cited_paper in G.nodes and citing_paper != cited_paper:
                    # Add directed edge from citing to cited paper
                    if G.has_edge(citing_paper, cited_paper):
                        # Increase weight if edge already exists
                        G[citing_paper][cited_paper]['weight'] += 1
                        G[citing_paper][cited_paper]['citation_count'] += 1
                    else:
                        # Add new citation edge
                        G.add_edge(citing_paper, cited_paper, 
                                 edge_type='citation', 
                                 weight=2.0,  # Higher weight for direct citations
                                 citation_count=1)
                    citation_count += 1
        
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{total_papers} papers, found {citation_count} citation edges")
    
    print(f"Added {citation_count} direct citation edges")


def add_semantic_edges(G: nx.DiGraph, simulation_df: pd.DataFrame) -> None:
    """Add semantic similarity edges based on text content."""
    print("Adding semantic similarity edges...")
    
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
        _create_semantic_edges_in_batches(G, tfidf_matrix, doc_ids, relevant_indices, 
                                        similarity_threshold, relevant_threshold)
        
    except Exception as e:
        print(f"Error creating similarity edges: {e}")
        import traceback
        traceback.print_exc()


def add_cocitation_edges(G: nx.DiGraph, simulation_df: pd.DataFrame) -> None:
    """Add co-citation edges between papers cited together."""
    print("Adding co-citation edges...")
    
    # Create efficient mapping from URL to openalex_id
    url_to_id = {}
    id_set = set()
    for _, row in simulation_df.iterrows():
        openalex_id = row['openalex_id']
        id_set.add(openalex_id)
        # Handle both full URLs and just IDs
        if openalex_id.startswith('https://openalex.org/'):
            url_to_id[openalex_id] = openalex_id
            url_to_id[openalex_id.split('/')[-1]] = openalex_id
        else:
            url_to_id[f"https://openalex.org/{openalex_id}"] = openalex_id
            url_to_id[openalex_id] = openalex_id
    
    # Build co-citation matrix efficiently
    cocitation_counts = defaultdict(lambda: defaultdict(int))
    total_papers = len(simulation_df)
    
    # Process papers in batches for progress reporting
    for idx, row in simulation_df.iterrows():
        citing_paper = row['openalex_id']
        referenced_works = row.get('referenced_works', [])
        
        if referenced_works and isinstance(referenced_works, list):
            # Find cited papers that are in our dataset
            cited_papers_in_dataset = []
            
            for ref_url in referenced_works:
                ref_clean = ref_url.strip()
                
                # Quick lookup using our mapping
                cited_paper = None
                if ref_clean in url_to_id:
                    cited_paper = url_to_id[ref_clean]
                elif ref_clean.startswith('https://openalex.org/'):
                    ref_id = ref_clean.split('/')[-1]
                    if ref_id in url_to_id:
                        cited_paper = url_to_id[ref_id]
                
                if cited_paper and cited_paper in G.nodes:
                    cited_papers_in_dataset.append(cited_paper)
            
            # Calculate co-citation for pairs of cited papers
            for i, paper1 in enumerate(cited_papers_in_dataset):
                for j, paper2 in enumerate(cited_papers_in_dataset):
                    if i < j:  # Avoid duplicates and self-citations
                        cocitation_counts[paper1][paper2] += 1
                        cocitation_counts[paper2][paper1] += 1
        
        # Progress reporting
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{total_papers} papers for co-citation analysis")
    
    # Add co-citation edges
    cocitation_edge_count = 0
    min_cocitation_threshold = 2  # Minimum co-citation threshold
    
    for paper1, cited_by in cocitation_counts.items():
        for paper2, count in cited_by.items():
            if count >= min_cocitation_threshold:
                # Co-citation weight (lower than direct citation)
                weight = min(1.5, 0.5 + count * 0.1)
                
                if not G.has_edge(paper1, paper2):
                    G.add_edge(paper1, paper2, edge_type='cocitation', weight=weight, cocitation_count=count)
                    cocitation_edge_count += 1
                else:
                    # Update existing edge
                    G[paper1][paper2]['cocitation_count'] = count
                    G[paper1][paper2]['weight'] = max(G[paper1][paper2]['weight'], weight)
                
                if not G.has_edge(paper2, paper1):
                    G.add_edge(paper2, paper1, edge_type='cocitation', weight=weight, cocitation_count=count)
                    cocitation_edge_count += 1
                else:
                    # Update existing edge
                    G[paper2][paper1]['cocitation_count'] = count
                    G[paper2][paper1]['weight'] = max(G[paper2][paper1]['weight'], weight)
    
    print(f"Added {cocitation_edge_count} co-citation edges")


def add_bibliographic_coupling_edges(G: nx.DiGraph, simulation_df: pd.DataFrame) -> None:
    """Add bibliographic coupling edges between papers that cite the same references."""
    print("Adding bibliographic coupling edges...")
    
    # Build paper references efficiently
    paper_references = {}
    total_papers = len(simulation_df)
    
    for idx, row in simulation_df.iterrows():
        paper_id = row['openalex_id']
        referenced_works = row.get('referenced_works', [])
        
        if referenced_works and isinstance(referenced_works, list):
            paper_references[paper_id] = set()
            
            for ref_url in referenced_works:
                ref_clean = ref_url.strip()
                paper_references[paper_id].add(ref_clean)
        
        # Progress reporting
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{total_papers} papers for bibliographic coupling")
    
    # Calculate bibliographic coupling counts efficiently
    coupling_counts = defaultdict(lambda: defaultdict(int))
    papers = [p for p in paper_references.keys() if p in G.nodes]
    
    print(f"Calculating coupling for {len(papers)} papers...")
    
    for i, paper1 in enumerate(papers):
        for j, paper2 in enumerate(papers):
            if i < j:  # Avoid duplicates
                # Count common references
                common_refs = paper_references[paper1].intersection(paper_references[paper2])
                if len(common_refs) >= 2:  # Minimum coupling threshold
                    coupling_counts[paper1][paper2] = len(common_refs)
                    coupling_counts[paper2][paper1] = len(common_refs)
        
        # Progress reporting for coupling calculation
        if (i + 1) % 200 == 0:
            print(f"Calculated coupling for {i + 1}/{len(papers)} papers")
    
    # Add bibliographic coupling edges
    coupling_edge_count = 0
    min_coupling_threshold = 2  # Minimum coupling threshold
    
    for paper1, coupled_with in coupling_counts.items():
        for paper2, count in coupled_with.items():
            if count >= min_coupling_threshold:
                # Bibliographic coupling weight (moderate strength)
                weight = min(1.8, 0.8 + count * 0.1)
                
                if not G.has_edge(paper1, paper2):
                    G.add_edge(paper1, paper2, edge_type='coupling', weight=weight, coupling_count=count)
                    coupling_edge_count += 1
                else:
                    # Update existing edge
                    G[paper1][paper2]['coupling_count'] = count
                    G[paper1][paper2]['weight'] = max(G[paper1][paper2]['weight'], weight)
                
                if not G.has_edge(paper2, paper1):
                    G.add_edge(paper2, paper1, edge_type='coupling', weight=weight, coupling_count=count)
                    coupling_edge_count += 1
                else:
                    # Update existing edge
                    G[paper2][paper1]['coupling_count'] = count
                    G[paper2][paper1]['weight'] = max(G[paper2][paper1]['weight'], weight)
    
    print(f"Added {coupling_edge_count} bibliographic coupling edges")


def _create_semantic_edges_in_batches(G: nx.DiGraph, tfidf_matrix, doc_ids: List[str], 
                                    relevant_indices: List[int], similarity_threshold: float,
                                    relevant_threshold: float) -> None:
    """Create semantic similarity edges efficiently using batched processing."""
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
                doc1, doc2 = doc_ids[relevant_indices[i]], doc_ids[relevant_indices[j]]
                sim_weight = min(1.0, rel_sims[i, j])  # Semantic similarity weight
                
                # Add undirected semantic edge (both directions)
                if not G.has_edge(doc1, doc2):
                    G.add_edge(doc1, doc2, edge_type='semantic', weight=sim_weight, similarity=rel_sims[i, j])
                    edge_count += 1
                else:
                    G[doc1][doc2]['similarity'] = rel_sims[i, j]
                    G[doc1][doc2]['weight'] = max(G[doc1][doc2]['weight'], sim_weight)
                
                if not G.has_edge(doc2, doc1):
                    G.add_edge(doc2, doc1, edge_type='semantic', weight=sim_weight, similarity=rel_sims[i, j])
                    edge_count += 1
                else:
                    G[doc2][doc1]['similarity'] = rel_sims[i, j]
                    G[doc2][doc1]['weight'] = max(G[doc2][doc1]['weight'], sim_weight)
    
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
                    doc1, doc2 = doc_ids[batch_idx], doc_ids[rel_idx]
                    sim_weight = min(1.0, cross_sims[i, j])
                    
                    # Add undirected semantic edge (both directions)
                    if not G.has_edge(doc1, doc2):
                        G.add_edge(doc1, doc2, edge_type='semantic', weight=sim_weight, similarity=cross_sims[i, j])
                        edge_count += 1
                    else:
                        G[doc1][doc2]['similarity'] = cross_sims[i, j]
                        G[doc1][doc2]['weight'] = max(G[doc1][doc2]['weight'], sim_weight)
                    
                    if not G.has_edge(doc2, doc1):
                        G.add_edge(doc2, doc1, edge_type='semantic', weight=sim_weight, similarity=cross_sims[i, j])
                        edge_count += 1
                    else:
                        G[doc2][doc1]['similarity'] = cross_sims[i, j]
                        G[doc2][doc1]['weight'] = max(G[doc2][doc1]['weight'], sim_weight)
        
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
                    doc1, doc2 = doc_ids[idx], doc_ids[j]
                    sim_weight = min(1.0, sim)
                    
                    # Add undirected semantic edge
                    if not G.has_edge(doc1, doc2):
                        G.add_edge(doc1, doc2, edge_type='semantic', weight=sim_weight, similarity=sim)
                        edge_count += 1
                    else:
                        G[doc1][doc2]['similarity'] = sim
                        G[doc1][doc2]['weight'] = max(G[doc1][doc2]['weight'], sim_weight)
        
        print(f"Processed batch {batch_start+1}-{batch_end} of {n_docs}, semantic edges so far: {edge_count}")
    
    print(f"Created {edge_count} semantic similarity edges")


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