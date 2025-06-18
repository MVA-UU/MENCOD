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
import os
import sys

# Add the models directory to the path to import synergy_dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
if models_dir not in sys.path:
    sys.path.append(models_dir)

from synergy_dataset import Dataset


def build_network_from_simulation(simulation_df: pd.DataFrame, dataset_name: str = None) -> nx.DiGraph:
    """Build a citation network from simulation data with both semantic and citation edges.
    
    Args:
        simulation_df: DataFrame containing simulation results
        dataset_name: Name of the dataset (e.g., 'appenzeller', 'hall', etc.) for loading synergy data
    """
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
    
    # Load synergy dataset for citation information if dataset_name is provided
    synergy_data = None
    if dataset_name:
        synergy_data = _load_synergy_dataset(dataset_name)
    
    # Add direct citation edges
    add_citation_edges(G, simulation_df, synergy_data)
    
    # Add semantic similarity edges (undirected edges for content similarity)
    add_semantic_edges(G, simulation_df)
    
    # Add co-citation and bibliographic coupling edges
    add_cocitation_edges(G, simulation_df, synergy_data)
    add_bibliographic_coupling_edges(G, simulation_df, synergy_data)
    
    return G


def _load_synergy_dataset(dataset_name: str) -> Optional[Dict]:
    """Load synergy dataset configuration and data.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'appenzeller', 'hall', etc.)
        
    Returns:
        Dictionary containing synergy dataset data or None if error
    """
    try:
        # Load datasets configuration
        # current_dir is CITATION_EMBEDDINGS, so parent is ADS Thesis
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, 'data', 'datasets.json')
        with open(config_path, 'r') as f:
            datasets_config = json.load(f)
        
        if dataset_name not in datasets_config:
            print(f"Warning: Dataset '{dataset_name}' not found in configuration")
            return None
            
        synergy_name = datasets_config[dataset_name]['synergy_dataset_name']
        print(f"Loading Synergy dataset: {synergy_name}")
        
        # Load the synergy dataset
        dataset = Dataset(synergy_name)
        synergy_data = dataset.to_dict(["id", "title", "referenced_works"])
        
        print(f"Loaded {len(synergy_data)} documents from Synergy dataset")
        return synergy_data
        
    except Exception as e:
        print(f"Error loading synergy dataset for {dataset_name}: {e}")
        return None


def add_citation_edges(G: nx.DiGraph, simulation_df: pd.DataFrame, synergy_data: Optional[Dict] = None) -> None:
    """Add direct citation edges with weights based on citation frequency."""
    print("Adding direct citation edges...")
    
    if synergy_data is None:
        print("Warning: No synergy data provided, cannot add citation edges")
        return
    
    # Create mapping from openalex_id in simulation to check what documents we have
    simulation_ids = set(simulation_df['openalex_id'].tolist())
    
    citation_count = 0
    total_papers = len(simulation_df)
    processed_papers = 0
    
    for _, row in simulation_df.iterrows():
        citing_paper = row['openalex_id']
        
        # Get referenced works from synergy data
        if citing_paper in synergy_data:
            referenced_works = synergy_data[citing_paper].get('referenced_works', [])
            
            if referenced_works and isinstance(referenced_works, list):
                for ref_id in referenced_works:
                    # Check if the referenced paper is in our simulation dataset
                    if ref_id in simulation_ids and ref_id != citing_paper and ref_id in G.nodes:
                        # Add directed edge from citing to cited paper
                        if G.has_edge(citing_paper, ref_id):
                            # Increase weight if edge already exists
                            G[citing_paper][ref_id]['weight'] += 1
                            G[citing_paper][ref_id]['citation_count'] += 1
                        else:
                            # Add new citation edge
                            G.add_edge(citing_paper, ref_id, 
                                     edge_type='citation', 
                                     weight=2.0,  # Higher weight for direct citations
                                     citation_count=1)
                        citation_count += 1
        
        processed_papers += 1
        if processed_papers % 500 == 0:
            print(f"Processed {processed_papers}/{total_papers} papers, found {citation_count} citation edges")
    
    print(f"Added {citation_count} direct citation edges")


def add_semantic_edges(G: nx.DiGraph, simulation_df: pd.DataFrame) -> None:
    """Add enhanced semantic similarity edges using SPECTER2 embeddings."""
    print("Adding semantic similarity edges using SPECTER2 embeddings...")
    
    # Load SPECTER2 embeddings and metadata
    # Get the project root directory (should be ADS Thesis)
    # current_dir is CITATION_EMBEDDINGS, so parent is ADS Thesis
    project_root = os.path.dirname(current_dir)
    embeddings_path = os.path.join(project_root, 'appenzeller_embeddings', 'specter2_embeddings.npy')
    metadata_path = os.path.join(project_root, 'appenzeller_embeddings', 'specter2_metadata.json')
    
    print(f"Looking for SPECTER2 embeddings at: {embeddings_path}")
    print(f"Looking for metadata at: {metadata_path}")
    
    if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
        print(f"ERROR: SPECTER2 embeddings not found!")
        print(f"Embeddings exists: {os.path.exists(embeddings_path)}")
        print(f"Metadata exists: {os.path.exists(metadata_path)}")
        print(f"Current dir: {current_dir}")
        print(f"Project root: {project_root}")
        raise FileNotFoundError("SPECTER2 embeddings are required for this citation network implementation")
    
    # Load embeddings and metadata
    try:
        embeddings = np.load(embeddings_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"Loaded SPECTER2 embeddings: {embeddings.shape}")
        print(f"Metadata for {metadata['num_documents']} documents")
    except Exception as e:
        print(f"Error loading SPECTER2 embeddings: {e}")
        raise RuntimeError(f"Failed to load SPECTER2 embeddings: {e}")
    
    # Create mapping from OpenAlex ID to embedding index
    openalex_to_index = {}
    for doc in metadata['documents']:
        # Clean the OpenAlex ID (remove https:// prefix if present)
        openalex_id = doc['openalex_id']
        if openalex_id.startswith('https://openalex.org/'):
            openalex_id = openalex_id.replace('https://openalex.org/', '')
        openalex_to_index[openalex_id] = doc['index']
    
    # Create list of document IDs that have embeddings and are in the simulation
    doc_ids = []
    embedding_indices = []
    
    for _, row in simulation_df.iterrows():
        doc_id = row['openalex_id']
        # Clean the simulation doc_id too
        clean_doc_id = doc_id
        if clean_doc_id.startswith('https://openalex.org/'):
            clean_doc_id = clean_doc_id.replace('https://openalex.org/', '')
        
        if clean_doc_id in openalex_to_index:
            doc_ids.append(doc_id)  # Keep original ID for graph
            embedding_indices.append(openalex_to_index[clean_doc_id])
        elif doc_id not in G.nodes:
            continue  # Skip if not in graph anyway
    
    if not doc_ids:
        print("Warning: No documents found with SPECTER2 embeddings")
        return
    
    print(f"Found {len(doc_ids)} documents with SPECTER2 embeddings in simulation")
    
    # Extract relevant embeddings
    relevant_embeddings = embeddings[embedding_indices]
    print(f"Using embeddings matrix shape: {relevant_embeddings.shape}")
    
    # Identify relevant document indices for enhanced similarity calculation
    relevant_indices = []
    for i, doc_id in enumerate(doc_ids):
        doc_row = simulation_df[simulation_df['openalex_id'] == doc_id]
        if not doc_row.empty and doc_row.iloc[0].get('label_included', 0) == 1:
            relevant_indices.append(i)
    
    print(f"Found {len(relevant_indices)} relevant documents for enhanced similarity")
    
    # Calculate similarity thresholds based on dataset sparsity
    relevant_count = len(relevant_indices)
    total_count = len(doc_ids)
    sparsity_factor = relevant_count / total_count if total_count > 0 else 0
    
    # Use very aggressive thresholds for SPECTER2 to prevent network explosion
    # SPECTER2 embeddings tend to have higher cosine similarities than TF-IDF
    print(f"Dataset sparsity factor: {sparsity_factor:.4f} ({relevant_count}/{total_count})")
    
    if total_count > 2000:  # Large dataset - use very high thresholds
        base_threshold = 0.95
        relevant_threshold = 0.93
        print("Large dataset: Using very aggressive thresholds to control network size")
    elif sparsity_factor < 0.01:  # Extremely sparse
        base_threshold = 0.93
        relevant_threshold = 0.90
        print("Extremely sparse dataset: Using very high thresholds")
    elif sparsity_factor < 0.05:  # Very sparse
        base_threshold = 0.90
        relevant_threshold = 0.87
        print("Very sparse dataset: Using high thresholds")
    else:  # Dense dataset
        base_threshold = 0.87
        relevant_threshold = 0.83
        print("Dense dataset: Using moderate thresholds")
    
    print(f"Final similarity thresholds: general={base_threshold:.4f}, relevant={relevant_threshold:.4f}")
    
    # Use smaller batch size for better memory management and progress tracking
    batch_size = 100
    print(f"Processing with batch size: {batch_size}")
    
    # Process in batches with enhanced edge weighting using SPECTER2 embeddings
    _create_enhanced_semantic_edges_from_embeddings(
        G, relevant_embeddings, doc_ids, relevant_indices, 
        base_threshold, relevant_threshold, batch_size
    )


def _create_enhanced_semantic_edges_from_embeddings(G: nx.DiGraph, embeddings_matrix, doc_ids: List[str], 
                                    relevant_indices: List[int], similarity_threshold: float,
                                    relevant_threshold: float, batch_size: int = 200) -> None:
    """Create semantic edges using SPECTER2 embeddings with enhanced weighting for outlier detection."""
    import time
    from tqdm import tqdm
    
    total_docs = len(doc_ids)
    semantic_edges = 0
    start_time = time.time()
    
    print(f"Starting semantic edge creation for {total_docs} documents...")
    print(f"Estimated total comparisons: ~{total_docs * (total_docs - 1) // 2:,}")
    print(f"Using similarity thresholds: general={similarity_threshold:.3f}, relevant={relevant_threshold:.3f}")
    print(f"Processing in batches of {batch_size} documents")
    
    # Pre-normalize embeddings for faster cosine similarity computation
    from sklearn.preprocessing import normalize
    normalized_embeddings = normalize(embeddings_matrix, norm='l2')
    print("Embeddings normalized for efficient cosine similarity computation")
    
    # Convert relevant indices to set for O(1) lookup
    relevant_set = set(relevant_indices)
    
    # Process documents in batches to manage memory and provide progress updates
    total_batches = (total_docs + batch_size - 1) // batch_size
    batch_count = 0
    
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        batch_indices = list(range(batch_start, batch_end))
        batch_count += 1
        
        # Calculate similarities for this batch against ALL documents
        batch_embeddings = normalized_embeddings[batch_indices]
        similarities = np.dot(batch_embeddings, normalized_embeddings.T)
        
        batch_edges = 0
        
        # Process each document in the batch
        for i, global_i in enumerate(batch_indices):
            doc_i = doc_ids[global_i]
            is_i_relevant = global_i in relevant_set
            
            # Only process similarities with documents that come after this one (avoid duplicates)
            for global_j in range(global_i + 1, total_docs):
                doc_j = doc_ids[global_j]
                is_j_relevant = global_j in relevant_set
                similarity = similarities[i, global_j]
                
                # Dynamic threshold based on document relevance
                if is_i_relevant or is_j_relevant:
                    threshold = relevant_threshold
                    edge_weight = 1.5  # Higher weight for relevant connections
                else:
                    threshold = similarity_threshold
                    edge_weight = 1.0
                
                if similarity >= threshold:
                    # Enhanced edge weighting based on similarity strength
                    if similarity >= 0.95:
                        weight_multiplier = 3.0  # Extremely high similarity
                    elif similarity >= 0.90:
                        weight_multiplier = 2.5  # Very high similarity
                    elif similarity >= 0.85:
                        weight_multiplier = 2.0  # High similarity
                    elif similarity >= 0.80:
                        weight_multiplier = 1.8  # Good similarity
                    else:
                        weight_multiplier = 1.2  # Moderate similarity
                    
                    final_weight = edge_weight * weight_multiplier * similarity
                    
                    # Add undirected semantic edge (both directions)
                    G.add_edge(doc_i, doc_j, 
                             edge_type='semantic', 
                             weight=final_weight,
                             similarity=similarity,
                             relevance_pattern=is_i_relevant or is_j_relevant,
                             embedding_type='specter2')
                    
                    G.add_edge(doc_j, doc_i, 
                             edge_type='semantic', 
                             weight=final_weight,
                             similarity=similarity,
                             relevance_pattern=is_i_relevant or is_j_relevant,
                             embedding_type='specter2')
                    
                    batch_edges += 2  # Count both directions
        
        semantic_edges += batch_edges
        elapsed_time = time.time() - start_time
        avg_time_per_batch = elapsed_time / batch_count
        estimated_total_time = avg_time_per_batch * total_batches
        remaining_time = estimated_total_time - elapsed_time
        
        # Progress update every batch
        print(f"Batch {batch_count}/{total_batches} | Docs {batch_start+1}-{batch_end} | "
              f"Edges this batch: {batch_edges} | Total edges: {semantic_edges} | "
              f"Elapsed: {elapsed_time:.1f}s | ETA: {remaining_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\nCompleted semantic edge creation in {total_time:.2f} seconds")
    print(f"Created {semantic_edges} enhanced semantic similarity edges using SPECTER2 embeddings")
    print(f"Average processing rate: {total_docs / total_time:.1f} documents/second")


def _add_tfidf_semantic_edges(G: nx.DiGraph, simulation_df: pd.DataFrame) -> None:
    """Fallback function to add TF-IDF based semantic edges when SPECTER2 embeddings are not available."""
    print("Adding TF-IDF-based semantic similarity edges...")
    
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
    
    # Enhanced feature selection for outlier detection
    if relevant_count < 30 and (relevant_count / total_count) < 0.02:
        # Extremely sparse: focus on discriminative features
        max_features = min(500, relevant_count * 15)
        min_df = max(2, int(total_count * 0.001))  # Very rare terms
        max_df = min(0.7, max(0.3, relevant_count / total_count * 20))  # Avoid too common terms
        print(f"Using focused feature set for extremely sparse dataset: {max_features} features")
    elif relevant_count < 100:
        # Moderately sparse: balanced approach
        max_features = min(1000, relevant_count * 10)
        min_df = max(2, int(total_count * 0.002))
        max_df = 0.85
        print(f"Using balanced feature set for sparse dataset: {max_features} features")
    else:
        # Dense dataset: comprehensive features
        max_features = min(2000, relevant_count * 8)
        min_df = max(3, int(total_count * 0.003))
        max_df = 0.9
        print(f"Using comprehensive feature set: {max_features} features")
    
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better semantic capture
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,  # Better handling of term frequency
            norm='l2'
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
    except Exception as e:
        print(f"Error creating TF-IDF matrix: {e}")
        return
    
    # Calculate pairwise similarities in memory-efficient batches
    batch_size = 200
    total_docs = len(doc_ids)
    
    # Identify relevant document indices for enhanced similarity calculation
    relevant_indices = []
    for i, doc_id in enumerate(doc_ids):
        doc_row = simulation_df[simulation_df['openalex_id'] == doc_id]
        if not doc_row.empty and doc_row.iloc[0].get('label_included', 0) == 1:
            relevant_indices.append(i)
    
    print(f"Found {len(relevant_indices)} relevant documents for enhanced similarity")
    
    # Dynamic similarity thresholds based on dataset sparsity
    base_threshold = 0.15
    relevant_threshold = 0.12
    
    # Adjust thresholds based on relevant document density
    sparsity_factor = relevant_count / total_count
    if sparsity_factor < 0.01:  # Extremely sparse
        base_threshold = 0.20
        relevant_threshold = 0.15
        print("Using high thresholds for extremely sparse dataset")
    elif sparsity_factor < 0.05:  # Very sparse
        base_threshold = 0.17
        relevant_threshold = 0.14
        print("Using elevated thresholds for very sparse dataset")
    
    print(f"Using similarity thresholds: general={base_threshold:.4f}, relevant={relevant_threshold:.4f}")
    
    # Process in batches with enhanced edge weighting
    _create_enhanced_semantic_edges_in_batches(
        G, tfidf_matrix, doc_ids, relevant_indices, 
        base_threshold, relevant_threshold, batch_size
    )


def _create_enhanced_semantic_edges_in_batches(G: nx.DiGraph, tfidf_matrix, doc_ids: List[str], 
                                    relevant_indices: List[int], similarity_threshold: float,
                                    relevant_threshold: float, batch_size: int = 200) -> None:
    """Create semantic edges with enhanced weighting for outlier detection."""
    total_docs = len(doc_ids)
    semantic_edges = 0
    
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        batch_indices = list(range(batch_start, batch_end))
        
        # Calculate similarities for this batch
        batch_tfidf = tfidf_matrix[batch_indices]
        similarities = cosine_similarity(batch_tfidf, tfidf_matrix)
        
        for i, global_i in enumerate(batch_indices):
            doc_i = doc_ids[global_i]
            is_i_relevant = global_i in relevant_indices
            
            for global_j in range(total_docs):
                if global_i >= global_j:  # Avoid duplicates and self-loops
                    continue
                
                doc_j = doc_ids[global_j]
                is_j_relevant = global_j in relevant_indices
                similarity = similarities[i, global_j]
                
                # Dynamic threshold based on document relevance
                if is_i_relevant or is_j_relevant:
                    threshold = relevant_threshold
                    edge_weight = 1.5  # Higher weight for relevant connections
                else:
                    threshold = similarity_threshold
                    edge_weight = 1.0
                
                if similarity >= threshold:
                    # Enhanced edge weighting based on similarity strength and relevance
                    if similarity >= 0.3:
                        weight_multiplier = 2.0  # Very high similarity
                    elif similarity >= 0.25:
                        weight_multiplier = 1.8  # High similarity
                    elif similarity >= 0.2:
                        weight_multiplier = 1.5  # Good similarity
                    else:
                        weight_multiplier = 1.2  # Moderate similarity
                    
                    final_weight = edge_weight * weight_multiplier * similarity
                    
                    # Add undirected semantic edge (both directions)
                    G.add_edge(doc_i, doc_j, 
                             edge_type='semantic', 
                             weight=final_weight,
                             similarity=similarity,
                             relevance_pattern=is_i_relevant or is_j_relevant)
                    
                    G.add_edge(doc_j, doc_i, 
                             edge_type='semantic', 
                             weight=final_weight,
                             similarity=similarity,
                             relevance_pattern=is_i_relevant or is_j_relevant)
                    
                    semantic_edges += 2  # Count both directions
        
        if (batch_start // batch_size + 1) % 5 == 0:
            print(f"Processed batch {batch_start + 1}-{batch_end} of {total_docs}, semantic edges so far: {semantic_edges}")
    
    print(f"Created {semantic_edges} enhanced semantic similarity edges")


def add_cocitation_edges(G: nx.DiGraph, simulation_df: pd.DataFrame, synergy_data: Optional[Dict] = None) -> None:
    """Add co-citation edges between papers cited together."""
    print("Adding co-citation edges...")
    
    if synergy_data is None:
        print("Warning: No synergy data provided, cannot add co-citation edges")
        return
    
    # Create set of simulation IDs for efficient lookup
    simulation_ids = set(simulation_df['openalex_id'].tolist())
    
    # Build co-citation matrix efficiently
    cocitation_counts = defaultdict(lambda: defaultdict(int))
    total_papers = len(simulation_df)
    
    # Process papers in batches for progress reporting
    for idx, row in simulation_df.iterrows():
        citing_paper = row['openalex_id']
        
        # Get referenced works from synergy data
        if citing_paper in synergy_data:
            referenced_works = synergy_data[citing_paper].get('referenced_works', [])
            
            if referenced_works and isinstance(referenced_works, list):
                # Find cited papers that are in our simulation dataset
                cited_papers_in_dataset = [
                    ref_id for ref_id in referenced_works 
                    if ref_id in simulation_ids and ref_id in G.nodes
                ]
                
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


def add_bibliographic_coupling_edges(G: nx.DiGraph, simulation_df: pd.DataFrame, synergy_data: Optional[Dict] = None) -> None:
    """Add bibliographic coupling edges between papers that cite the same references."""
    print("Adding bibliographic coupling edges...")
    
    if synergy_data is None:
        print("Warning: No synergy data provided, cannot add bibliographic coupling edges")
        return
    
    # Create set of simulation IDs for efficient lookup
    simulation_ids = set(simulation_df['openalex_id'].tolist())
    
    # Build paper references efficiently
    paper_references = {}
    total_papers = len(simulation_df)
    
    for idx, row in simulation_df.iterrows():
        paper_id = row['openalex_id']
        
        # Get referenced works from synergy data
        if paper_id in synergy_data:
            referenced_works = synergy_data[paper_id].get('referenced_works', [])
            
            if referenced_works and isinstance(referenced_works, list):
                paper_references[paper_id] = set(referenced_works)
        
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