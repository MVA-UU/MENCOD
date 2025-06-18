"""
Citation Network Model for Hybrid Outlier Detection

This module provides citation-based features for identifying outlier documents
that are missed by content-based ranking methods.
"""

import pandas as pd
import numpy as np
import networkx as nx
import os
import sys
from typing import Dict, List, Tuple, Optional
from scipy.sparse import csr_matrix
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import pickle
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import gc

# Fix import for direct script execution
script_dir = os.path.dirname(os.path.abspath(__file__))
# First add the CITATION_NETWORK_ENRICHED directory (one level up)
citation_network_dir = os.path.dirname(script_dir)
if citation_network_dir not in sys.path:
    sys.path.insert(0, citation_network_dir)
# Then add the project root (two levels up) as fallback
project_root = os.path.dirname(citation_network_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import dataset utilities
from dataset_utils import (
    load_datasets_config,
    prompt_dataset_selection,
    load_dataset,
    load_external_data,
    identify_outlier_in_simulation,
    create_training_data,
    get_search_pool
)

# Import feature extraction modules
from CitationNetwork.citation_features import (
    get_connectivity_features,
    get_coupling_features, 
    get_neighborhood_features,
    get_advanced_features,
    get_temporal_features,
    get_efficiency_features,
    get_zero_features
)

# Import scoring modules
from CitationNetwork.citation_scoring import (
    calculate_isolation_deviation,
    calculate_coupling_deviation,
    calculate_neighborhood_deviation,
    calculate_advanced_score,
    calculate_temporal_score,
    calculate_efficiency_score,
    get_adaptive_weights
)

# Import network utilities
from CitationNetwork.network_utils import (
    build_network_from_simulation,
    calculate_distance_baselines
)

# Import ranking module
from CitationNetwork.citation_ranking import apply_sparse_dataset_ranking_adjustments

# Global shared memory references for worker processes
_shared_adjacency = None
_shared_adjacency_shape = None
_shared_node_data = None
_shared_relevant_docs = None

def _init_worker_shared_memory(adjacency_name, adjacency_shape, node_data_name, relevant_docs_name):
    """Initialize worker process with shared memory references."""
    global _shared_adjacency, _shared_adjacency_shape, _shared_node_data, _shared_relevant_docs
    
    # Attach to existing shared memory
    _shared_adjacency = shared_memory.SharedMemory(name=adjacency_name)
    _shared_adjacency_shape = adjacency_shape
    _shared_node_data = shared_memory.SharedMemory(name=node_data_name)
    _shared_relevant_docs = shared_memory.SharedMemory(name=relevant_docs_name)

def _extract_features_worker_shared(doc_batch):
    """Worker function using shared memory."""
    global _shared_adjacency, _shared_adjacency_shape, _shared_node_data, _shared_relevant_docs
    
    # Reconstruct data from shared memory
    adjacency_bytes = bytes(_shared_adjacency.buf)
    adjacency_data = json.loads(adjacency_bytes.decode('utf-8'))
    
    node_data_bytes = bytes(_shared_node_data.buf)
    node_data = json.loads(node_data_bytes.decode('utf-8'))
    
    relevant_docs_bytes = bytes(_shared_relevant_docs.buf)
    relevant_docs = set(json.loads(relevant_docs_bytes.decode('utf-8')))
    
    # Process documents in this batch
    features = []
    for doc_id in doc_batch:
        if doc_id not in node_data:
            features.append(get_zero_features(doc_id))
            continue
        
        # Calculate features using the shared data
        doc_features = {
            'openalex_id': doc_id,
            **_calculate_features_from_shared_data(doc_id, adjacency_data, relevant_docs)
        }
        features.append(doc_features)
    
    return features

def _calculate_features_from_shared_data(doc_id, adjacency_data, relevant_documents):
    """Calculate features using shared memory data."""
    if doc_id not in adjacency_data:
        return get_zero_features(doc_id)
    
    neighbors = adjacency_data[doc_id]
    total_degree = len(neighbors)
    
    if total_degree == 0:
        return get_zero_features(doc_id)
    
    # Edge type analysis
    citation_out = sum(1 for edge_data in neighbors.values() if edge_data.get('edge_type') == 'citation')
    semantic_edges = sum(1 for edge_data in neighbors.values() if edge_data.get('edge_type') == 'semantic')
    coupling_edges = sum(1 for edge_data in neighbors.values() if edge_data.get('edge_type') == 'coupling')
    cocitation_edges = sum(1 for edge_data in neighbors.values() if edge_data.get('edge_type') == 'cocitation')
    
    # Relevant connections
    relevant_connections = sum(1 for neighbor_id in neighbors.keys() if neighbor_id in relevant_documents)
    relevant_ratio = relevant_connections / max(1, total_degree)
    
    # Coupling and co-citation scores
    coupling_scores = [edge_data.get('coupling_count', 0) for edge_data in neighbors.values() 
                      if edge_data.get('edge_type') == 'coupling']
    coupling_score = sum(coupling_scores)
    
    cocitation_scores = [edge_data.get('cocitation_count', 0) for edge_data in neighbors.values()
                        if edge_data.get('edge_type') == 'cocitation']
    cocitation_score = sum(cocitation_scores)
    
    return {
        'total_degree': total_degree,
        'out_degree': total_degree,
        'in_degree': 0,
        'citation_out_degree': citation_out,
        'citation_in_degree': 0,
        'semantic_degree': semantic_edges,
        'cocitation_degree': cocitation_edges,
        'coupling_degree': coupling_edges,
        'relevant_connections': relevant_connections,
        'relevant_ratio': relevant_ratio,
        'coupling_score': coupling_score,
        'cocitation_score': cocitation_score,
        'neighborhood_enrichment_1hop': relevant_ratio,
        'neighborhood_enrichment_2hop': 0.0,
        'citation_diversity': 0.0,
        'relevant_betweenness': 0.0,
        'structural_anomaly': 0.0,
        'semantic_isolation': 0.0,
        'citation_velocity': 0.0,
        'age_normalized_impact': 0.0,
        'citation_burst_score': 0.0,
        'temporal_isolation': 0.0,
        'recent_citation_ratio': 0.0,
        'citation_acceleration': 0.0,
        'local_clustering': 0.0,
        'edge_type_diversity': len(set(edge_data.get('edge_type', 'unknown') for edge_data in neighbors.values())),
        'relevant_path_efficiency': relevant_ratio,
        'citation_authority': 0.0,
        'semantic_coherence': 0.0,
        'network_position_score': relevant_ratio
    }

def _create_shared_memory_data(G, relevant_documents, target_documents):
    """Create shared memory objects for multiprocessing."""
    print("Creating shared memory for multiprocessing...")
    
    # Create minimal subgraph
    nodes_needed = set(target_documents)
    nodes_needed.update(relevant_documents)
    
    # Add immediate neighbors of target documents only (not full 2-hop)
    print(f"Adding neighbors for {len(target_documents)} target documents...")
    for doc in target_documents:
        if doc in G.nodes:
            neighbors = list(G.neighbors(doc))[:50]  # Limit to 50 neighbors per doc
            nodes_needed.update(neighbors)
    
    print(f"Subgraph contains {len(nodes_needed)} nodes (reduced from {len(G.nodes)})")
    
    # Extract subgraph
    subgraph = G.subgraph(nodes_needed)
    
    # Create adjacency data (only essential information)
    adjacency_data = {}
    node_data = {}
    
    for node in subgraph.nodes(data=True):
        node_id, attrs = node
        node_data[node_id] = attrs
        adjacency_data[node_id] = {}
        
        # Store only essential edge data to minimize memory
        for neighbor in subgraph.neighbors(node_id):
            if subgraph.has_edge(node_id, neighbor):
                edge_data = subgraph[node_id][neighbor]
                # Store only essential fields
                essential_edge_data = {
                    'edge_type': edge_data.get('edge_type', 'unknown'),
                    'weight': edge_data.get('weight', 1.0),
                    'coupling_count': edge_data.get('coupling_count', 0),
                    'cocitation_count': edge_data.get('cocitation_count', 0)
                }
                adjacency_data[node_id][neighbor] = essential_edge_data
    
    # Convert to JSON for shared memory
    adjacency_json = json.dumps(adjacency_data).encode('utf-8')
    node_data_json = json.dumps(node_data).encode('utf-8')
    relevant_docs_json = json.dumps(list(relevant_documents)).encode('utf-8')
    
    # Create shared memory objects
    print(f"Creating shared memory: adjacency={len(adjacency_json)} bytes, nodes={len(node_data_json)} bytes")
    
    shm_adjacency = shared_memory.SharedMemory(create=True, size=len(adjacency_json))
    shm_adjacency.buf[:len(adjacency_json)] = adjacency_json
    
    shm_node_data = shared_memory.SharedMemory(create=True, size=len(node_data_json))
    shm_node_data.buf[:len(node_data_json)] = node_data_json
    
    shm_relevant_docs = shared_memory.SharedMemory(create=True, size=len(relevant_docs_json))
    shm_relevant_docs.buf[:len(relevant_docs_json)] = relevant_docs_json
    
    return shm_adjacency, shm_node_data, shm_relevant_docs

def _prepare_graph_for_multiprocessing(G, relevant_documents, target_documents=None):
    """Convert NetworkX graph to multiprocessing-friendly format."""
    print("Preparing graph data for multiprocessing...")
    
    # If target_documents provided, only extract subgraph for those + relevant docs + their neighbors
    if target_documents:
        print(f"Extracting subgraph for {len(target_documents)} target documents...")
        
        # Get all nodes we need: target docs + relevant docs + their immediate neighbors
        nodes_needed = set(target_documents)
        nodes_needed.update(relevant_documents)
        
        # Add immediate neighbors of target and relevant documents
        for doc in list(nodes_needed):
            if doc in G.nodes:
                nodes_needed.update(G.neighbors(doc))
        
        print(f"Subgraph contains {len(nodes_needed)} nodes (reduced from {len(G.nodes)})")
        
        # Extract subgraph
        subgraph = G.subgraph(nodes_needed)
        
        # Convert subgraph to adjacency dictionary
        adjacency = {}
        node_attributes = {}
        
        for node in subgraph.nodes(data=True):
            node_id, attrs = node
            node_attributes[node_id] = attrs
            adjacency[node_id] = {}
            
            # Store outgoing edges with their attributes
            for neighbor in subgraph.neighbors(node_id):
                if subgraph.has_edge(node_id, neighbor):
                    edge_data = subgraph[node_id][neighbor]
                    adjacency[node_id][neighbor] = edge_data
        
        print(f"Subgraph converted to adjacency format: {len(adjacency)} nodes, {sum(len(neighbors) for neighbors in adjacency.values())} edges")
        
    else:
        # Fallback: convert entire graph (slow)
        print("Converting entire graph (this may take a while)...")
        adjacency = {}
        node_attributes = {}
        
        for node in G.nodes(data=True):
            node_id, attrs = node
            node_attributes[node_id] = attrs
            adjacency[node_id] = {}
            
            # Store outgoing edges with their attributes
            for neighbor in G.neighbors(node_id):
                if G.has_edge(node_id, neighbor):
                    edge_data = G[node_id][neighbor]
                    adjacency[node_id][neighbor] = edge_data
        
        print(f"Full graph converted to adjacency format: {len(adjacency)} nodes")
    
    return adjacency, node_attributes

def _extract_features_batch(args):
    """Extract features for a batch of documents."""
    doc_ids, G, relevant_documents, simulation_data = args
    features = []
    
    for doc_id in doc_ids:
        if doc_id not in G.nodes:
            features.append(get_zero_features(doc_id))
            continue
        
        doc_features = {
            'openalex_id': doc_id,
            **get_connectivity_features(G, doc_id, relevant_documents),
            **get_coupling_features(G, doc_id, relevant_documents),
            **get_neighborhood_features(G, doc_id, relevant_documents),
            **get_advanced_features(G, doc_id, relevant_documents),
            **get_temporal_features(G, doc_id, relevant_documents, simulation_data),
            **get_efficiency_features(G, doc_id, relevant_documents)
        }
        features.append(doc_features)
    
    return features


def _calculate_scores_batch(args):
    """Calculate relevance scores for a batch of documents."""
    doc_ids, G, relevant_documents, baseline_stats = args
    scores = {}
    
    for doc_id in doc_ids:
        if doc_id not in G.nodes:
            scores[doc_id] = 0.0
            continue
        
        # Extract features for this document
        doc_features = {
            'openalex_id': doc_id,
            **get_connectivity_features(G, doc_id, relevant_documents),
            **get_coupling_features(G, doc_id, relevant_documents),
            **get_neighborhood_features(G, doc_id, relevant_documents),
            **get_advanced_features(G, doc_id, relevant_documents),
            **get_efficiency_features(G, doc_id, relevant_documents)
        }
        
        # Calculate individual scores
        iso_score = calculate_isolation_deviation(pd.Series(doc_features), baseline_stats, G, relevant_documents)
        coupling_score = calculate_coupling_deviation(pd.Series(doc_features), baseline_stats)
        neighborhood_score = calculate_neighborhood_deviation(pd.Series(doc_features), baseline_stats)
        advanced_score = calculate_advanced_score(pd.Series(doc_features), len(relevant_documents) / len(G.nodes))
        efficiency_score = calculate_efficiency_score(pd.Series(doc_features), baseline_stats)
        
        scores[doc_id] = {
            'isolation': iso_score,
            'coupling': coupling_score,
            'neighborhood': neighborhood_score,
            'advanced': advanced_score,
            'efficiency': efficiency_score
        }
    
    return scores


class CitationNetworkModel:
    """Citation-based feature extractor for outlier detection."""
    
    def __init__(self, dataset_name: Optional[str] = None, n_cores: Optional[int] = None):
        """
        Initialize the Citation Network model.
        
        Args:
            dataset_name: Optional name of dataset to use. If None, will prompt user.
            n_cores: Number of CPU cores to use for parallel processing. If None, will use all available cores.
        """
        # If dataset_name is not provided, prompt user to select one
        if dataset_name is None:
            self.dataset_name = prompt_dataset_selection()
        else:
            self.dataset_name = dataset_name
            
        # Set number of cores for parallel processing
        if n_cores is None:
            self.n_cores = mp.cpu_count()
        else:
            self.n_cores = min(n_cores, mp.cpu_count())
            
        print(f"Using dataset: {self.dataset_name}")
        print(f"Using {self.n_cores} CPU cores for parallel processing")
        
        # Load dataset configuration
        self.datasets_config = load_datasets_config()
        if self.dataset_name not in self.datasets_config:
            raise ValueError(f"Dataset '{self.dataset_name}' not found in configuration")
        
        self.dataset_config = self.datasets_config[self.dataset_name]
        
        self.G = None
        self.relevant_documents = set()
        self.baseline_stats = None
        self.is_fitted = False
        self.simulation_data = None
    
    def fit(self, simulation_df: Optional[pd.DataFrame] = None, include_external: bool = True) -> 'CitationNetworkModel':
        """
        Build citation network and identify relevant documents.
        
        Args:
            simulation_df: Optional DataFrame with simulation results.
                           If None, will load from dataset configuration.
            include_external: Whether to include external data for network enrichment
        
        Returns:
            self: Returns the fitted model
        """
        print("Building comprehensive citation network...")
        
        # Load simulation data if not provided
        if simulation_df is None:
            simulation_df, _, external_df = load_dataset(self.dataset_name, include_external=include_external)
        else:
            # If simulation_df is provided but we want external data, load it separately
            external_df = None
            if include_external:
                external_df = load_external_data(self.dataset_name)
        
        # Store the simulation data for later use
        self.simulation_data = simulation_df.copy()
        
        # Combine simulation data with external data for network building
        network_data = simulation_df.copy()
        if external_df is not None and len(external_df) > 0:
            try:
                # Make a copy to avoid modifying the original external_df
                external_clean = external_df.copy()
                
                # Ensure external data has the same columns as simulation data
                # Add missing columns with default values
                for col in simulation_df.columns:
                    if col not in external_clean.columns:
                        if col == 'label_included':
                            external_clean[col] = 0  # External docs are never relevant
                        elif col == 'is_external':
                            external_clean[col] = True
                        else:
                            external_clean[col] = None
                
                # Convert data types to match simulation_df where possible
                for col in simulation_df.columns:
                    if col in external_clean.columns:
                        try:
                            # Try to match the dtype from simulation_df
                            if simulation_df[col].dtype != external_clean[col].dtype:
                                external_clean[col] = external_clean[col].astype(simulation_df[col].dtype)
                        except (TypeError, ValueError):
                            # If type conversion fails, keep as is
                            print(f"Warning: Could not convert column '{col}' to match simulation data type")
                            pass
                
                # Remove any columns that are in external but not in simulation
                # BUT keep important citation columns for network building
                important_citation_columns = {'referenced_works', 'referenced_works_count', 'cited_by_count'}
                extra_columns = set(external_clean.columns) - set(simulation_df.columns) - important_citation_columns
                if extra_columns:
                    print(f"Removing extra columns from external data: {extra_columns}")
                    external_clean = external_clean.drop(columns=list(extra_columns))
                
                # Add citation columns to simulation_df if they don't exist
                for citation_col in important_citation_columns:
                    if citation_col in external_clean.columns and citation_col not in simulation_df.columns:
                        simulation_df[citation_col] = None  # Add as None for simulation data
                
                # Reorder columns to match simulation_df (now includes citation columns)
                common_columns = [col for col in simulation_df.columns if col in external_clean.columns]
                external_clean = external_clean[common_columns]
                
                # Combine datasets
                print(f"Combining {len(simulation_df)} simulation documents with {len(external_clean)} external documents")
                network_data = pd.concat([simulation_df, external_clean], ignore_index=True)
                network_data = network_data.drop_duplicates(subset=['openalex_id'])
                print(f"Total network data: {len(network_data)} unique documents")
                
                # Verify citation data is available
                if 'referenced_works' in network_data.columns:
                    docs_with_refs = network_data['referenced_works'].notna().sum()
                    print(f"Citation data available: {docs_with_refs} documents have reference information")
                
            except Exception as e:
                print(f"Error combining external data: {e}")
                print("Proceeding with simulation data only")
                network_data = simulation_df.copy()
        
        # Create a comprehensive network from the combined data
        # This now includes citation, semantic, co-citation, and bibliographic coupling edges
        self.G = build_network_from_simulation(network_data, self.dataset_name, self.n_cores)
        
        # Identify relevant documents (only from original simulation data, not external)
        self.relevant_documents = set([
            row['openalex_id'] for _, row in simulation_df.iterrows() 
            if row['label_included'] == 1 and row['openalex_id'] in self.G.nodes
        ])
        
        self.is_fitted = True
        self.baseline_stats = self._calculate_baseline_stats()
        
        # Print network statistics by edge type
        edge_types = defaultdict(int)
        total_edges = 0
        for u, v, data in self.G.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] += 1
            total_edges += 1
        
        # Count original vs external nodes
        original_nodes = len([n for n in self.G.nodes if not self.G.nodes[n].get('is_external', False)])
        external_nodes = len([n for n in self.G.nodes if self.G.nodes[n].get('is_external', False)])
        
        print(f"Citation network built: {len(self.G.nodes)} nodes ({original_nodes} original, {external_nodes} external), {total_edges} edges")
        print("Edge distribution by type:")
        for edge_type, count in sorted(edge_types.items()):
            print(f"  {edge_type}: {count} edges")
        print(f"Relevant documents identified: {len(self.relevant_documents)} (from original dataset only)")
        print(f"Baseline citation patterns calculated from {len(self.relevant_documents)} relevant docs")
        
        return self
    
    def get_citation_features(self, target_documents: List[str]) -> pd.DataFrame:
        """Extract citation-based features for target documents using multiprocessing."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting features")
        
        # For small numbers of documents, use single-threaded processing
        if len(target_documents) < 100 or self.n_cores == 1:
            features = []
            for doc_id in target_documents:
                if doc_id not in self.G.nodes:
                    features.append(get_zero_features(doc_id))
                    continue
                
                doc_features = {
                    'openalex_id': doc_id,
                    **get_connectivity_features(self.G, doc_id, self.relevant_documents),
                    **get_coupling_features(self.G, doc_id, self.relevant_documents),
                    **get_neighborhood_features(self.G, doc_id, self.relevant_documents),
                    **get_advanced_features(self.G, doc_id, self.relevant_documents),
                    **get_temporal_features(self.G, doc_id, self.relevant_documents, self.simulation_data),
                    **get_efficiency_features(self.G, doc_id, self.relevant_documents)
                }
                features.append(doc_features)
        else:
            # Use optimized shared memory multiprocessing
            print(f"Extracting features for {len(target_documents)} documents using {self.n_cores} cores (shared memory)")
            
            # Create shared memory objects
            shm_adjacency, shm_node_data, shm_relevant_docs = _create_shared_memory_data(self.G, self.relevant_documents, target_documents)
            
            # Split documents into batches
            batch_size = max(50, len(target_documents) // self.n_cores)
            batches = [target_documents[i:i + batch_size] for i in range(0, len(target_documents), batch_size)]
            
            print(f"Processing {len(batches)} batches with {batch_size} documents each")
            
            features = []
            try:
                # Use ProcessPoolExecutor with shared memory initializer
                with ProcessPoolExecutor(
                    max_workers=self.n_cores,
                    initializer=_init_worker_shared_memory,
                    initargs=(shm_adjacency.name, None, shm_node_data.name, shm_relevant_docs.name)
                ) as executor:
                    # Submit all batches
                    future_to_batch = {executor.submit(_extract_features_worker_shared, batch): i for i, batch in enumerate(batches)}
                    
                    # Collect results as they complete
                    completed = 0
                    for future in as_completed(future_to_batch):
                        batch_idx = future_to_batch[future]
                        try:
                            batch_features = future.result()
                            features.extend(batch_features)
                            completed += 1
                            progress = (completed / len(batches)) * 100
                            print(f"  Completed batch {completed}/{len(batches)} ({progress:.1f}%)")
                        except Exception as exc:
                            print(f"Batch {batch_idx} generated an exception: {exc}")
                            # Fallback for failed batch
                            failed_batch = batches[batch_idx]
                            for doc_id in failed_batch:
                                features.append(get_zero_features(doc_id))
                
                print(f"Shared memory multiprocessing completed: {len(features)} documents processed")
                
            except Exception as e:
                print(f"Multiprocessing failed: {e}, falling back to single-threaded processing")
                # Fallback to single-threaded
                features = []
                for i, doc_id in enumerate(target_documents):
                    if doc_id not in self.G.nodes:
                        features.append(get_zero_features(doc_id))
                        continue
                    
                    doc_features = {
                        'openalex_id': doc_id,
                        **get_connectivity_features(self.G, doc_id, self.relevant_documents),
                        **get_coupling_features(self.G, doc_id, self.relevant_documents),
                        **get_neighborhood_features(self.G, doc_id, self.relevant_documents),
                        **get_advanced_features(self.G, doc_id, self.relevant_documents),
                        **get_temporal_features(self.G, doc_id, self.relevant_documents, self.simulation_data),
                        **get_efficiency_features(self.G, doc_id, self.relevant_documents)
                    }
                    features.append(doc_features)
                    
                    if (i + 1) % 100 == 0:
                        progress = ((i + 1) / len(target_documents)) * 100
                        print(f"  Fallback progress: {i + 1}/{len(target_documents)} ({progress:.1f}%)")
            
            finally:
                # Clean up shared memory
                try:
                    shm_adjacency.close()
                    shm_adjacency.unlink()
                    shm_node_data.close()
                    shm_node_data.unlink()
                    shm_relevant_docs.close()
                    shm_relevant_docs.unlink()
                    print("Shared memory cleaned up")
                except:
                    pass
        
        return pd.DataFrame(features)
    
    def predict_relevance_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """Generate citation-based outlier scores using relative deviation from baseline."""
        if not self.is_fitted or not self.baseline_stats:
            return {doc_id: 0.0 for doc_id in target_documents}
        
        search_pool_size = len(target_documents)
        
        # Calculate dataset sparsity measures
        relevant_ratio = len(self.relevant_documents) / len(self.G.nodes) if self.G.nodes else 0
        sparsity_factor = 1 - min(0.9, max(0.1, relevant_ratio * 10))  # Higher for sparser datasets
        
        print(f"Dataset relevant ratio: {relevant_ratio:.4f}, sparsity factor: {sparsity_factor:.4f}")
        
        # Get adaptive weights based on dataset characteristics
        weights = get_adaptive_weights(sparsity_factor)
        print(f"Using adaptive feature weights based on dataset sparsity: {weights}")
        
        # Scaling factors for score adjustment - adapt continuously based on sparsity
        coupling_scaling = 1.0 + sparsity_factor * 0.5  # More scaling for sparser datasets
        isolation_scaling = 1.0 - sparsity_factor * 0.2
        
        # Use optimized multiprocessing for all large datasets
        if search_pool_size > 500 and self.n_cores > 1:
            print(f"Processing {search_pool_size} documents using {self.n_cores} cores (optimized scoring)...")
            
            # Process in larger batches for scoring (less overhead than feature extraction)
            batch_size = max(200, search_pool_size // self.n_cores)
            batches = [target_documents[i:i + batch_size] for i in range(0, len(target_documents), batch_size)]
            
            print(f"Processing {len(batches)} batches with ~{batch_size} documents each")
            
            # Create shared memory objects for scoring
            shm_adjacency, shm_node_data, shm_relevant_docs = _create_shared_memory_data(self.G, self.relevant_documents, target_documents)
            
            all_scores = {'isolation': [], 'coupling': [], 'neighborhood': [], 'advanced': [], 'temporal': [], 'efficiency': []}
            raw_scores = {}
            feature_cache = {}
            
            try:
                with ProcessPoolExecutor(
                    max_workers=self.n_cores,
                    initializer=_init_worker_shared_memory,
                    initargs=(shm_adjacency.name, None, shm_node_data.name, shm_relevant_docs.name)
                ) as executor:
                    # Submit all batches for feature extraction
                    future_to_batch = {executor.submit(_extract_features_worker_shared, batch): i for i, batch in enumerate(batches)}
                    
                    # Process results as they complete
                    completed = 0
                    for future in as_completed(future_to_batch):
                        batch_idx = future_to_batch[future]
                        try:
                            batch_features = future.result()
                            
                            # Calculate scores for this batch
                            for doc_features in batch_features:
                                doc_id = doc_features['openalex_id']
                                feature_cache[doc_id] = doc_features
                                
                                # Calculate component scores
                                iso_score = calculate_isolation_deviation(doc_features, self.baseline_stats, self.G, self.relevant_documents)
                                coup_score = calculate_coupling_deviation(doc_features, self.baseline_stats)
                                neigh_score = calculate_neighborhood_deviation(doc_features, self.baseline_stats)
                                adv_score = calculate_advanced_score(doc_features, relevant_ratio)
                                temp_score = calculate_temporal_score(doc_features, self.baseline_stats)
                                eff_score = calculate_efficiency_score(doc_features, self.baseline_stats)
                                
                                # Apply scaling factors
                                coup_score = min(1.0, coup_score * coupling_scaling)
                                iso_score = min(1.0, iso_score * isolation_scaling)
                                
                                # Data-driven score adjustments based on sparsity
                                if sparsity_factor > 0.7:  # Very sparse dataset
                                    if doc_features.get('relevant_connections', 0) > 0:
                                        mean_ratio = self.baseline_stats.get('mean_relevant_ratio', 0.2)
                                        if mean_ratio > 0:
                                            ratio_factor = min(1.5, doc_features.get('relevant_ratio', 0) / mean_ratio)
                                            coup_score = min(0.95, coup_score * ratio_factor)
                                
                                # Store component scores
                                all_scores['isolation'].append(iso_score)
                                all_scores['coupling'].append(coup_score)
                                all_scores['neighborhood'].append(neigh_score)
                                all_scores['advanced'].append(adv_score)
                                all_scores['temporal'].append(temp_score)
                                all_scores['efficiency'].append(eff_score)
                                
                                # Store raw combined score
                                raw_scores[doc_id] = (
                                    weights['isolation'] * iso_score + 
                                    weights['coupling'] * coup_score + 
                                    weights['neighborhood'] * neigh_score +
                                    weights['temporal'] * temp_score +
                                    weights['efficiency'] * eff_score
                                )
                            
                            completed += 1
                            progress = (completed / len(batches)) * 100
                            print(f"  Completed scoring batch {completed}/{len(batches)} ({progress:.1f}%)")
                            
                        except Exception as exc:
                            print(f"Scoring batch {batch_idx} generated an exception: {exc}")
                            # Add zero scores for failed batch
                            failed_batch = batches[batch_idx]
                            for doc_id in failed_batch:
                                raw_scores[doc_id] = 0.0
                                feature_cache[doc_id] = get_zero_features(doc_id)
                
                print(f"Shared memory scoring completed: {len(raw_scores)} documents processed")
                
            except Exception as e:
                print(f"Multiprocessing scoring failed: {e}, falling back to single-threaded")
                # Fallback to single-threaded processing
                return self._process_single_threaded(target_documents, weights, coupling_scaling, isolation_scaling, sparsity_factor, relevant_ratio)
            
            finally:
                # Clean up shared memory
                try:
                    shm_adjacency.close()
                    shm_adjacency.unlink()
                    shm_node_data.close()
                    shm_node_data.unlink()
                    shm_relevant_docs.close()
                    shm_relevant_docs.unlink()
                    print("Shared memory cleaned up")
                except:
                    pass
        
        else:
            # Single-threaded processing for smaller datasets
            print(f"Processing {search_pool_size} documents in single-threaded mode...")
            return self._process_single_threaded(target_documents, weights, coupling_scaling, isolation_scaling, sparsity_factor, relevant_ratio)
        
        # Normalize and post-process scores
        scores = self._normalize_scores(raw_scores, all_scores, feature_cache, sparsity_factor)
        return scores
    
    def _process_single_threaded(self, target_documents, weights, coupling_scaling, isolation_scaling, sparsity_factor, relevant_ratio):
        """Single-threaded processing fallback."""
        batch_size = 200
        all_scores = {'isolation': [], 'coupling': [], 'neighborhood': [], 'advanced': [], 'temporal': [], 'efficiency': []}
        raw_scores = {}
        feature_cache = {}
        
        # Process in batches
        for batch_start in range(0, len(target_documents), batch_size):
            batch_end = min(batch_start + batch_size, len(target_documents))
            batch_docs = target_documents[batch_start:batch_end]
            
            # Get features for this batch
            features_df = self.get_citation_features(batch_docs)
            
            # Calculate scores for each document in batch
            for _, row in features_df.iterrows():
                doc_id = row['openalex_id']
                feature_cache[doc_id] = row.to_dict()
                
                # Calculate component scores
                iso_score = calculate_isolation_deviation(row, self.baseline_stats, self.G, self.relevant_documents)
                coup_score = calculate_coupling_deviation(row, self.baseline_stats)
                neigh_score = calculate_neighborhood_deviation(row, self.baseline_stats)
                adv_score = calculate_advanced_score(row, relevant_ratio)
                temp_score = calculate_temporal_score(row, self.baseline_stats)
                eff_score = calculate_efficiency_score(row, self.baseline_stats)
                
                # Apply scaling factors
                coup_score = min(1.0, coup_score * coupling_scaling)
                iso_score = min(1.0, iso_score * isolation_scaling)
                
                # Data-driven score adjustments
                if sparsity_factor > 0.7:
                    if row['relevant_connections'] > 0:
                        mean_ratio = self.baseline_stats.get('mean_relevant_ratio', 0.2)
                        if mean_ratio > 0:
                            ratio_factor = min(1.5, row.get('relevant_ratio', 0) / mean_ratio)
                            coup_score = min(0.95, coup_score * ratio_factor)
                
                # Store component scores
                all_scores['isolation'].append(iso_score)
                all_scores['coupling'].append(coup_score)
                all_scores['neighborhood'].append(neigh_score)
                all_scores['advanced'].append(adv_score)
                all_scores['temporal'].append(temp_score)
                all_scores['efficiency'].append(eff_score)
                
                # Store raw combined score
                raw_scores[doc_id] = (
                    weights['isolation'] * iso_score + 
                    weights['coupling'] * coup_score + 
                    weights['neighborhood'] * neigh_score +
                    weights['temporal'] * temp_score +
                    weights['efficiency'] * eff_score
                )
            
            # Report progress
            if len(target_documents) > 500:
                print(f"Processed batch {batch_start+1}-{batch_end} of {len(target_documents)}")
        
        # Normalize and post-process scores
        scores = self._normalize_scores(raw_scores, all_scores, feature_cache, sparsity_factor)
        return scores
    
    def _normalize_scores(self, raw_scores, all_scores, feature_cache, sparsity_factor):
        """Normalize and post-process scores based on dataset characteristics."""
        # Calculate statistics for score normalization
        score_stats = {}
        for component, values in all_scores.items():
            if values:
                score_stats[component] = {
                    'mean': np.mean(values),
                    'std': max(np.std(values), 0.01),
                    'max': max(values),
                    'min': min(values)
                }
        
        # Second pass - normalize and adjust scores
        scores = {}
        for doc_id, raw_score in raw_scores.items():
            # Use data-driven adjustment based on sparsity factor
            if sparsity_factor > 0.8:  # Extremely sparse
                adj_score = raw_score  # Will be replaced by rank-based normalization
            elif sparsity_factor > 0.6:  # Very sparse
                adj_score = min(1.0, raw_score ** (0.8 + sparsity_factor * 0.2))
            elif sparsity_factor > 0.4:  # Moderately sparse
                contrast_factor = 1.0 + (sparsity_factor - 0.4) * 0.5
                adj_score = min(1.0, raw_score ** contrast_factor)
            else:  # Normal density
                adj_score = raw_score
                
            scores[doc_id] = min(1.0, max(0.0, adj_score))
        
        # For extremely sparse datasets, apply rank-based normalization
        # DISABLED: This was making results worse for academic outlier detection
        # if sparsity_factor > 0.8:
        #     scores = apply_sparse_dataset_ranking_adjustments(
        #         scores, feature_cache, self.relevant_documents, sparsity_factor, self.baseline_stats
        #     )
        
        return scores
    
    def analyze_outlier(self, outlier_id: str) -> Dict[str, float]:
        """
        Analyze citation network features of a specific outlier.
        
        Args:
            outlier_id: OpenAlex ID of the outlier to analyze
        
        Returns:
            Dictionary with detailed citation network analysis
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analyzing outliers")
            
        features_df = self.get_citation_features([outlier_id])
        if features_df.empty:
            return {}
            
        features = features_df.iloc[0]
        
        # OPTIMIZED: Calculate distances using efficient methods
        dist = float('inf')
        
        # First check direct connections (distance = 1)
        if outlier_id in self.G.nodes:
            outlier_neighbors = set(self.G.neighbors(outlier_id))
            direct_relevant_connections = outlier_neighbors.intersection(self.relevant_documents)
            
            if direct_relevant_connections:
                dist = 1  # Direct connection found
            else:
                # Check 2-hop connections (distance = 2) 
                two_hop_relevant = set()
                for neighbor in outlier_neighbors:
                    if neighbor in self.G.nodes:
                        neighbor_neighbors = set(self.G.neighbors(neighbor))
                        two_hop_relevant.update(neighbor_neighbors.intersection(self.relevant_documents))
                
                if two_hop_relevant:
                    dist = 2  # 2-hop connection found
                else:
                    # Only do expensive shortest path for a sample of relevant docs
                    sample_size = min(5, len(self.relevant_documents))  # Limit expensive calculations
                    import random
                    relevant_sample = random.sample(list(self.relevant_documents), sample_size)
                    
                    for rel_doc in relevant_sample:
                        if rel_doc in self.G.nodes:
                            try:
                                d = nx.shortest_path_length(self.G, outlier_id, rel_doc)
                                dist = min(dist, d)
                                if dist <= 3:  # Stop early if we find a reasonably close connection
                                    break
                            except nx.NetworkXNoPath:
                                continue
        
        # Get relevance scores (higher = more relevant)
        iso_score = calculate_isolation_deviation(features, self.baseline_stats, self.G, self.relevant_documents)
        coup_score = calculate_coupling_deviation(features, self.baseline_stats)
        neigh_score = calculate_neighborhood_deviation(features, self.baseline_stats)
        
        # Get the dataset sparsity ratio
        relevant_ratio = len(self.relevant_documents) / len(self.G.nodes) if self.G.nodes else 0
        adv_score = calculate_advanced_score(features, relevant_ratio)
        
        # Create analysis dictionary with safe feature access
        analysis = {
            'distance_to_relevant': dist,
            'total_citations': features.get('citation_in_degree', 0),
            'relevant_connections': features.get('relevant_connections', 0),
            'relevant_connections_ratio': features.get('relevant_ratio', 0),
            'max_coupling_strength': features.get('coupling_score', 0),
            'neighborhood_enrichment_1hop': features.get('neighborhood_enrichment_1hop', 0),
            'neighborhood_enrichment_2hop': features.get('neighborhood_enrichment_2hop', 0),
            'citation_diversity': features.get('coupling_diversity', 0),
            'relevant_betweenness': features.get('relevant_betweenness', 0),
            'structural_anomaly': features.get('structural_anomaly', 0),
            'semantic_isolation': features.get('semantic_isolation', 0.0),
            'weighted_influence': features.get('weighted_influence', 0),
            'weighted_connections': features.get('weighted_connections', 0),
            'cocitation_score': features.get('cocitation_score', 0),
            'dataset_sparsity_ratio': relevant_ratio,
            'isolation_score': iso_score,
            'coupling_score': coup_score,
            'neighborhood_score': neigh_score,
            'advanced_score': adv_score
        }
        
        # Calculate combined score based on dataset characteristics (DISABLED advanced_score)
        weights = get_adaptive_weights(1 - min(0.9, max(0.1, relevant_ratio * 10)))
        combined_score = (
            weights['isolation'] * iso_score + 
            weights['coupling'] * coup_score + 
            weights['neighborhood'] * neigh_score
            # + weights['advanced'] * adv_score  # DISABLED
        )
            
        analysis['combined_score'] = combined_score
        
        return analysis
    
    def _calculate_baseline_stats(self) -> Dict[str, float]:
        """Calculate enhanced baseline citation statistics from relevant documents for sparse networks."""
        if not self.relevant_documents:
            return {}
        
        # Extract features from relevant documents
        rel_features = self.get_citation_features(list(self.relevant_documents))
        if rel_features.empty:
            return {}
        
        baseline = {}
        
        # Basic citation statistics with proper handling of sparse data
        citation_cols = ['citation_in_degree', 'citation_out_degree', 'total_citations']
        for col in citation_cols:
            if col in rel_features.columns:
                values = rel_features[col].dropna()
                if not values.empty:
                    baseline[f'mean_{col}'] = values.mean()
                    baseline[f'std_{col}'] = max(values.std(), 0.1)  # Minimum std for stability
                    baseline[f'median_{col}'] = values.median()
                    baseline[f'q75_{col}'] = values.quantile(0.75)
                    baseline[f'q90_{col}'] = values.quantile(0.90)
        
        # Connection and relationship statistics
        connection_cols = ['relevant_connections', 'relevant_ratio', 'coupling_score', 'cocitation_score']
        for col in connection_cols:
            if col in rel_features.columns:
                values = rel_features[col].dropna()
                if not values.empty:
                    baseline[f'mean_{col}'] = values.mean()
                    baseline[f'std_{col}'] = max(values.std(), 0.01)
                    baseline[f'median_{col}'] = values.median()
        
        # Neighborhood enrichment statistics
        neighborhood_cols = ['neighborhood_enrichment_1hop', 'neighborhood_enrichment_2hop']
        for col in neighborhood_cols:
            if col in rel_features.columns:
                values = rel_features[col].dropna()
                if not values.empty:
                    baseline[f'mean_{col}'] = values.mean()
                    baseline[f'std_{col}'] = max(values.std(), 0.01)
                    baseline[f'max_{col}'] = values.max()
        
        # Advanced network statistics
        advanced_cols = ['weighted_influence', 'weighted_connections', 'semantic_coherence', 
                        'citation_authority', 'network_position_score']
        for col in advanced_cols:
            if col in rel_features.columns:
                values = rel_features[col].dropna()
                if not values.empty:
                    baseline[f'mean_{col}'] = values.mean()
                    baseline[f'std_{col}'] = max(values.std(), 0.01)
        
        # Temporal statistics
        temporal_cols = ['citation_velocity', 'age_normalized_impact', 'citation_burst_score']
        for col in temporal_cols:
            if col in rel_features.columns:
                values = rel_features[col].dropna()
                if not values.empty:
                    baseline[f'mean_{col}'] = values.mean()
                    baseline[f'std_{col}'] = max(values.std(), 0.01)
        
        # Calculate pairwise distances between relevant documents for network analysis
        # OPTIMIZED: Limit expensive shortest path calculations for sparse datasets
        distances = []
        centralities = []
        
        # Only calculate distances if we have a reasonable number of relevant docs
        if len(self.relevant_documents) <= 50:  # Limit to avoid O(n²) explosion
            print(f"Calculating baseline network statistics for {len(self.relevant_documents)} relevant documents...")
            
            # Create relevant subgraph once for efficiency
            relevant_subgraph = self.G.subgraph(self.relevant_documents)
            
            for doc1 in self.relevant_documents:
                if doc1 not in self.G.nodes:
                    continue
                    
                # Calculate centrality within relevant document subgraph
                if len(relevant_subgraph.nodes) > 1:
                    try:
                        degree_cent = relevant_subgraph.degree(doc1)
                        centralities.append(degree_cent)
                    except:
                        pass
                
                # OPTIMIZED: Only calculate distances to a sample of other relevant docs
                sample_size = min(10, len(self.relevant_documents) - 1)  # Max 10 distance calculations per doc
                other_docs = [d for d in self.relevant_documents if d != doc1 and d in self.G.nodes]
                
                if other_docs:
                    # Sample random subset to avoid O(n²) complexity
                    import random
                    sample_docs = random.sample(other_docs, min(sample_size, len(other_docs)))
                    
                    for doc2 in sample_docs:
                        try:
                            # Try efficient path in relevant subgraph first
                            if doc2 in relevant_subgraph.nodes:
                                try:
                                    dist = nx.shortest_path_length(relevant_subgraph, doc1, doc2)
                                except nx.NetworkXNoPath:
                                    # Fallback to full graph (more expensive)
                                    dist = nx.shortest_path_length(self.G, doc1, doc2)
                            else:
                                dist = nx.shortest_path_length(self.G, doc1, doc2)
                            distances.append(dist)
                        except nx.NetworkXNoPath:
                            distances.append(float('inf'))
        else:
            print(f"Skipping expensive distance calculations for {len(self.relevant_documents)} relevant documents (too many)")
            # For large numbers of relevant docs, use simpler connectivity measures
            relevant_subgraph = self.G.subgraph(self.relevant_documents)
            for doc1 in self.relevant_documents:
                if doc1 in relevant_subgraph.nodes:
                    degree_cent = relevant_subgraph.degree(doc1)
                    centralities.append(degree_cent)
        
        # Distance statistics
        if distances:
            finite_distances = [d for d in distances if d != float('inf')]
            if finite_distances:
                baseline['mean_relevant_distance'] = np.mean(finite_distances)
                baseline['std_relevant_distance'] = max(np.std(finite_distances), 0.1)
                baseline['median_relevant_distance'] = np.median(finite_distances)
                baseline['connectivity_ratio'] = len(finite_distances) / len(distances)
            else:
                baseline['mean_relevant_distance'] = float('inf')
                baseline['connectivity_ratio'] = 0.0
        
        # Centrality statistics
        if centralities:
            baseline['mean_relevant_centrality'] = np.mean(centralities)
            baseline['std_relevant_centrality'] = max(np.std(centralities), 0.1)
        
        # Network density and sparsity measures
        total_possible_edges = len(self.relevant_documents) * (len(self.relevant_documents) - 1)
        if total_possible_edges > 0:
            actual_edges = sum(1 for doc1 in self.relevant_documents 
                             for doc2 in self.relevant_documents 
                             if doc1 != doc2 and doc1 in self.G.nodes and doc2 in self.G.nodes 
                             and self.G.has_edge(doc1, doc2))
            baseline['relevant_network_density'] = actual_edges / total_possible_edges
        else:
            baseline['relevant_network_density'] = 0.0
        
        # Edge type distribution in relevant subgraph
        edge_types = defaultdict(int)
        for doc1 in self.relevant_documents:
            for doc2 in self.relevant_documents:
                if (doc1 != doc2 and doc1 in self.G.nodes and doc2 in self.G.nodes 
                    and self.G.has_edge(doc1, doc2)):
                    edge_data = self.G.get_edge_data(doc1, doc2)
                    if edge_data:
                        edge_type = edge_data.get('edge_type', 'unknown')
                        edge_types[edge_type] += 1
        
        total_relevant_edges = sum(edge_types.values())
        if total_relevant_edges > 0:
            for edge_type, count in edge_types.items():
                baseline[f'relevant_{edge_type}_ratio'] = count / total_relevant_edges
        
        return baseline


def main():
    """Test citation network model with a selected dataset and show outlier ranking."""
    
    # Create model (will prompt user to select dataset)
    model = CitationNetworkModel()
    
    # Load dataset
    simulation_df, dataset_config, _ = load_dataset(model.dataset_name)
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
    
    # Get outlier analysis
    analysis = model.analyze_outlier(outlier_id)
    print(f"\nOutlier citation features:")
    for k, v in analysis.items():
        print(f"  {k}: {v}")
    
    # Test outlier retrieval among irrelevant documents
    print(f"\n=== OUTLIER RETRIEVAL TEST ===")
    
    # Get search pool: outlier + all irrelevant documents
    search_pool = get_search_pool(simulation_df, outlier_id)
    print(f"Search pool size: {len(search_pool)} documents")
    
    # Score all documents in search pool
    print("Scoring documents in search pool...")
    scores = model.predict_relevance_scores(search_pool)
    print("Sorting results...")
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
    
    # Practical results
    top_50_ids = [doc_id for doc_id, _ in sorted_results[:50]]
    top_100_ids = [doc_id for doc_id, _ in sorted_results[:100]]
    
    found_in_top_50 = outlier_id in top_50_ids
    found_in_top_100 = outlier_id in top_100_ids
    
    print(f"\nPractical Results:")
    print(f"Found in top 50: {'YES' if found_in_top_50 else 'NO'}")
    print(f"Found in top 100: {'YES' if found_in_top_100 else 'NO'}")
    
    # Performance assessment
    if found_in_top_50:
        print("✅ EXCELLENT: Outlier found in top 50!")
    elif found_in_top_100:
        print("⚠️  GOOD: Outlier found in top 100")
    else:
        print("❌ NEEDS IMPROVEMENT: Outlier not found in top 100")
    
    return {
        'outlier_position': outlier_position,
        'outlier_score': outlier_score,
        'total_candidates': len(search_pool),
        'percentile': percentile,
        'found_in_top_50': found_in_top_50,
        'found_in_top_100': found_in_top_100
    }


if __name__ == "__main__":
    main() 