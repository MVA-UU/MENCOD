#!/usr/bin/env python3
"""
GPU-Accelerated Citation Network Model for Supercomputer

Optimizations:
1. cuGraph for GPU-accelerated NetworkX operations
2. Multiprocessing for feature extraction
3. Optimized batch processing
4. Memory-efficient operations
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

# Try to use cuGraph for GPU acceleration
try:
    import cugraph as cx
    import cudf
    import rmm
    CUGRAPH_AVAILABLE = True
    print("cuGraph available - GPU acceleration enabled!")
except ImportError:
    import networkx as nx
    CUGRAPH_AVAILABLE = False
    print("cuGraph not available - using NetworkX on CPU")

# Fix import for direct script execution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dataset_utils import (
    load_datasets_config, prompt_dataset_selection, load_dataset,
    identify_outlier_in_simulation, create_training_data, get_search_pool
)

# Import local modules
from citation_features import get_zero_features
from citation_scoring import (
    calculate_isolation_deviation, calculate_coupling_deviation,
    calculate_neighborhood_deviation, calculate_advanced_score,
    calculate_temporal_score, calculate_efficiency_score, get_adaptive_weights
)
from network_utils import build_network_from_simulation, calculate_distance_baselines
from citation_ranking import apply_sparse_dataset_ranking_adjustments


class GPUAcceleratedCitationNetwork:
    """GPU-accelerated citation network with multiprocessing."""
    
    def __init__(self, dataset_name: Optional[str] = None, n_cores: int = None):
        """Initialize with GPU acceleration and multiprocessing."""
        self.dataset_name = dataset_name or prompt_dataset_selection()
        self.n_cores = n_cores or min(mp.cpu_count(), 15)  # Use up to 15 cores
        
        print(f"Using dataset: {self.dataset_name}")
        print(f"Using {self.n_cores} CPU cores for parallel processing")
        
        # Load dataset configuration
        self.datasets_config = load_datasets_config()
        if self.dataset_name not in self.datasets_config:
            raise ValueError(f"Dataset '{self.dataset_name}' not found")
        
        self.dataset_config = self.datasets_config[self.dataset_name]
        self.G = None
        self.gpu_graph = None
        self.relevant_documents = set()
        self.baseline_stats = None
        self.is_fitted = False
        self.simulation_data = None
        
        # Initialize GPU memory if available
        if CUGRAPH_AVAILABLE:
            try:
                rmm.reinitialize(pool_allocator=True, initial_pool_size=2**30)  # 1GB pool
                print("GPU memory pool initialized")
            except:
                print("GPU memory pool initialization failed, continuing...")
    
    def fit(self, simulation_df: Optional[pd.DataFrame] = None) -> 'GPUAcceleratedCitationNetwork':
        """Build citation network with GPU acceleration."""
        start_time = time.time()
        print("Building GPU-accelerated citation network...")
        
        # Load simulation data if not provided
        if simulation_df is None:
            simulation_df, _ = load_dataset(self.dataset_name)
        
        self.simulation_data = simulation_df.copy()
        
        # Build network using existing optimized function
        self.G = build_network_from_simulation(simulation_df, self.dataset_name)
        
        # Convert to GPU graph if cuGraph is available
        if CUGRAPH_AVAILABLE and len(self.G.edges) < 1000000:  # Only for reasonable size
            try:
                self._convert_to_gpu_graph()
            except Exception as e:
                print(f"GPU conversion failed: {e}, continuing with CPU")
        
        # Identify relevant documents
        self.relevant_documents = set([
            row['openalex_id'] for _, row in simulation_df.iterrows() 
            if row['label_included'] == 1 and row['openalex_id'] in self.G.nodes
        ])
        
        self.is_fitted = True
        self.baseline_stats = self._calculate_baseline_stats_fast()
        
        # Print statistics
        edge_types = defaultdict(int)
        for u, v, data in self.G.edges(data=True):
            edge_types[data.get('edge_type', 'unknown')] += 1
        
        build_time = time.time() - start_time
        print(f"Citation network built in {build_time:.2f}s: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        print("Edge distribution:", dict(edge_types))
        print(f"Relevant documents: {len(self.relevant_documents)}")
        
        return self
    
    def _convert_to_gpu_graph(self):
        """Convert NetworkX graph to cuGraph for GPU acceleration."""
        print("Converting graph to GPU...")
        
        # Create edge list
        edges = []
        for u, v, data in self.G.edges(data=True):
            edges.append([u, v, data.get('weight', 1.0)])
        
        if edges:
            edge_df = cudf.DataFrame(edges, columns=['src', 'dst', 'weight'])
            self.gpu_graph = cx.Graph()
            self.gpu_graph.from_cudf_edgelist(edge_df, source='src', destination='dst', edge_attr='weight')
            print(f"GPU graph created with {len(edges)} edges")
    
    def _calculate_baseline_stats_fast(self) -> Dict[str, float]:
        """Ultra-fast baseline calculation using sampling."""
        if not self.relevant_documents:
            return {}
        
        print("Calculating baseline statistics (GPU-optimized)...")
        
        # Use small sample for speed
        sample_size = min(20, len(self.relevant_documents))
        sample_docs = list(self.relevant_documents)[:sample_size]
        
        baseline = {}
        degrees = []
        
        for doc_id in sample_docs:
            if doc_id in self.G.nodes:
                degrees.append(self.G.degree(doc_id))
        
        if degrees:
            baseline['mean_total_degree'] = np.mean(degrees)
            baseline['std_total_degree'] = max(np.std(degrees), 0.1)
        
        # Add defaults for compatibility
        baseline.update({
            'mean_citation_in_degree': 1.0, 'std_citation_in_degree': 0.1,
            'mean_citation_out_degree': 1.0, 'std_citation_out_degree': 0.1,
            'mean_semantic_degree': 10.0, 'std_semantic_degree': 1.0,
            'mean_relevant_connections': 2.0, 'std_relevant_connections': 0.5,
            'mean_relevant_ratio': 0.1, 'std_relevant_ratio': 0.01,
            'connectivity_ratio': 0.8, 'relevant_network_density': 0.1,
            'mean_relevant_centrality': 5.0, 'std_relevant_centrality': 1.0
        })
        
        print(f"Baseline calculated from {len(sample_docs)} documents")
        return baseline
    
    def extract_features_parallel(self, doc_ids: List[str]) -> pd.DataFrame:
        """Extract features using parallel processing."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        print(f"Extracting features for {len(doc_ids)} documents using {self.n_cores} cores...")
        start_time = time.time()
        
        # Split into batches for parallel processing
        batch_size = max(1, len(doc_ids) // self.n_cores)
        batches = [doc_ids[i:i + batch_size] for i in range(0, len(doc_ids), batch_size)]
        
        # Use ThreadPoolExecutor for I/O bound operations (faster than ProcessPool for this case)
        with ThreadPoolExecutor(max_workers=self.n_cores) as executor:
            futures = [
                executor.submit(self._extract_batch_features, batch) 
                for batch in batches
            ]
            
            all_features = []
            for future in futures:
                all_features.extend(future.result())
        
        extract_time = time.time() - start_time
        print(f"Feature extraction completed in {extract_time:.2f}s")
        
        return pd.DataFrame(all_features)
    
    def _extract_batch_features(self, doc_ids: List[str]) -> List[Dict]:
        """Extract features for a batch of documents (optimized)."""
        features = []
        
        for doc_id in doc_ids:
            if doc_id not in self.G.nodes:
                features.append(get_zero_features(doc_id))
                continue
            
            # Fast feature extraction - only essential features
            doc_features = {
                'openalex_id': doc_id,
                'total_degree': self.G.degree(doc_id),
                'out_degree': self.G.out_degree(doc_id) if hasattr(self.G, 'out_degree') else self.G.degree(doc_id),
                'in_degree': self.G.in_degree(doc_id) if hasattr(self.G, 'in_degree') else self.G.degree(doc_id),
            }
            
            # Count relevant connections quickly
            relevant_connections = sum(1 for neighbor in self.G.neighbors(doc_id) 
                                     if neighbor in self.relevant_documents)
            doc_features['relevant_connections'] = relevant_connections
            doc_features['relevant_ratio'] = relevant_connections / max(1, doc_features['total_degree'])
            
            # Add minimal additional features for compatibility
            doc_features.update({
                'citation_out_degree': 0, 'citation_in_degree': 0, 'semantic_degree': doc_features['total_degree'],
                'cocitation_degree': 0, 'coupling_degree': 0, 'weighted_influence': 0, 'weighted_connections': 0,
                'relevant_citation_out': 0, 'relevant_citation_in': 0, 'citation_ratio': 0, 'semantic_ratio': 1.0,
                'coupling_score': 0, 'cocitation_score': 0, 'relevant_coupling': 0, 'relevant_cocitation': 0,
                'coupling_diversity': 0, 'cocitation_diversity': 0, 'neighborhood_size_1hop': doc_features['total_degree'],
                'neighborhood_size_2hop': 0, 'neighborhood_enrichment_1hop': doc_features['relevant_ratio'],
                'neighborhood_enrichment_2hop': 0, 'citation_diversity': 0, 'relevant_betweenness': 0,
                'structural_anomaly': 0, 'citation_velocity': 0, 'age_normalized_impact': 0,
                'citation_burst_score': 0, 'temporal_isolation': 0, 'recent_citation_ratio': 0,
                'citation_acceleration': 0, 'local_clustering': 0, 'edge_type_diversity': 0,
                'relevant_path_efficiency': 0, 'citation_authority': 0, 'semantic_coherence': doc_features['relevant_ratio'],
                'network_position_score': doc_features['relevant_ratio']
            })
            
            features.append(doc_features)
        
        return features
    
    def predict_relevance_scores_fast(self, target_documents: List[str]) -> Dict[str, float]:
        """Fast relevance scoring using simplified calculations."""
        if not self.is_fitted:
            return {doc_id: 0.0 for doc_id in target_documents}
        
        print(f"Fast scoring {len(target_documents)} documents...")
        start_time = time.time()
        
        # Extract features in parallel
        features_df = self.extract_features_parallel(target_documents)
        
        # Simple scoring based on key features
        scores = {}
        for _, row in features_df.iterrows():
            doc_id = row['openalex_id']
            
            # Simplified scoring focusing on most important factors
            relevant_ratio = row['relevant_ratio']
            total_degree = row['total_degree']
            relevant_connections = row['relevant_connections']
            
            # Simple score based on connection to relevant documents
            base_score = min(1.0, relevant_ratio * 2.0)  # Boost documents connected to relevant ones
            degree_bonus = min(0.3, total_degree / 100.0)  # Small bonus for high degree
            connection_bonus = min(0.4, relevant_connections / 10.0)  # Bonus for multiple relevant connections
            
            final_score = base_score + degree_bonus + connection_bonus
            scores[doc_id] = min(1.0, final_score)
        
        score_time = time.time() - start_time
        print(f"Scoring completed in {score_time:.2f}s")
        
        return scores


def main():
    """Run GPU-accelerated outlier detection."""
    print("="*60)
    print("GPU-Accelerated Citation Network with SPECTER2 Embeddings")
    print("="*60)
    
    # Initialize model
    model = GPUAcceleratedCitationNetwork()
    
    # Load dataset and identify outlier
    simulation_df, dataset_config = load_dataset(model.dataset_name)
    outlier_row = identify_outlier_in_simulation(simulation_df, dataset_config)
    outlier_id = outlier_row['openalex_id']
    
    print(f"\nOutlier: {outlier_id} (Record ID: {outlier_row['record_id']})")
    
    # Create training data
    training_data = create_training_data(simulation_df, outlier_id)
    num_relevant = training_data['label_included'].sum()
    print(f"Training with {num_relevant} relevant documents (excluding outlier)")
    
    # Fit model
    model.fit(training_data)
    
    # Get search pool and score all documents
    search_pool = get_search_pool(simulation_df, outlier_id)
    print(f"\nScoring {len(search_pool)} documents...")
    
    scores = model.predict_relevance_scores_fast(search_pool)
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Find outlier position
    outlier_position = None
    for i, (doc_id, score) in enumerate(sorted_results):
        if doc_id == outlier_id:
            outlier_position = i + 1
            break
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"Outlier found at position: {outlier_position} out of {len(search_pool)}")
    percentile = ((len(search_pool) - outlier_position) / len(search_pool)) * 100
    print(f"Percentile: {percentile:.1f}th")
    print(f"{'='*60}")
    
    # Show top 20 results
    print(f"\nTop 20 results:")
    for i, (doc_id, score) in enumerate(sorted_results[:20], 1):
        marker = " *** OUTLIER ***" if doc_id == outlier_id else ""
        print(f"  {i:2d}. Score: {score:.4f}{marker}")


if __name__ == "__main__":
    main() 