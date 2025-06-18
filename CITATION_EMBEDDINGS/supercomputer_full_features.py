#!/usr/bin/env python3
"""
Full-Featured GPU-Accelerated Citation Network for Supercomputer

This version uses complete citation network features for maximum accuracy
while maintaining GPU acceleration and multiprocessing optimizations.
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
from citation_features import (
    get_connectivity_features, get_coupling_features, get_neighborhood_features,
    get_advanced_features, get_temporal_features, get_efficiency_features, get_zero_features
)
from citation_scoring import (
    calculate_isolation_deviation, calculate_coupling_deviation,
    calculate_neighborhood_deviation, calculate_advanced_score,
    calculate_temporal_score, calculate_efficiency_score, get_adaptive_weights
)
from network_utils import build_network_from_simulation
from citation_ranking import apply_sparse_dataset_ranking_adjustments


class FullFeaturedGPUCitationNetwork:
    """Full-featured GPU-accelerated citation network."""
    
    def __init__(self, dataset_name: Optional[str] = None, n_cores: int = None):
        """Initialize with GPU acceleration and multiprocessing."""
        self.dataset_name = dataset_name or prompt_dataset_selection()
        self.n_cores = n_cores or min(mp.cpu_count(), 15)
        
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
                rmm.reinitialize(pool_allocator=True, initial_pool_size=2**29)  # 512MB pool
                print("GPU memory pool initialized (512MB)")
            except:
                print("GPU memory pool initialization failed, continuing...")
    
    def fit(self, simulation_df: Optional[pd.DataFrame] = None) -> 'FullFeaturedGPUCitationNetwork':
        """Build citation network with GPU acceleration."""
        start_time = time.time()
        print("Building full-featured GPU-accelerated citation network...")
        
        if simulation_df is None:
            simulation_df, _ = load_dataset(self.dataset_name)
        
        self.simulation_data = simulation_df.copy()
        
        # Build comprehensive network
        self.G = build_network_from_simulation(simulation_df, self.dataset_name)
        
        # Convert to GPU if beneficial
        if CUGRAPH_AVAILABLE and 50000 < len(self.G.edges) < 2000000:
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
        self.baseline_stats = self._calculate_comprehensive_baseline()
        
        # Print detailed statistics
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
        
        edges = []
        for u, v, data in self.G.edges(data=True):
            edges.append([u, v, data.get('weight', 1.0)])
        
        if edges:
            edge_df = cudf.DataFrame(edges, columns=['src', 'dst', 'weight'])
            self.gpu_graph = cx.Graph()
            self.gpu_graph.from_cudf_edgelist(edge_df, source='src', destination='dst', edge_attr='weight')
            print(f"GPU graph created with {len(edges)} edges")
    
    def _calculate_comprehensive_baseline(self) -> Dict[str, float]:
        """Calculate comprehensive baseline statistics efficiently."""
        if not self.relevant_documents:
            return {}
        
        print("Calculating comprehensive baseline statistics...")
        
        # Sample for efficiency but get good statistics
        sample_size = min(50, len(self.relevant_documents))
        sample_docs = list(self.relevant_documents)[:sample_size]
        
        # Extract features for baseline calculation
        baseline_features = []
        for doc_id in sample_docs:
            if doc_id in self.G.nodes:
                # Collect all feature types
                features = {'openalex_id': doc_id}
                features.update(get_connectivity_features(self.G, doc_id, self.relevant_documents))
                features.update(get_coupling_features(self.G, doc_id, self.relevant_documents))
                features.update(get_neighborhood_features(self.G, doc_id, self.relevant_documents))
                features.update(get_advanced_features(self.G, doc_id, self.relevant_documents))
                features.update(get_temporal_features(self.G, doc_id, self.relevant_documents, self.simulation_data))
                features.update(get_efficiency_features(self.G, doc_id, self.relevant_documents))
                baseline_features.append(features)
        
        if not baseline_features:
            return {}
        
        # Calculate statistics
        feature_df = pd.DataFrame(baseline_features)
        baseline = {}
        
        for col in feature_df.columns:
            if col != 'openalex_id' and feature_df[col].dtype in ['int64', 'float64']:
                baseline[f'mean_{col}'] = float(feature_df[col].mean())
                baseline[f'std_{col}'] = max(float(feature_df[col].std()), 0.1)
        
        print(f"Comprehensive baseline calculated from {len(baseline_features)} documents")
        return baseline
    
    def extract_features_parallel_full(self, doc_ids: List[str]) -> pd.DataFrame:
        """Extract full citation features using parallel processing."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        print(f"Extracting full features for {len(doc_ids)} documents using {self.n_cores} cores...")
        start_time = time.time()
        
        # Split into batches
        batch_size = max(1, len(doc_ids) // self.n_cores)
        batches = [doc_ids[i:i + batch_size] for i in range(0, len(doc_ids), batch_size)]
        
        # Use ThreadPoolExecutor to avoid GPU object serialization issues
        with ThreadPoolExecutor(max_workers=self.n_cores) as executor:
            futures = [
                executor.submit(self._extract_batch_features_full, batch, i) 
                for i, batch in enumerate(batches)
            ]
            
            all_features = []
            completed = 0
            total_batches = len(futures)
            
            for future in futures:
                all_features.extend(future.result())
                completed += 1
                print(f"Completed batch {completed}/{total_batches} ({completed/total_batches*100:.1f}%)")
                
                # Estimate remaining time
                elapsed = time.time() - start_time
                if completed > 0:
                    avg_time_per_batch = elapsed / completed
                    remaining_batches = total_batches - completed
                    estimated_remaining = avg_time_per_batch * remaining_batches
                    print(f"  Elapsed: {elapsed:.1f}s, Estimated remaining: {estimated_remaining:.1f}s")
        
        extract_time = time.time() - start_time
        print(f"Full feature extraction completed in {extract_time:.2f}s")
        
        return pd.DataFrame(all_features)
    
    def _extract_batch_features_full(self, doc_ids: List[str], batch_idx: int = 0) -> List[Dict]:
        """Extract full features for a batch of documents."""
        features = []
        batch_size = len(doc_ids)
        
        for i, doc_id in enumerate(doc_ids):
            try:
                if doc_id not in self.G.nodes:
                    features.append(get_zero_features(doc_id))
                    continue
                
                # Extract all feature types
                doc_features = {'openalex_id': doc_id}
                doc_features.update(get_connectivity_features(self.G, doc_id, self.relevant_documents))
                doc_features.update(get_coupling_features(self.G, doc_id, self.relevant_documents))
                doc_features.update(get_neighborhood_features(self.G, doc_id, self.relevant_documents))
                doc_features.update(get_advanced_features(self.G, doc_id, self.relevant_documents))
                doc_features.update(get_temporal_features(self.G, doc_id, self.relevant_documents, self.simulation_data))
                doc_features.update(get_efficiency_features(self.G, doc_id, self.relevant_documents))
                features.append(doc_features)
            except Exception as e:
                print(f"Error extracting features for {doc_id}: {e}")
                features.append(get_zero_features(doc_id))
        
        return features
    
    def predict_relevance_scores_full(self, target_documents: List[str]) -> Dict[str, float]:
        """Full relevance scoring using all citation network features."""
        if not self.is_fitted:
            return {doc_id: 0.0 for doc_id in target_documents}
        
        print(f"Full scoring {len(target_documents)} documents...")
        start_time = time.time()
        
        # Extract comprehensive features
        features_df = self.extract_features_parallel_full(target_documents)
        feature_time = time.time() - start_time
        print(f"Feature extraction completed in {feature_time:.2f}s, starting scoring...")
        
        # Calculate comprehensive scores with progress tracking
        scores = {}
        total_docs = len(features_df)
        scoring_start = time.time()
        
        for idx, (_, row) in enumerate(features_df.iterrows()):
            doc_id = row['openalex_id']
            
            try:
                # Use optimized scoring pipeline
                isolation_score = calculate_isolation_deviation(row, self.baseline_stats, self.G, self.relevant_documents)
                coupling_score = calculate_coupling_deviation(row, self.baseline_stats)
                neighborhood_score = calculate_neighborhood_deviation(row, self.baseline_stats)
                
                # Calculate dataset relevant ratio for advanced score
                dataset_size = len(self.simulation_data)
                relevant_ratio = len(self.relevant_documents) / dataset_size
                advanced_score = calculate_advanced_score(row, relevant_ratio)
                
                temporal_score = calculate_temporal_score(row, self.baseline_stats)
                efficiency_score = calculate_efficiency_score(row, self.baseline_stats)
                
                # Get adaptive weights
                sparsity_factor = 1 - min(0.9, max(0.1, relevant_ratio * 10))
                weights = get_adaptive_weights(sparsity_factor)
                
                # Calculate weighted final score
                final_score = (
                    weights['isolation'] * isolation_score +
                    weights['coupling'] * coupling_score +
                    weights['neighborhood'] * neighborhood_score +
                    weights['advanced'] * advanced_score +
                    weights['temporal'] * temporal_score +
                    weights['efficiency'] * efficiency_score
                )
                
                scores[doc_id] = max(0.0, min(1.0, final_score))
                
            except Exception as e:
                print(f"Error scoring {doc_id}: {e}")
                scores[doc_id] = 0.0
            
            # Progress logging every 500 documents
            if (idx + 1) % 500 == 0 or (idx + 1) == total_docs:
                elapsed = time.time() - scoring_start
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (total_docs - idx - 1) / rate if rate > 0 else 0
                print(f"Scored {idx + 1}/{total_docs} documents ({(idx + 1)/total_docs*100:.1f}%) | "
                      f"Rate: {rate:.1f} docs/sec | ETA: {remaining:.1f}s")
        
        total_score_time = time.time() - start_time
        print(f"Full scoring completed in {total_score_time:.2f}s")
        print(f"  Feature extraction: {feature_time:.2f}s")
        print(f"  Scoring computation: {total_score_time - feature_time:.2f}s")
        
        return scores


def main():
    """Run full-featured GPU-accelerated outlier detection."""
    print("="*80)
    print("Full-Featured GPU-Accelerated Citation Network with SPECTER2 Embeddings")
    print("="*80)
    
    # Initialize model
    model = FullFeaturedGPUCitationNetwork()
    
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
    print(f"\nScoring {len(search_pool)} documents with full features...")
    
    scores = model.predict_relevance_scores_full(search_pool)
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Find outlier position
    outlier_position = None
    for i, (doc_id, score) in enumerate(sorted_results):
        if doc_id == outlier_id:
            outlier_position = i + 1
            break
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"Outlier found at position: {outlier_position} out of {len(search_pool)}")
    percentile = ((len(search_pool) - outlier_position) / len(search_pool)) * 100
    print(f"Percentile: {percentile:.1f}th")
    print(f"{'='*80}")
    
    # Show top 20 results with scores
    print(f"\nTop 20 results:")
    for i, (doc_id, score) in enumerate(sorted_results[:20], 1):
        marker = " *** OUTLIER ***" if doc_id == outlier_id else ""
        print(f"  {i:2d}. Score: {score:.4f}{marker}")


if __name__ == "__main__":
    main() 