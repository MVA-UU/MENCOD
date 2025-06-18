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

# Fix import for direct script execution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # One level up: CITATION_EMBEDDINGS -> ADS Thesis
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import dataset utilities
from models.dataset_utils import (
    load_datasets_config,
    prompt_dataset_selection,
    load_dataset,
    identify_outlier_in_simulation,
    create_training_data,
    get_search_pool
)

# Import feature extraction modules from local module
from citation_features import (
    get_connectivity_features,
    get_coupling_features, 
    get_neighborhood_features,
    get_advanced_features,
    get_temporal_features,
    get_efficiency_features,
    get_zero_features
)

# Import scoring modules from local module
from citation_scoring import (
    calculate_isolation_deviation,
    calculate_coupling_deviation,
    calculate_neighborhood_deviation,
    calculate_advanced_score,
    calculate_temporal_score,
    calculate_efficiency_score,
    get_adaptive_weights
)

# Import network utilities from local module
from network_utils import (
    build_network_from_simulation,
    calculate_distance_baselines
)

# Import ranking module from local module
from citation_ranking import apply_sparse_dataset_ranking_adjustments


class CitationNetworkModel:
    """Citation-based feature extractor for outlier detection."""
    
    def __init__(self, dataset_name: Optional[str] = None):
        """
        Initialize the Citation Network model.
        
        Args:
            dataset_name: Optional name of dataset to use. If None, will prompt user.
        """
        # If dataset_name is not provided, prompt user to select one
        if dataset_name is None:
            self.dataset_name = prompt_dataset_selection()
        else:
            self.dataset_name = dataset_name
            
        print(f"Using dataset: {self.dataset_name}")
        
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
    
    def fit(self, simulation_df: Optional[pd.DataFrame] = None) -> 'CitationNetworkModel':
        """
        Build citation network and identify relevant documents.
        
        Args:
            simulation_df: Optional DataFrame with simulation results.
                           If None, will load from dataset configuration.
        
        Returns:
            self: Returns the fitted model
        """
        print("Building comprehensive citation network...")
        
        # Load simulation data if not provided
        if simulation_df is None:
            simulation_df, _ = load_dataset(self.dataset_name)
        
        # Store the simulation data for later use
        self.simulation_data = simulation_df.copy()
        
        # Create a comprehensive network from the simulation data
        # This now includes citation, semantic, co-citation, and bibliographic coupling edges
        self.G = build_network_from_simulation(simulation_df, self.dataset_name)
        
        # Identify relevant documents
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
        
        print(f"Citation network built: {len(self.G.nodes)} nodes, {total_edges} edges")
        print("Edge distribution by type:")
        for edge_type, count in sorted(edge_types.items()):
            print(f"  {edge_type}: {count} edges")
        print(f"Relevant documents identified: {len(self.relevant_documents)}")
        print(f"Baseline citation patterns calculated from {len(self.relevant_documents)} relevant docs")
        
        return self
    
    def get_citation_features(self, target_documents: List[str]) -> pd.DataFrame:
        """Extract citation-based features for target documents."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting features")
        
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
        
        return pd.DataFrame(features)
    
    def predict_relevance_scores(self, target_documents: List[str]) -> Dict[str, float]:
        """Generate citation-based outlier scores using relative deviation from baseline."""
        if not self.is_fitted or not self.baseline_stats:
            return {doc_id: 0.0 for doc_id in target_documents}
        
        # Process documents in smaller batches for large search pools
        batch_size = 200
        search_pool_size = len(target_documents)
        scores = {}
        
        # Display progress for large search pools
        if search_pool_size > 500:
            print(f"Processing {search_pool_size} documents in batches of {batch_size}...")
        
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
        
        # Process in batches
        all_scores = {'isolation': [], 'coupling': [], 'neighborhood': [], 'advanced': [], 'temporal': [], 'efficiency': []}
        raw_scores = {}
        feature_cache = {}  # Cache features for score normalization
        
        # Process in batches
        for batch_start in range(0, len(target_documents), batch_size):
            batch_end = min(batch_start + batch_size, len(target_documents))
            batch_docs = target_documents[batch_start:batch_end]
            
            # Get features for this batch
            features_df = self.get_citation_features(batch_docs)
            
            # Calculate scores for each document in batch
            for _, row in features_df.iterrows():
                doc_id = row['openalex_id']
                
                # Cache features for later use
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
                
                # Data-driven score adjustments based on sparsity
                if sparsity_factor > 0.7:  # Very sparse dataset
                    # For very sparse datasets, boost documents with any relevant connections
                    if row['relevant_connections'] > 0:
                        # Calculate boost based on relevant connection ratio compared to average
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
                
                # Store raw combined score (DISABLED advanced_score for testing)
                raw_scores[doc_id] = (
                    weights['isolation'] * iso_score + 
                    weights['coupling'] * coup_score + 
                    weights['neighborhood'] * neigh_score +
                    weights['temporal'] * temp_score +
                    weights['efficiency'] * eff_score
                    # + weights['advanced'] * adv_score  # DISABLED
                )
            
            # Report progress for large search pools
            if search_pool_size > 500:
                print(f"Processed batch {batch_start+1}-{batch_end} of {search_pool_size}")
        
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
        
        # Calculate distances using NetworkX's shortest_path_length
        dist = float('inf')
        for rel_doc in self.relevant_documents:
            if rel_doc in self.G.nodes:
                try:
                    d = nx.shortest_path_length(self.G, outlier_id, rel_doc)
                    dist = min(dist, d)
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
        """Calculate optimized baseline citation statistics from relevant documents."""
        if not self.relevant_documents:
            return {}
        
        print("Calculating baseline statistics (optimized for large networks)...")
        
        # Quick baseline calculation using simple degree statistics
        baseline = {}
        
        # Sample a subset of relevant documents for detailed analysis if there are many
        sample_relevant = list(self.relevant_documents)
        if len(sample_relevant) > 50:
            import random
            sample_relevant = random.sample(sample_relevant, 50)
            print(f"Using sample of {len(sample_relevant)} relevant documents for baseline")
        
        # Fast degree-based statistics
        degrees = []
        citation_in = []
        citation_out = []
        semantic_edges = []
        
        for doc_id in sample_relevant:
            if doc_id not in self.G.nodes:
                continue
                
            total_degree = self.G.degree(doc_id)
            degrees.append(total_degree)
            
            # Count edge types efficiently
            cin, cout, sem = 0, 0, 0
            for neighbor in self.G.neighbors(doc_id):
                edge_data = self.G.get_edge_data(doc_id, neighbor)
                if edge_data:
                    edge_type = edge_data.get('edge_type', 'unknown')
                    if edge_type == 'citation':
                        cout += 1
                    elif edge_type == 'semantic':
                        sem += 1
                        
            # Incoming citations
            for neighbor in self.G.predecessors(doc_id):
                edge_data = self.G.get_edge_data(neighbor, doc_id)
                if edge_data and edge_data.get('edge_type') == 'citation':
                    cin += 1
            
            citation_in.append(cin)
            citation_out.append(cout)
            semantic_edges.append(sem)
        
        # Calculate basic statistics
        if degrees:
            baseline['mean_total_degree'] = np.mean(degrees)
            baseline['std_total_degree'] = max(np.std(degrees), 0.1)
            baseline['median_total_degree'] = np.median(degrees)
        
        if citation_in:
            baseline['mean_citation_in_degree'] = np.mean(citation_in)
            baseline['std_citation_in_degree'] = max(np.std(citation_in), 0.1)
            
        if citation_out:
            baseline['mean_citation_out_degree'] = np.mean(citation_out)
            baseline['std_citation_out_degree'] = max(np.std(citation_out), 0.1)
            
        if semantic_edges:
            baseline['mean_semantic_degree'] = np.mean(semantic_edges)
            baseline['std_semantic_degree'] = max(np.std(semantic_edges), 0.1)
        
        # Network connectivity (simple version)
        relevant_subgraph = self.G.subgraph(sample_relevant)
        if len(relevant_subgraph.nodes) > 1:
            # Simple connectivity measure
            actual_edges = len(relevant_subgraph.edges)
            possible_edges = len(sample_relevant) * (len(sample_relevant) - 1)
            baseline['relevant_network_density'] = actual_edges / max(possible_edges, 1)
            
            # Average degree in relevant subgraph
            if relevant_subgraph.nodes:
                degrees_sub = [relevant_subgraph.degree(n) for n in relevant_subgraph.nodes]
                baseline['mean_relevant_centrality'] = np.mean(degrees_sub)
                baseline['std_relevant_centrality'] = max(np.std(degrees_sub), 0.1)
        else:
            baseline['relevant_network_density'] = 0.0
            baseline['mean_relevant_centrality'] = 0.0
            baseline['std_relevant_centrality'] = 0.1
        
        # Add some default values for compatibility
        baseline['mean_relevant_connections'] = baseline.get('mean_total_degree', 1.0)
        baseline['std_relevant_connections'] = baseline.get('std_total_degree', 0.1)
        baseline['mean_relevant_ratio'] = 0.1
        baseline['std_relevant_ratio'] = 0.01
        baseline['connectivity_ratio'] = min(1.0, baseline.get('relevant_network_density', 0.1) * 10)
        
        print(f"Baseline calculated from {len(sample_relevant)} relevant documents")
        return baseline


def main():
    """Test citation network model with a selected dataset and show outlier ranking."""
    
    # Create model (will prompt user to select dataset)
    model = CitationNetworkModel()
    
    # Load dataset
    simulation_df, dataset_config = load_dataset(model.dataset_name)
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
    scores = model.predict_relevance_scores(search_pool)
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