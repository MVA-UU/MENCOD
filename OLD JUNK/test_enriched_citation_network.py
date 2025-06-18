#!/usr/bin/env python3
"""
Test script for the enriched citation network implementation.

This script tests the enhanced citation network model that incorporates
external data from OpenAlex to make the citation network more complete.
"""

import sys
import os

# Add the CITATION_NETWORK_ENRICHED directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CITATION_NETWORK_ENRICHED'))

from CitationNetwork.citation_network import CitationNetworkModel
from dataset_utils import load_dataset, get_outlier_info, identify_outlier_in_simulation


def test_enriched_citation_network():
    """Test the enriched citation network with external data."""
    
    print("=" * 60)
    print("Testing Enriched Citation Network Implementation")
    print("=" * 60)
    
    # Test with appenzeller dataset (has external data available)
    dataset_name = "appenzeller"
    
    print(f"\n1. Testing dataset loading with external data for '{dataset_name}'")
    print("-" * 50)
    
    # Load dataset with external data
    simulation_df, dataset_config, external_df = load_dataset(dataset_name, include_external=True)
    
    print(f"Original simulation data: {len(simulation_df)} documents")
    if external_df is not None:
        print(f"External data: {len(external_df)} documents")
    else:
        print("No external data available")
    
    # Get outlier information
    outlier_info = get_outlier_info(dataset_name)
    print(f"Expected outlier record IDs: {outlier_info['record_ids']}")
    
    # Identify outlier in simulation
    outlier_row = identify_outlier_in_simulation(simulation_df, dataset_config)
    print(f"Found outlier: record_id={outlier_row['record_id']}, openalex_id={outlier_row['openalex_id']}")
    
    print(f"\n2. Building enriched citation network")
    print("-" * 50)
    
    # Initialize and fit the citation network model with external data
    # Use 4 cores for testing, you can increase this for supercomputer runs
    model = CitationNetworkModel(dataset_name, n_cores=4)
    model.fit(include_external=True)
    
    print(f"\n3. Testing feature extraction")
    print("-" * 50)
    
    # Test feature extraction on a small subset
    test_docs = simulation_df['openalex_id'].head(10).tolist()
    features = model.get_citation_features(test_docs)
    
    print(f"Extracted features for {len(test_docs)} documents")
    print(f"Feature columns: {list(features.columns)}")
    
    print(f"\n4. Testing relevance scoring")
    print("-" * 50)
    
    # Test relevance scoring
    scores = model.predict_relevance_scores(test_docs)
    
    print(f"Generated relevance scores for {len(scores)} documents")
    print("Sample scores:")
    for i, (doc_id, score) in enumerate(list(scores.items())[:5]):
        print(f"  {doc_id}: {score:.4f}")
    
    print(f"\n5. Testing outlier analysis")
    print("-" * 50)
    
    # Analyze the known outlier
    outlier_id = outlier_row['openalex_id']
    outlier_analysis = model.analyze_outlier(outlier_id)
    
    print(f"Outlier analysis for {outlier_id}:")
    for metric, value in outlier_analysis.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\n6. Network statistics")
    print("-" * 50)
    
    # Print detailed network statistics
    total_nodes = len(model.G.nodes())
    original_nodes = len([n for n in model.G.nodes() if not model.G.nodes[n].get('is_external', False)])
    external_nodes = len([n for n in model.G.nodes() if model.G.nodes[n].get('is_external', False)])
    
    print(f"Total network nodes: {total_nodes}")
    print(f"  - Original documents: {original_nodes}")
    print(f"  - External documents: {external_nodes}")
    print(f"Total network edges: {len(model.G.edges())}")
    print(f"Relevant documents: {len(model.relevant_documents)}")
    
    # Edge type statistics
    edge_types = {}
    for u, v, data in model.G.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print("Edge distribution:")
    for edge_type, count in sorted(edge_types.items()):
        print(f"  {edge_type}: {count}")
    
    print(f"\n7. Testing without external data for comparison")
    print("-" * 50)
    
    # Test without external data for comparison
    model_no_external = CitationNetworkModel(dataset_name, n_cores=4)
    model_no_external.fit(include_external=False)
    
    print(f"Network without external data:")
    print(f"  Total nodes: {len(model_no_external.G.nodes())}")
    print(f"  Total edges: {len(model_no_external.G.edges())}")
    
    improvement_nodes = total_nodes - len(model_no_external.G.nodes())
    improvement_edges = len(model.G.edges()) - len(model_no_external.G.edges())
    
    print(f"\nImprovement with external data:")
    print(f"  Additional nodes: {improvement_nodes}")
    print(f"  Additional edges: {improvement_edges}")
    
    print(f"\n" + "=" * 60)
    print("Test completed successfully!")
    print("The enriched citation network implementation is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    test_enriched_citation_network() 