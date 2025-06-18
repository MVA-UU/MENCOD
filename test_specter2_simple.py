#!/usr/bin/env python3
"""
Simple test script for SPECTER2 embeddings in citation network
"""

import sys
import os
import pandas as pd

# Add the CITATION_EMBEDDINGS directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CITATION_EMBEDDINGS'))

from network_utils import add_semantic_edges
import networkx as nx

def test_specter2_simple():
    """Simple test of SPECTER2 embeddings on a small subset."""
    print("="*60)
    print("Simple SPECTER2 Embeddings Test")
    print("="*60)
    
    try:
        # Create a small sample DataFrame (first 100 documents)
        from models.dataset_utils import load_dataset
        full_df, _ = load_dataset('appenzeller')
        
        # Take only first 100 documents for testing
        small_df = full_df.head(100).copy()
        print(f"Testing with {len(small_df)} documents")
        
        # Create a simple graph
        G = nx.DiGraph()
        
        # Add nodes
        for _, row in small_df.iterrows():
            doc_id = row['openalex_id']
            G.add_node(doc_id, title=row.get('title', ''), abstract=row.get('abstract', ''))
            G.nodes[doc_id]['label_included'] = row['label_included']
        
        print(f"Created graph with {len(G.nodes)} nodes")
        
        # Test SPECTER2 semantic edges
        print("\nTesting SPECTER2 semantic similarity...")
        add_semantic_edges(G, small_df)
        
        # Count semantic edges
        semantic_edges = 0
        for u, v, data in G.edges(data=True):
            if data.get('edge_type') == 'semantic' and data.get('embedding_type') == 'specter2':
                semantic_edges += 1
        
        print(f"\nResults:")
        print(f"Total edges: {len(G.edges)}")
        print(f"SPECTER2 semantic edges: {semantic_edges}")
        
        # Test similarity values
        similarities = []
        for u, v, data in G.edges(data=True):
            if data.get('edge_type') == 'semantic' and 'similarity' in data:
                similarities.append(data['similarity'])
        
        if similarities:
            import numpy as np
            print(f"Similarity statistics:")
            print(f"  Min: {np.min(similarities):.4f}")
            print(f"  Max: {np.max(similarities):.4f}")
            print(f"  Mean: {np.mean(similarities):.4f}")
            print(f"  Std: {np.std(similarities):.4f}")
        
        print("\n" + "="*60)
        print("SUCCESS: SPECTER2 embeddings working correctly!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_specter2_simple()
    sys.exit(0 if success else 1) 