#!/usr/bin/env python3
"""
Test script for Citation Network with SPECTER2 embeddings
"""

import sys
import os

# Add the CITATION_EMBEDDINGS directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CITATION_EMBEDDINGS'))

from citation_network import CitationNetworkModel

def test_specter2_citation_network():
    """Test the citation network with SPECTER2 embeddings using Appenzeller dataset."""
    print("="*60)
    print("Testing Citation Network with SPECTER2 Embeddings")
    print("="*60)
    
    try:
        # Initialize the model with Appenzeller dataset
        print("Initializing Citation Network Model...")
        model = CitationNetworkModel(dataset_name='appenzeller')
        
        # Fit the model (this will build the network with SPECTER2 embeddings)
        print("\nFitting the model...")
        model.fit()
        
        # Get some basic statistics
        print(f"\nNetwork Statistics:")
        print(f"Number of nodes: {len(model.G.nodes)}")
        print(f"Number of edges: {len(model.G.edges)}")
        print(f"Relevant documents: {len(model.relevant_documents)}")
        
        # Test feature extraction on a small subset
        print("\nTesting feature extraction...")
        # Get first 10 document IDs for testing
        doc_ids = list(model.G.nodes)[:10]
        
        features_df = model.get_citation_features(doc_ids)
        print(f"Features extracted for {len(features_df)} documents")
        print("Feature columns:", list(features_df.columns))
        
        # Test relevance scoring
        print("\nTesting relevance scoring...")
        scores = model.predict_relevance_scores(doc_ids)
        print(f"Scores generated for {len(scores)} documents")
        
        # Show some example scores
        print("\nExample scores:")
        for doc_id, score in list(scores.items())[:5]:
            print(f"  {doc_id}: {score:.4f}")
        
        print("\n" + "="*60)
        print("SUCCESS: Citation Network with SPECTER2 embeddings works!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_specter2_citation_network()
    sys.exit(0 if success else 1) 