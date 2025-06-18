#!/usr/bin/env python3
"""
Example: How to use Citation Network with SPECTER2 embeddings

This demonstrates the proper way to use the citation network for research.
"""

import sys
import os

# Add the CITATION_EMBEDDINGS directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CITATION_EMBEDDINGS'))

def main():
    """Demonstrate citation network usage with SPECTER2 embeddings."""
    print("="*60)
    print("Citation Network with SPECTER2 Embeddings - Standalone")
    print("="*60)
    
    # You can now run the citation network directly:
    from citation_network import CitationNetworkModel
    
    print("\nTo use the citation network with SPECTER2 embeddings:")
    print("1. Import: from CITATION_EMBEDDINGS.citation_network import CitationNetworkModel")
    print("2. Initialize: model = CitationNetworkModel(dataset_name='appenzeller')")
    print("3. Fit: model.fit()")
    print("4. Extract features: features = model.get_citation_features(document_ids)")
    print("5. Get scores: scores = model.predict_relevance_scores(document_ids)")
    
    print(f"\nSPECTER2 embeddings configuration:")
    print(f"- Location: appenzeller_embeddings/specter2_embeddings.npy")
    print(f"- Dimensions: 768 (confirmed working)")
    print(f"- Documents: 2,680 (from Appenzeller dataset)")
    print(f"- Model: allenai/specter2 with proximity adapter")
    
    print(f"\nNetwork configuration:")
    print(f"- Semantic edges: SPECTER2 embeddings (not TF-IDF)")
    print(f"- Citation edges: From Synergy dataset")
    print(f"- Co-citation and bibliographic coupling: From Synergy dataset")
    print(f"- Similarity thresholds: Optimized for SPECTER2 (0.80+ for general, 0.75+ for relevant)")
    
    print(f"\n" + "="*60)
    print("SETUP COMPLETE: Citation network ready with SPECTER2 embeddings!")
    print("You can now use citation_network.py from CITATION_EMBEDDINGS/")
    print("="*60)

if __name__ == "__main__":
    main() 