#!/usr/bin/env python3
"""
Test script for enhanced citation network outlier detection.

This script demonstrates the research-backed outlier detection methods
implemented in the citation network model.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from citation_network import CitationNetworkModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_outlier_detection():
    """Test the enhanced outlier detection capabilities."""
    
    print("🔍 Testing Enhanced Citation Network Outlier Detection")
    print("=" * 60)
    
    # Initialize model with outlier detection enabled
    model = CitationNetworkModel(
        dataset_name=None,  # Use simulation data
        enable_gpu=False,   # Disable for testing
        enable_semantic=False,  # Disable semantic features for testing (no embeddings)
        outlier_contamination=0.1  # Expect 10% outliers
    )
    
    # Create sample simulation data for testing
    sample_data = create_sample_data()
    
    print(f"📊 Created sample dataset with {len(sample_data)} documents")
    
    # Fit the model
    print("\n🔧 Fitting model with outlier detection...")
    model.fit(simulation_df=sample_data)
    
    # Test outlier detection methods
    print("\n🎯 Testing outlier detection methods:")
    
    # 1. Get outlier documents
    outlier_docs = model.get_outlier_documents()
    print(f"   • Found {len(outlier_docs)} outlier documents")
    
    if outlier_docs:
        print("   • Top 3 outliers:")
        for i, outlier in enumerate(outlier_docs[:3]):
            print(f"     {i+1}. Document: {outlier['document_id'][:20]}...")
            print(f"        LOF Score: {outlier['lof_score']:.3f}")
            print(f"        IF Score: {outlier['isolation_forest_score']:.3f}")
            print(f"        Ensemble Score: {outlier['ensemble_score']:.3f}")
            print(f"        Is Relevant: {outlier['is_relevant']}")
            print()
    
    # 2. Get anomalous citations
    anomalous_citations = model.get_anomalous_citations()
    print(f"   • Found {len(anomalous_citations)} anomalous citations")
    
    if anomalous_citations:
        print("   • Top 3 anomalous citations:")
        for i, citation in enumerate(anomalous_citations[:3]):
            print(f"     {i+1}. Citation: {citation['citing_document'][:15]}... → {citation['cited_document'][:15]}...")
            print(f"        Ensemble Score: {citation['ensemble_score']:.3f}")
            print(f"        Features: {list(citation['features'].keys())[:5]}...")
            print()
    
    # 3. Test relevance scoring with outlier adjustments
    print("\n📈 Testing relevance scoring with outlier adjustments:")
    
    # Get some test documents
    test_docs = sample_data['openalex_id'].tolist()[:10]
    relevance_scores = model.predict_relevance_scores(test_docs)
    
    print(f"   • Computed relevance scores for {len(test_docs)} documents")
    print("   • Sample scores:")
    for i, (doc_id, score) in enumerate(list(relevance_scores.items())[:5]):
        print(f"     {i+1}. {doc_id[:20]}...: {score:.3f}")
    
    # 4. Test detailed document analysis
    print("\n🔍 Testing detailed document analysis:")
    
    if test_docs:
        analysis = model.analyze_document(test_docs[0])
        print(f"   • Analysis for document: {test_docs[0][:30]}...")
        print(f"     - Relevance Score: {analysis['relevance_score']:.3f}")
        print(f"     - In Network: {analysis['in_network']}")
        print(f"     - Has Embeddings: {analysis.get('has_embeddings', False)}")
        if 'neighbors' in analysis:
            print(f"     - Neighbors: {len(analysis['neighbors'])}")
    
    print("\n✅ Outlier detection testing completed!")
    print("\n📝 Summary of Research-Based Improvements:")
    print("   • Local Outlier Factor (LOF) for density-based anomaly detection")
    print("   • Isolation Forest for global anomaly detection")
    print("   • Edge attribute analysis for citation relationship anomalies")
    print("   • Purpose-based citation analysis for scientific relevance")
    print("   • Multi-modal fusion of semantic and structural features")
    print("   • Ensemble scoring for robust outlier identification")


def create_sample_data():
    """Create sample simulation data for testing."""
    np.random.seed(42)
    
    # Create sample document IDs
    doc_ids = [f"https://openalex.org/W{2000000000 + i}" for i in range(100)]
    
    # Create sample data with some patterns for outlier detection
    data = []
    for i, doc_id in enumerate(doc_ids):
        # Create some relevant and irrelevant documents
        is_relevant = np.random.choice([0, 1], p=[0.8, 0.2])  # 20% relevant
        
        # Create sample references (citation patterns)
        num_refs = np.random.poisson(5)  # Average 5 references
        if i < 20:  # First 20 documents
            references = doc_ids[max(0, i-10):i]  # Local citations
        else:
            references = np.random.choice(doc_ids[:i], min(num_refs, i), replace=False).tolist()
        
        data.append({
            'openalex_id': doc_id,
            'title': f"Sample Research Paper {i+1}",
            'year': 2020 + (i % 5),  # Years 2020-2024
            'label_included': is_relevant,
            'label': is_relevant,
            'references': references,
            'authors': [f"Author{i+1}A", f"Author{i+1}B"],
            'venue': f"Journal{(i % 10) + 1}",
            'affiliations': [f"University{(i % 5) + 1}"]
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    test_outlier_detection() 