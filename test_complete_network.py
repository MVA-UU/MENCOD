#!/usr/bin/env python3
"""
Test script to verify enhanced citation network building with full synergy dataset.

This script tests the new functionality where the citation network is built
from the FULL synergy dataset while ensuring only simulation papers are 
eligible for outlier detection.
"""

import logging
import pandas as pd
from ECINOD import CitationNetworkOutlierDetector
from utils import load_simulation_data, load_full_synergy_csv, load_datasets_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_enhanced_network_building():
    """Test the enhanced citation network building functionality."""
    print("=" * 70)
    print("TESTING ENHANCED CITATION NETWORK BUILDING")
    print("=" * 70)
    
    # Use a small dataset for testing
    dataset_name = "sep"  # Small dataset for quick testing
    
    try:
        # Load simulation data (subset)
        print(f"\n1. Loading simulation data for '{dataset_name}'...")
        simulation_df = load_simulation_data(dataset_name)
        print(f"   Simulation dataset: {len(simulation_df)} papers")
        
        # Load full synergy CSV for comparison  
        print(f"\n2. Loading full synergy dataset...")
        full_synergy_df = load_full_synergy_csv(dataset_name)
        if full_synergy_df is not None:
            print(f"   Full synergy dataset: {len(full_synergy_df)} papers")
            print(f"   Network expansion: {len(full_synergy_df) - len(simulation_df)} additional papers")
        else:
            print("   Could not load full synergy dataset")
            return
        
        # Initialize detector
        print(f"\n3. Initializing enhanced outlier detector...")
        detector = CitationNetworkOutlierDetector(random_state=42)
        
        # Run enhanced outlier detection
        print(f"\n4. Running enhanced outlier detection...")
        results = detector.fit_predict_outliers(simulation_df, dataset_name=dataset_name)
        
        # Analyze results
        print(f"\n" + "=" * 50)
        print("ENHANCED NETWORK ANALYSIS RESULTS")
        print("=" * 50)
        
        # Get network statistics
        network_stats = detector.network_builder.get_network_stats(detector.graph)
        
        print(f"\nNetwork Statistics:")
        print(f"  Total nodes in network: {network_stats['nodes']}")
        print(f"  Total edges in network: {network_stats['edges']}")
        print(f"  Simulation eligible nodes: {network_stats['simulation_eligible_nodes']}")
        print(f"  Background nodes: {network_stats['background_nodes']}")
        print(f"  Network density: {network_stats['density']:.6f}")
        print(f"  Connected components: {network_stats['components']}")
        
        # Verify eligibility marking
        simulation_eligible_nodes = detector.network_builder.get_simulation_eligible_nodes(detector.graph)
        simulation_ids = set(simulation_df['openalex_id'].tolist())
        
        print(f"\nEligibility Verification:")
        print(f"  Simulation papers in dataset: {len(simulation_ids)}")
        print(f"  Simulation eligible nodes in network: {len(simulation_eligible_nodes)}")
        
        # Check if all simulation papers are marked as eligible
        eligible_set = set(simulation_eligible_nodes)
        missing_eligible = simulation_ids - eligible_set
        extra_eligible = eligible_set - simulation_ids
        
        if missing_eligible:
            print(f"  ⚠️  Missing eligible papers: {len(missing_eligible)}")
        else:
            print(f"  ✅ All simulation papers marked as eligible")
            
        if extra_eligible:
            print(f"  ⚠️  Extra eligible papers: {len(extra_eligible)}")
        else:
            print(f"  ✅ No extra papers marked as eligible")
        
        # Show top outliers
        print(f"\nTop 5 Outlier Documents (Ensemble Scores):")
        doc_ids = results['openalex_ids']
        ensemble_scores = results['ensemble_scores']
        
        # Sort by ensemble score (descending)
        sorted_indices = ensemble_scores.argsort()[::-1]
        
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i]
            doc_id = doc_ids[idx]
            score = ensemble_scores[idx]
            
            # Get paper info
            paper_info = simulation_df[simulation_df['openalex_id'] == doc_id]
            if not paper_info.empty:
                title = paper_info.iloc[0].get('title', 'Unknown')[:80] + "..."
                label = paper_info.iloc[0].get('label_included', 'Unknown')
                print(f"  {i+1}. Score: {score:.4f}, Label: {label}, Title: {title}")
        
        # Compare with expected outliers
        datasets_config = load_datasets_config()
        if dataset_name in datasets_config and 'outlier_ids' in datasets_config[dataset_name]:
            expected_outliers = datasets_config[dataset_name]['outlier_ids']
            print(f"\nExpected Outliers: {expected_outliers}")
            
            # Check ranking of expected outliers
            for outlier_id in expected_outliers:
                outlier_papers = simulation_df[simulation_df['record_id'] == outlier_id]
                if not outlier_papers.empty:
                    outlier_openalex_id = outlier_papers.iloc[0]['openalex_id']
                    if outlier_openalex_id in doc_ids:
                        outlier_idx = list(doc_ids).index(outlier_openalex_id)
                        outlier_score = ensemble_scores[outlier_idx]
                        rank = (ensemble_scores > outlier_score).sum() + 1
                        percentile = ((len(ensemble_scores) - rank + 1) / len(ensemble_scores)) * 100
                        print(f"  Outlier {outlier_id}: Rank {rank}/{len(ensemble_scores)}, "
                              f"Score: {outlier_score:.4f}, Percentile: {percentile:.1f}%")
        
        print(f"\n" + "=" * 70)
        print("ENHANCED NETWORK TESTING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_enhanced_network_building() 