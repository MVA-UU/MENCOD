#!/usr/bin/env python3
"""
Optimized Citation Network Script for Supercomputer Execution.

This script provides a command-line interface for running the enriched citation network
implementation with configurable multiprocessing options for supercomputer environments.
"""

import argparse
import os
import sys
import time
import json
from typing import List, Dict

# Add the CITATION_NETWORK_ENRICHED directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CITATION_NETWORK_ENRICHED'))

from CitationNetwork.citation_network import CitationNetworkModel
from dataset_utils import get_available_datasets, load_dataset, get_outlier_info, identify_outlier_in_simulation, get_search_pool


def main():
    parser = argparse.ArgumentParser(
        description='Run enriched citation network analysis with multiprocessing support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset options
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=get_available_datasets(),
        required=True,
        help='Dataset to use for analysis'
    )
    
    # Performance options
    parser.add_argument(
        '--cores', 
        type=int, 
        default=None,
        help='Number of CPU cores to use (default: all available cores)'
    )
    
    parser.add_argument(
        '--no-external', 
        action='store_true',
        help='Disable external data enrichment'
    )
    
    # Output options
    parser.add_argument(
        '--output', 
        type=str, 
        default='citation_network_results.json',
        help='Output file for results'
    )
    
    parser.add_argument(
        '--features-output', 
        type=str, 
        default='citation_features.csv',
        help='Output CSV file for extracted features'
    )
    
    parser.add_argument(
        '--scores-output', 
        type=str, 
        default='relevance_scores.csv',
        help='Output CSV file for relevance scores'
    )
    
    # Analysis options
    parser.add_argument(
        '--full-analysis', 
        action='store_true',
        help='Perform full analysis on entire search pool (may be slow for large datasets)'
    )
    
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=1000,
        help='Sample size for analysis when --full-analysis is not used'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ENRICHED CITATION NETWORK ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Cores: {args.cores if args.cores else 'all available'}")
    print(f"External data: {'disabled' if args.no_external else 'enabled'}")
    print(f"Full analysis: {'yes' if args.full_analysis else f'sample of {args.sample_size}'}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Load dataset information
        print("\n1. Loading dataset information...")
        simulation_df, dataset_config, external_df = load_dataset(args.dataset, include_external=not args.no_external)
        outlier_info = get_outlier_info(args.dataset)
        outlier_row = identify_outlier_in_simulation(simulation_df, dataset_config)
        
        print(f"   Original documents: {len(simulation_df)}")
        if external_df is not None:
            print(f"   External documents: {len(external_df)}")
        print(f"   Target outlier: {outlier_row['openalex_id']} (record_id={outlier_row['record_id']})")
        
        # Initialize and fit the model
        print("\n2. Building enriched citation network...")
        model = CitationNetworkModel(args.dataset, n_cores=args.cores)
        
        network_start = time.time()
        model.fit(include_external=not args.no_external)
        network_time = time.time() - network_start
        
        print(f"   Network built in {network_time:.2f} seconds")
        print(f"   Network nodes: {len(model.G.nodes)}")
        print(f"   Network edges: {len(model.G.edges)}")
        print(f"   Relevant documents: {len(model.relevant_documents)}")
        
        # Determine analysis scope
        if args.full_analysis:
            # Full search pool analysis
            search_pool = get_search_pool(simulation_df, outlier_row['openalex_id'])
            analysis_docs = search_pool
            print(f"\n3. Performing full analysis on {len(analysis_docs)} documents...")
        else:
            # Sample analysis
            search_pool = get_search_pool(simulation_df, outlier_row['openalex_id'])
            import random
            random.seed(42)  # For reproducibility
            
            # Always include the outlier
            sample_docs = [outlier_row['openalex_id']]
            remaining_docs = [doc for doc in search_pool if doc != outlier_row['openalex_id']]
            sample_size = min(args.sample_size - 1, len(remaining_docs))
            sample_docs.extend(random.sample(remaining_docs, sample_size))
            
            analysis_docs = sample_docs
            print(f"\n3. Performing sample analysis on {len(analysis_docs)} documents (including outlier)...")
        
        # Extract features
        features_start = time.time()
        features_df = model.get_citation_features(analysis_docs)
        features_time = time.time() - features_start
        
        print(f"   Features extracted in {features_time:.2f} seconds")
        print(f"   Feature dimensions: {features_df.shape}")
        
        # Calculate relevance scores
        scores_start = time.time()
        relevance_scores = model.predict_relevance_scores(analysis_docs)
        scores_time = time.time() - scores_start
        
        print(f"   Relevance scores calculated in {scores_time:.2f} seconds")
        
        # Analyze outlier performance
        outlier_id = outlier_row['openalex_id']
        if outlier_id in relevance_scores:
            outlier_score = relevance_scores[outlier_id]
            outlier_rank = sum(1 for score in relevance_scores.values() if score > outlier_score) + 1
            
            print(f"\n4. Outlier Analysis Results:")
            print(f"   Outlier score: {outlier_score:.6f}")
            print(f"   Outlier rank: {outlier_rank} out of {len(relevance_scores)}")
            print(f"   Percentile: {100 * (len(relevance_scores) - outlier_rank + 1) / len(relevance_scores):.2f}%")
            
            # Detailed outlier analysis
            outlier_analysis = model.analyze_outlier(outlier_id)
            print(f"   Detailed analysis:")
            for metric, value in sorted(outlier_analysis.items()):
                print(f"     {metric}: {value:.6f}")
        else:
            print(f"\n4. Warning: Outlier {outlier_id} not found in analysis results")
        
        # Save results
        print(f"\n5. Saving results...")
        
        # Save features
        features_df.to_csv(args.features_output, index=False)
        print(f"   Features saved to: {args.features_output}")
        
        # Save scores
        import pandas as pd
        scores_df = pd.DataFrame([
            {'openalex_id': doc_id, 'relevance_score': score}
            for doc_id, score in relevance_scores.items()
        ]).sort_values('relevance_score', ascending=False)
        scores_df.to_csv(args.scores_output, index=False)
        print(f"   Scores saved to: {args.scores_output}")
        
        # Save comprehensive results
        results = {
            'dataset': args.dataset,
            'parameters': {
                'cores': args.cores,
                'external_data': not args.no_external,
                'full_analysis': args.full_analysis,
                'sample_size': args.sample_size if not args.full_analysis else len(analysis_docs)
            },
            'dataset_info': {
                'original_documents': len(simulation_df),
                'external_documents': len(external_df) if external_df is not None else 0,
                'target_outlier': outlier_row['openalex_id'],
                'outlier_record_id': int(outlier_row['record_id'])
            },
            'network_stats': {
                'nodes': len(model.G.nodes),
                'edges': len(model.G.edges),
                'relevant_documents': len(model.relevant_documents)
            },
            'performance': {
                'network_build_time': network_time,
                'feature_extraction_time': features_time,
                'scoring_time': scores_time,
                'total_time': time.time() - start_time
            },
            'outlier_results': {
                'outlier_id': outlier_id,
                'outlier_score': outlier_score if outlier_id in relevance_scores else None,
                'outlier_rank': outlier_rank if outlier_id in relevance_scores else None,
                'total_documents': len(relevance_scores),
                'percentile': 100 * (len(relevance_scores) - outlier_rank + 1) / len(relevance_scores) if outlier_id in relevance_scores else None
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"   Results saved to: {args.output}")
        
        total_time = time.time() - start_time
        print(f"\n6. Analysis completed in {total_time:.2f} seconds")
        print("=" * 80)
        
        return results
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 