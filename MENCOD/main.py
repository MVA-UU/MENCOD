"""
Standalone Citation Network Outlier Detection Script

Provides the same main functionality as the original citation_network.py
for running MENCOD as a standalone application.
"""

import time
import logging
import argparse
from MENCOD import CitationNetworkOutlierDetector
from utils import (
    prompt_dataset_selection, load_simulation_data, 
    load_datasets_config, evaluate_outlier_ranking
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MENCOD - Citation Network Outlier Detection')
    parser.add_argument('--RRF', '--rrf', action='store_true', 
                       help='Use Robust Reciprocal Rank Fusion for ensemble scoring')
    return parser.parse_args()


def main():
    """Standalone execution of MENCOD."""
    # Parse command line arguments
    args = parse_arguments()
    
    print("=" * 60)
    print("MENCOD - EXTENDED CITATION NETWORK OUTLIER DETECTION")
    print("=" * 60)
    
    if args.RRF:
        print("Using Robust Reciprocal Rank Fusion (RRF) for ensemble scoring")
    else:
        print("Using default variance-weighted ensemble scoring")
    
    try:
        # Dataset Selection
        print("\nStep 1: Dataset Selection")
        dataset_name = prompt_dataset_selection()
        
        # Load Data
        print(f"\nStep 2: Loading dataset '{dataset_name}'...")
        simulation_df = load_simulation_data(dataset_name)
        print(f"Loaded {len(simulation_df)} documents")
        
        # Initialize and Run Model
        print(f"\nStep 3: Running Citation Network Outlier Detection...")
        print("Methods: LOF (embeddings), Isolation Forest")
        
        detector = CitationNetworkOutlierDetector(random_state=42, use_rrf=args.RRF)
        
        start_time = time.time()
        results = detector.fit_predict_outliers(simulation_df, dataset_name=dataset_name)
        runtime = time.time() - start_time
        
        print(f"Model completed in {runtime:.2f} seconds")
        
        # Evaluate Known Outliers
        print(f"\n" + "=" * 50)
        print("OUTLIER RANKING PERFORMANCE")
        print("=" * 50)
        
        # Create score dictionary for ranking
        doc_ids = results['openalex_ids']
        ensemble_scores = results['ensemble_scores']
        scores_dict = dict(zip(doc_ids, ensemble_scores))
        
        # Load datasets config for outlier evaluation
        datasets_config = load_datasets_config()
        outlier_ranking_results = evaluate_outlier_ranking(scores_dict, dataset_name, datasets_config)
        
        if outlier_ranking_results:
            for result in outlier_ranking_results:
                print(f"\nKnown Outlier: {result['outlier_id']}")
                print(f"  Rank: {result['rank']} out of {result['total_documents']}")
                print(f"  Ensemble Score: {result['score']:.4f}")
                print(f"  Percentile: {result['percentile']:.1f}%")
                
                # Display individual method subscores for this outlier
                outlier_id = result['outlier_id']
                try:
                    # Find the index of this document in the results
                    doc_index = list(doc_ids).index(outlier_id)
                    
                    # Extract individual subscores
                    lof_emb_score = results.get('lof_embeddings_scores', [0] * len(doc_ids))[doc_index]
                    lof_net_score = results.get('lof_network_scores', [0] * len(doc_ids))[doc_index]  
                    lof_mix_score = results.get('lof_mixed_scores', [0] * len(doc_ids))[doc_index]
                    isolation_score = results.get('isolation_forest_scores', [0] * len(doc_ids))[doc_index]
                    
                    print(f"  Individual Method Scores:")
                    print(f"    LOF-Embeddings:    {lof_emb_score:.4f}")
                    print(f"    LOF-Network:       {lof_net_score:.4f}")
                    print(f"    LOF-Mixed:         {lof_mix_score:.4f}")
                    print(f"    Isolation Forest:  {isolation_score:.4f}")
                    
                except (ValueError, IndexError):
                    print(f"  Individual scores not available for {outlier_id}")
                
                if result['percentile'] >= 95:
                    performance = "Excellent ✓"
                elif result['percentile'] >= 90:
                    performance = "Very Good ✓"
                elif result['percentile'] >= 80:
                    performance = "Good"
                else:
                    performance = "Poor ✗"
                print(f"  Performance: {performance}")
                
                # Show comprehensive thesis-level analysis for the first known outlier
                if result == outlier_ranking_results[0]:
                    print(f"\n" + "=" * 60)
                    print("DETAILED THESIS ANALYSIS FOR FIRST KNOWN OUTLIER")
                    print("=" * 60)
                    detector.print_thesis_analysis(outlier_id)
        else:
            print("No known outliers defined for this dataset.")
        
        # Show Top Documents with Detailed Scores
        print(f"\n" + "=" * 50)
        print("TOP OUTLIER DOCUMENTS - DETAILED BREAKDOWN")
        print("=" * 50)
        
        detector.print_outlier_score_summary(top_k=10)
        
        # Method Comparison
        print(f"\n" + "=" * 50)
        print("METHOD COMPARISON")
        print("=" * 50)
        
        detector.print_method_comparison()
        
        print(f"\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 