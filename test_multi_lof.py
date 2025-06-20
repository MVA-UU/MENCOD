"""
Test script for Multi-LOF implementation
"""

import pandas as pd
import logging
from ECINOD.core import CitationNetworkOutlierDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_multi_lof():
    """Test the Multi-LOF implementation with a small dataset."""
    
    print("=" * 80)
    print("TESTING MULTI-LOF IMPLEMENTATION")
    print("=" * 80)
    
    # Load a simulation dataset for testing
    try:
        # Try hall dataset first (should be small for testing)
        simulation_df = pd.read_csv('data/simulations/hall.csv')
        dataset_name = 'hall'
        print(f"Loaded {dataset_name} dataset: {len(simulation_df)} papers")
    except FileNotFoundError:
        try:
            # Fallback to appenzeller
            simulation_df = pd.read_csv('data/simulations/appenzeller.csv')
            dataset_name = 'appenzeller'
            print(f"Loaded {dataset_name} dataset: {len(simulation_df)} papers")
        except FileNotFoundError:
            print("No simulation datasets found. Please ensure data/simulations/ contains CSV files.")
            return
    
    # Initialize detector
    detector = CitationNetworkOutlierDetector(random_state=42)
    
    # Run Multi-LOF detection
    print("\nRunning Multi-LOF outlier detection...")
    try:
        results = detector.fit_predict_outliers(simulation_df, dataset_name)
        
        print("\n" + "=" * 60)
        print("MULTI-LOF RESULTS SUMMARY")
        print("=" * 60)
        
        # Print method scores summary
        for method_key in results.keys():
            if method_key.endswith('_scores'):
                method_name = method_key.replace('_scores', '').upper()
                scores = results[method_key]
                print(f"{method_name:20s}: {len(scores)} scores, "
                      f"mean={scores.mean():.4f}, std={scores.std():.4f}")
        
        # Print top outliers
        print("\n")
        detector.print_outlier_score_summary(top_k=5)
        
        # Print method comparison
        detector.print_method_comparison()
        
        # Test individual method access
        print(f"\n" + "=" * 60)
        print("TESTING INDIVIDUAL METHOD ACCESS")
        print("=" * 60)
        
        methods_to_test = ['lof_embeddings', 'lof_network', 'lof_mixed', 'isolation_forest', 'ensemble']
        
        for method in methods_to_test:
            try:
                outliers = detector.get_outlier_documents(method, top_k=3)
                print(f"\n{method.upper()} - Top 3 outliers:")
                for _, row in outliers.head(3).iterrows():
                    print(f"  {row['rank']}. {row['document_id']} (score: {row['outlier_score']:.4f})")
            except Exception as e:
                print(f"\nError testing {method}: {e}")
        
        print(f"\n" + "=" * 60)
        print("MULTI-LOF TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during Multi-LOF detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_lof() 