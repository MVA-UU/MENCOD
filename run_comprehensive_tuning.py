"""
Run comprehensive hyperparameter tuning with fine granularity.
"""

from hyperparameter_tuning import HyperparameterTuner
import numpy as np

def main():
    tuner = HyperparameterTuner()
    
    # Grid search with fine granularity around the promising range
    print('Running comprehensive grid search...')
    citation_weights = np.arange(0.0, 1.05, 0.05).tolist()  # 0.0 to 1.0 in steps of 0.05
    results = tuner.grid_search(citation_weights=citation_weights)
    
    # Analyze top results
    print('\n' + '='*50)
    tuner.analyze_results(top_n=15)
    tuner.save_results('grid_search_results.json')
    
    # Also try random search for comparison
    print('\n' + '='*50)
    print('Running random search for comparison...')
    random_results = tuner.random_search(n_iterations=30, seed=42)
    
    if random_results:
        print('\nRandom search results:')
        tuner.analyze_results(top_n=10)

if __name__ == "__main__":
    main() 