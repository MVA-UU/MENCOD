"""
Quick Hyperparameter Test

Simple script to quickly test a few key weight combinations for the hybrid model.
"""

import pandas as pd
from models.hybrid_models import HybridOutlierDetector, ModelWeights
from time import time

def quick_test():
    """Test a few key weight combinations quickly."""
    print("=== QUICK HYPERPARAMETER TEST ===")
    
    # Load simulation data
    simulation_df = pd.read_csv('data/simulation.csv')
    
    # Prepare training data (exclude rank 26 outlier)
    training_data = simulation_df.copy()
    training_data['label_included'] = training_data['asreview_ranking'].apply(
        lambda x: 1 if x <= 25 else 0
    )
    
    # Get outlier info
    outlier_row = simulation_df[simulation_df['record_id'] == 497]
    outlier_id = outlier_row.iloc[0]['openalex_id']
    
    # Create search pool
    irrelevant_docs = simulation_df[simulation_df['label_included'] == 0]['openalex_id'].tolist()
    search_pool = [outlier_id] + irrelevant_docs
    
    print(f"Outlier ID: {outlier_id}")
    print(f"Search pool size: {len(search_pool)}")
    
    # Test different weight combinations
    test_weights = [
        ("Citation Only", ModelWeights(citation_network=1.0, content_similarity=0.0)),
        ("Content Only", ModelWeights(citation_network=0.0, content_similarity=1.0)),
        ("Citation Heavy", ModelWeights(citation_network=0.8, content_similarity=0.2)),
        ("Content Heavy", ModelWeights(citation_network=0.2, content_similarity=0.8)),
        ("Balanced", ModelWeights(citation_network=0.5, content_similarity=0.5)),
        ("Current Test", ModelWeights(citation_network=0.1, content_similarity=0.9)),
    ]
    
    results = []
    
    for name, weights in test_weights:
        print(f"\n--- Testing: {name} ---")
        print(f"Citation: {weights.citation_network:.1f}, Content: {weights.content_similarity:.1f}")
        
        start_time = time()
        
        try:
            # Create and fit detector
            detector = HybridOutlierDetector(model_weights=weights)
            detector.fit(training_data)
            
            # Get scores
            scores = detector.predict_relevance_scores(search_pool)
            
            # Find outlier position
            sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            outlier_position = None
            outlier_score = 0.0
            
            for pos, (doc_id, score) in enumerate(sorted_results, 1):
                if doc_id == outlier_id:
                    outlier_position = pos
                    outlier_score = score
                    break
            
            if outlier_position is None:
                outlier_position = len(search_pool)
            
            percentile = ((len(search_pool) - outlier_position) / len(search_pool)) * 100
            eval_time = time() - start_time
            
            results.append({
                'name': name,
                'citation_weight': weights.citation_network,
                'content_weight': weights.content_similarity,
                'position': outlier_position,
                'percentile': percentile,
                'score': outlier_score,
                'time': eval_time
            })
            
            print(f"Position: {outlier_position}/{len(search_pool)} ({percentile:.1f}th percentile)")
            print(f"Score: {outlier_score:.4f}")
            print(f"Time: {eval_time:.1f}s")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Summary
    print(f"\n=== SUMMARY ===")
    results.sort(key=lambda x: x['position'])
    
    print(f"{'Approach':<15} {'Position':<10} {'Percentile':<12} {'Score':<8} {'Time':<6}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['name']:<15} {r['position']:<10} {r['percentile']:<12.1f} {r['score']:<8.4f} {r['time']:<6.1f}s")
    
    # Best result
    if results:
        best = results[0]
        print(f"\nBest approach: {best['name']}")
        print(f"Weights: Citation={best['citation_weight']:.1f}, Content={best['content_weight']:.1f}")
        print(f"Performance: Position {best['position']} ({best['percentile']:.1f}th percentile)")
    
    return results

if __name__ == "__main__":
    quick_test() 