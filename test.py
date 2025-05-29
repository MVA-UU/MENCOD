import pandas as pd
import numpy as np
from models.hybrid_models import HybridOutlierDetector, ModelWeights

def test_outlier_retrieval():
    """
    Test if we can actually find the known outlier among irrelevant documents.
    This is the real test of the algorithm's effectiveness.
    """
    print("=== OUTLIER RETRIEVAL TEST ===")
    print("Testing if we can find record_id=497 (ranking #26) among irrelevant documents...")
    
    # Load data
    simulation_df = pd.read_csv('data/simulation.csv')
    
    # CRITICAL FIX: Create training data that excludes rank 26
    # Only include ranks 1-25 as relevant (realistic scenario)
    training_data = simulation_df.copy()
    training_data['label_included'] = training_data['asreview_ranking'].apply(
        lambda x: 1 if x <= 25 else 0
    )
    
    print(f"Training with {training_data['label_included'].sum()} relevant documents (ranks 1-25 only)")
    print("Rank 26 is NOT included in training - this is the outlier we're trying to find")
    
    # Get the known outlier
    outlier_row = simulation_df[simulation_df['record_id'] == 497]
    outlier_id = outlier_row.iloc[0]['openalex_id']
    
    # Get all irrelevant documents
    irrelevant_docs = simulation_df[simulation_df['label_included'] == 0]
    print(f"Total irrelevant documents: {len(irrelevant_docs)}")
    
    # Create the search pool: outlier + all irrelevant documents
    search_pool = [outlier_id] + irrelevant_docs['openalex_id'].tolist()
    print(f"Search pool size: {len(search_pool)} documents")
    
    # Test different approaches using current hybrid model
    approaches = {
        'Original Hybrid (3 models)': ModelWeights(citation_network=0.33, content_similarity=0.33, confidence_calibration=0.34, minilm_embeddings=0.0),
        'Balanced (All Four)': ModelWeights(citation_network=0.25, content_similarity=0.25, confidence_calibration=0.25, minilm_embeddings=0.25),
        'MiniLM Focused': ModelWeights(citation_network=0.2, content_similarity=0.2, confidence_calibration=0.2, minilm_embeddings=0.4),
        'MiniLM Heavy': ModelWeights(citation_network=0.15, content_similarity=0.15, confidence_calibration=0.2, minilm_embeddings=0.5),
        'Content + MiniLM': ModelWeights(citation_network=0.2, content_similarity=0.4, confidence_calibration=0.0, minilm_embeddings=0.4),
    }
    
    results = {}
    
    for approach_name, weights in approaches.items():
        print(f"\n--- Testing: {approach_name} ---")
        
        # Use hybrid detector with specified weights
        detector = HybridOutlierDetector(model_weights=weights)
        detector.fit(training_data)  # Use training data without rank 26
        scores = detector.predict_relevance_scores(search_pool)
        
        # Sort documents by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Find outlier position
        outlier_position = None
        for i, (doc_id, score) in enumerate(sorted_results):
            if doc_id == outlier_id:
                outlier_position = i + 1  # 1-indexed
                outlier_score = score
                break
        
        results[approach_name] = {
            'outlier_position': outlier_position,
            'outlier_score': outlier_score,
            'total_documents': len(search_pool)
        }
        
        print(f"Outlier found at position: {outlier_position} out of {len(search_pool)}")
        print(f"Outlier score: {outlier_score:.4f}")
        print(f"Percentile: {((len(search_pool) - outlier_position) / len(search_pool)) * 100:.2f}th percentile")
        
        # Show top 10 results
        print("Top 10 highest scoring documents:")
        for i, (doc_id, score) in enumerate(sorted_results[:10]):
            marker = " *** OUTLIER ***" if doc_id == outlier_id else ""
            print(f"  {i+1}. Score: {score:.4f}{marker}")
    
    # Summary comparison
    print(f"\n=== SUMMARY COMPARISON ===")
    for approach, result in results.items():
        pos = result['outlier_position']
        total = result['total_documents']
        percentile = ((total - pos) / total) * 100
        print(f"{approach}:")
        print(f"  Position: {pos}/{total} ({percentile:.1f}th percentile)")
        print(f"  Score: {result['outlier_score']:.4f}")
    
    return results

def test_stopping_rule_scenario():
    """
    Test more realistic scenario: can we find the outlier after a stopping rule?
    This simulates finding documents that would be missed by normal ASReview.
    """
    print(f"\n=== STOPPING RULE SCENARIO TEST ===")
    
    # Load data
    simulation_df = pd.read_csv('data/simulation.csv')
    
    # CRITICAL FIX: Create training data that excludes rank 26 outlier
    training_data = simulation_df.copy()
    training_data['label_included'] = training_data['asreview_ranking'].apply(
        lambda x: 1 if x <= 25 else 0
    )
    
    # Simulate stopping rule: find first 25 relevant documents, then stop
    relevant_docs = simulation_df[simulation_df['label_included'] == 1].sort_values('asreview_ranking')
    found_relevant = relevant_docs[relevant_docs['asreview_ranking'] < 26]  # First 25
    missed_outlier = relevant_docs[relevant_docs['asreview_ranking'] == 26]  # The outlier
    
    print(f"Found relevant documents: {len(found_relevant)}")
    print(f"Missed outlier: {missed_outlier.iloc[0]['record_id']} (ranking {missed_outlier.iloc[0]['asreview_ranking']})")
    
    # Get remaining documents that would be checked by hybrid model
    # These are documents ranked worse than where we stopped
    remaining_docs = simulation_df[simulation_df['asreview_ranking'] > 25]
    print(f"Remaining documents to search: {len(remaining_docs)}")
    
    # Test if hybrid model can find outlier in remaining documents
    outlier_id = missed_outlier.iloc[0]['openalex_id']
    search_candidates = remaining_docs['openalex_id'].tolist()
    
    # Use hybrid approach with optimal weights from tuning
    print("Testing hybrid approach on remaining documents...")
    
    # Fit models on the "found" relevant documents (simulating real scenario)
    detector = HybridOutlierDetector(model_weights=ModelWeights(citation_network=0.15, content_similarity=0.15, confidence_calibration=0.2, minilm_embeddings=0.5))
    detector.fit(training_data)  # Use training data without rank 26
    
    # Score remaining documents
    combined_scores = detector.predict_relevance_scores(search_candidates)
    
    # Sort and find outlier
    sorted_remaining = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    outlier_position_remaining = None
    for i, (doc_id, score) in enumerate(sorted_remaining):
        if doc_id == outlier_id:
            outlier_position_remaining = i + 1
            outlier_score_remaining = score
            break
    
    print(f"Outlier position in remaining documents: {outlier_position_remaining} out of {len(search_candidates)}")
    print(f"Would need to check top {outlier_position_remaining} documents to find the outlier")
    
    # Practical question: If we check top 100 documents, would we find it?
    top_100_ids = [doc_id for doc_id, _ in sorted_remaining[:100]]
    found_in_top_100 = outlier_id in top_100_ids
    
    top_50_ids = [doc_id for doc_id, _ in sorted_remaining[:50]]
    found_in_top_50 = outlier_id in top_50_ids
    
    print(f"Found in top 50: {'YES' if found_in_top_50 else 'NO'}")
    print(f"Found in top 100: {'YES' if found_in_top_100 else 'NO'}")
    
    return {
        'position_in_remaining': outlier_position_remaining,
        'total_remaining': len(search_candidates),
        'found_in_top_50': found_in_top_50,
        'found_in_top_100': found_in_top_100,
        'score': outlier_score_remaining
    }

def analyze_score_distribution():
    """
    Analyze the score distribution to understand outlier detection performance.
    """
    print(f"\n=== SCORE DISTRIBUTION ANALYSIS ===")
    
    # Load data
    simulation_df = pd.read_csv('data/simulation.csv')
    
    # CRITICAL FIX: Create training data that excludes rank 26 outlier
    training_data = simulation_df.copy()
    training_data['label_included'] = training_data['asreview_ranking'].apply(
        lambda x: 1 if x <= 25 else 0
    )
    
    # Setup hybrid model with optimal weights
    detector = HybridOutlierDetector(model_weights=ModelWeights(citation_network=0.15, content_similarity=0.15, confidence_calibration=0.2, minilm_embeddings=0.5))
    detector.fit(training_data)  # Use training data without rank 26
    
    # Get different document groups
    outlier_row = simulation_df[simulation_df['record_id'] == 497]
    outlier_id = outlier_row.iloc[0]['openalex_id']
    
    # Get OTHER relevant docs (excluding the outlier we're trying to find)
    relevant_docs = simulation_df[simulation_df['asreview_ranking'] <= 25]  # Only ranks 1-25
    other_relevant_ids = relevant_docs['openalex_id'].tolist()
    
    # Sample irrelevant documents for analysis
    irrelevant_docs = simulation_df[simulation_df['label_included'] == 0]
    irrelevant_sample = irrelevant_docs.sample(n=min(1000, len(irrelevant_docs)), random_state=42)
    irrelevant_ids = irrelevant_sample['openalex_id'].tolist()
    
    # Get scores for all groups
    all_docs = [outlier_id] + other_relevant_ids + irrelevant_ids
    
    combined_scores = detector.predict_relevance_scores(all_docs)
    
    # Analyze distributions
    outlier_score = combined_scores[outlier_id]
    other_relevant_scores = [combined_scores[doc_id] for doc_id in other_relevant_ids]
    irrelevant_scores = [combined_scores[doc_id] for doc_id in irrelevant_ids]
    
    print(f"Score Analysis:")
    print(f"Outlier score: {outlier_score:.4f}")
    print(f"Other relevant scores - Mean: {np.mean(other_relevant_scores):.4f}, Std: {np.std(other_relevant_scores):.4f}")
    print(f"Irrelevant scores - Mean: {np.mean(irrelevant_scores):.4f}, Std: {np.std(irrelevant_scores):.4f}")
    
    # Check percentiles
    all_scores = other_relevant_scores + irrelevant_scores
    outlier_percentile = (np.sum(np.array(all_scores) < outlier_score) / len(all_scores)) * 100
    print(f"Outlier percentile among all documents: {outlier_percentile:.1f}th percentile")
    
    # Check how many irrelevant documents score higher than outlier
    higher_scoring_irrelevant = np.sum(np.array(irrelevant_scores) > outlier_score)
    print(f"Irrelevant documents scoring higher than outlier: {higher_scoring_irrelevant} out of {len(irrelevant_scores)}")

if __name__ == "__main__":
    # Run comprehensive outlier detection test
    retrieval_results = test_outlier_retrieval()
    stopping_rule_results = test_stopping_rule_scenario()
    analyze_score_distribution()
    
    # Final assessment
    print(f"\n=== FINAL ASSESSMENT ===")
    print("Can the algorithm find the outlier? Summary:")
    
    for approach, result in retrieval_results.items():
        pos = result['outlier_position']
        total = result['total_documents']
        success = "GOOD" if pos <= 100 else "POOR"
        print(f"{approach}: Position {pos}/{total} - {success}")
    
    if stopping_rule_results['found_in_top_50']:
        print("✅ PRACTICAL SUCCESS: Outlier found in top 50 after stopping rule")
    elif stopping_rule_results['found_in_top_100']:
        print("⚠️  MODERATE SUCCESS: Outlier found in top 100 after stopping rule")
    else:
        print("❌ NEEDS IMPROVEMENT: Outlier not found in top 100")