"""
Simple Implementation: Can we detect rank 26 outlier using ranks 1-25?
"""

import pandas as pd
from models.hybrid_models import HybridOutlierDetector, ModelWeights


def main():
    """Simple test: can we detect rank 26 using knowledge of ranks 1-25?"""
    print("Testing Hybrid Outlier Detection: Can we find rank 26?")
    print("=" * 50)
    
    # Load data
    simulation_df = pd.read_csv('data/simulation.csv')
    
    # Setup scenario: only ranks 1-25 are marked as relevant
    training_data = simulation_df.copy()
    training_data['label_included'] = training_data['asreview_ranking'].apply(
        lambda x: 1 if x <= 25 else 0
    )
    
    # Get the rank 26 document info
    rank_26 = simulation_df[simulation_df['asreview_ranking'] == 26].iloc[0]
    target_id = rank_26['openalex_id']
    
    print(f"Target outlier: record_id={rank_26['record_id']}, rank=26")
    print(f"Title: {rank_26['title'][:80]}...")
    
    # Test candidates: documents ranked > 25 (including rank 26)
    test_candidates = simulation_df[simulation_df['asreview_ranking'] > 25]['openalex_id'].tolist()
    
    print(f"\nTesting different weighting strategies on {len(test_candidates)} candidates...")
    
    # Test 1: Default (new) weights - content-heavy
    print(f"\n1. Content-heavy weighting (citation:0.2, content:0.8)")
    detector1 = HybridOutlierDetector()
    detector1.fit(training_data)
    
    scores1 = detector1.predict_relevance_scores(test_candidates)
    target_score1 = scores1.get(target_id, 0.0)
    sorted_candidates1 = sorted(scores1.items(), key=lambda x: x[1], reverse=True)
    target_rank1 = next(i for i, (doc_id, _) in enumerate(sorted_candidates1, 1) if doc_id == target_id)
    
    # Test 2: Content-only
    print(f"\n2. Content-only (citation:0.0, content:1.0)")
    detector2 = HybridOutlierDetector(model_weights=ModelWeights(citation_network=0.0, content_similarity=1.0))
    detector2.fit(training_data)
    
    scores2 = detector2.predict_relevance_scores(test_candidates)
    target_score2 = scores2.get(target_id, 0.0)
    sorted_candidates2 = sorted(scores2.items(), key=lambda x: x[1], reverse=True)
    target_rank2 = next(i for i, (doc_id, _) in enumerate(sorted_candidates2, 1) if doc_id == target_id)
    
    # Test 3: Original balanced weights for comparison
    print(f"\n3. Balanced weighting (citation:0.6, content:0.4)")
    detector3 = HybridOutlierDetector(model_weights=ModelWeights(citation_network=0.6, content_similarity=0.4))
    detector3.fit(training_data)
    
    scores3 = detector3.predict_relevance_scores(test_candidates)
    target_score3 = scores3.get(target_id, 0.0)
    sorted_candidates3 = sorted(scores3.items(), key=lambda x: x[1], reverse=True)
    target_rank3 = next(i for i, (doc_id, _) in enumerate(sorted_candidates3, 1) if doc_id == target_id)
    
    # Get individual model scores for analysis
    citation_scores = detector1.citation_model.predict_relevance_scores([target_id])
    content_scores = detector1.content_model.predict_relevance_scores([target_id])
    target_citation = citation_scores.get(target_id, 0.0)
    target_content = content_scores.get(target_id, 0.0)
    
    # Results comparison
    print(f"\n{'='*60}")
    print(f"INDIVIDUAL MODEL SCORES:")
    print(f"Citation score: {target_citation:.4f}")
    print(f"Content score:  {target_content:.4f}")
    
    print(f"\nCOMBINATION RESULTS:")
    print(f"1. Content-heavy: Score={target_score1:.4f}, Rank={target_rank1}/{len(test_candidates)}")
    print(f"2. Content-only:  Score={target_score2:.4f}, Rank={target_rank2}/{len(test_candidates)}")
    print(f"3. Balanced:      Score={target_score3:.4f}, Rank={target_rank3}/{len(test_candidates)}")
    
    # Performance assessment
    def assess_performance(rank, total):
        if rank <= 10:
            return "🟢 Excellent: Top 10"
        elif rank <= 50:
            return "🟡 Good: Top 50"
        elif rank <= 100:
            return "🟠 Fair: Top 100"
        else:
            return "🔴 Poor: Low ranking"
    
    print(f"\nPERFORMANCE ASSESSMENT:")
    print(f"1. Content-heavy: {assess_performance(target_rank1, len(test_candidates))}")
    print(f"2. Content-only:  {assess_performance(target_rank2, len(test_candidates))}")
    print(f"3. Balanced:      {assess_performance(target_rank3, len(test_candidates))}")
    
    # Recommendation
    best_rank = min(target_rank1, target_rank2, target_rank3)
    if target_rank2 == best_rank:
        print(f"\n🎯 RECOMMENDATION: Use content-only model (specialized text analysis)")
        print(f"   The citation model is hurting performance for this outlier.")
    elif target_rank1 == best_rank:
        print(f"\n🎯 RECOMMENDATION: Use content-heavy weighting")
        print(f"   Small citation contribution helps slightly.")
    else:
        print(f"\n🎯 RECOMMENDATION: Unexpected - balanced approach works best")


if __name__ == "__main__":
    main() 