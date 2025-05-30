#!/usr/bin/env python
"""
Debug script to analyze why the outlier in Jeyaraman dataset is ranking at the bottom.
"""

import pandas as pd
import numpy as np
import os
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Fix import for direct script execution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dataset_utils import load_dataset, identify_outlier_in_simulation

def debug_jeyaraman_outlier():
    """Debug function to analyze the Jeyaraman outlier issue."""
    print("Debugging Jeyaraman outlier ranking issue...\n")
    
    # 1. Load the dataset
    dataset_name = "jeyaraman"
    simulation_df, dataset_config = load_dataset(dataset_name)
    print(f"Loaded {len(simulation_df)} documents from simulation")
    
    # 2. Identify the outlier
    outlier_row = identify_outlier_in_simulation(simulation_df, dataset_config)
    outlier_id = outlier_row['openalex_id']
    print(f"Outlier: {outlier_id} (Record ID: {outlier_row['record_id']})")
    
    # 3. Get relevant documents (excluding outlier)
    relevant_docs = simulation_df[
        (simulation_df['label_included'] == 1) & 
        (simulation_df['openalex_id'] != outlier_id)
    ]
    print(f"Found {len(relevant_docs)} relevant documents (excluding outlier)")
    
    # 4. Get a random sample of irrelevant documents
    irrelevant_docs = simulation_df[simulation_df['label_included'] == 0]
    random_irrelevant = irrelevant_docs.sample(min(20, len(irrelevant_docs)))
    print(f"Selected {len(random_irrelevant)} random irrelevant documents for comparison")
    
    # 5. Load the embedding model
    print("Loading MiniLM model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("MiniLM model loaded successfully!")
    
    # 6. Create text for outlier
    outlier_title = str(outlier_row.get('title', '') or '')
    outlier_abstract = str(outlier_row.get('abstract', '') or '')
    outlier_text = f"{outlier_title}. {outlier_abstract}".strip()
    print("\n--- OUTLIER DOCUMENT TEXT ---")
    print(f"Title: {outlier_title}")
    print(f"Abstract length: {len(outlier_abstract)} characters")
    print(f"First 100 chars: {outlier_abstract[:100]}...")
    
    # 7. Create embeddings for relevant docs, irrelevant docs, and outlier
    print("\nCreating embeddings...")
    
    # Prepare texts for relevant docs
    relevant_texts = []
    for _, row in relevant_docs.iterrows():
        title = str(row.get('title', '') or '')
        abstract = str(row.get('abstract', '') or '')
        combined_text = f"{title}. {abstract}".strip()
        if combined_text and len(combined_text) > 10:
            relevant_texts.append(combined_text)
        else:
            relevant_texts.append("No text available")
    
    # Prepare texts for irrelevant docs
    irrelevant_texts = []
    for _, row in random_irrelevant.iterrows():
        title = str(row.get('title', '') or '')
        abstract = str(row.get('abstract', '') or '')
        combined_text = f"{title}. {abstract}".strip()
        if combined_text and len(combined_text) > 10:
            irrelevant_texts.append(combined_text)
        else:
            irrelevant_texts.append("No text available")
    
    # Create embeddings
    relevant_embeddings = model.encode(relevant_texts, convert_to_numpy=True)
    irrelevant_embeddings = model.encode(irrelevant_texts, convert_to_numpy=True)
    outlier_embedding = model.encode([outlier_text], convert_to_numpy=True)[0]
    
    # 8. Calculate similarity distributions
    print("\n--- SIMILARITY ANALYSIS ---")
    
    # Similarity to relevant docs
    similarities_to_relevant = cosine_similarity(
        outlier_embedding.reshape(1, -1), 
        relevant_embeddings
    )[0]
    
    # Similarity to irrelevant docs
    similarities_to_irrelevant = cosine_similarity(
        outlier_embedding.reshape(1, -1), 
        irrelevant_embeddings
    )[0]
    
    print(f"Similarities to relevant docs:")
    print(f"  Max: {np.max(similarities_to_relevant):.4f}")
    print(f"  Mean: {np.mean(similarities_to_relevant):.4f}")
    print(f"  Min: {np.min(similarities_to_relevant):.4f}")
    print(f"  Std: {np.std(similarities_to_relevant):.4f}")
    
    print(f"\nSimilarities to irrelevant docs:")
    print(f"  Max: {np.max(similarities_to_irrelevant):.4f}")
    print(f"  Mean: {np.mean(similarities_to_irrelevant):.4f}")
    print(f"  Min: {np.min(similarities_to_irrelevant):.4f}")
    print(f"  Std: {np.std(similarities_to_irrelevant):.4f}")
    
    # 9. Check if outlier is truly at the bottom when compared to irrelevant docs
    print("\n--- RANKING CHECK ---")
    
    # Combine relevant and irrelevant for comparison
    all_embeddings = np.vstack([relevant_embeddings, irrelevant_embeddings])
    all_similarities = cosine_similarity(
        outlier_embedding.reshape(1, -1), 
        all_embeddings
    )[0]
    
    # Get document types for reference
    doc_types = ["relevant"] * len(relevant_embeddings) + ["irrelevant"] * len(irrelevant_embeddings)
    
    # Sort by similarity
    similarity_ranking = list(zip(all_similarities, doc_types))
    similarity_ranking.sort(reverse=True)
    
    # Report distribution
    top_10_types = [doc_type for _, doc_type in similarity_ranking[:10]]
    top_10_count = top_10_types.count("relevant")
    print(f"Top 10 most similar documents to outlier: {top_10_count} relevant, {10-top_10_count} irrelevant")
    
    bottom_10_types = [doc_type for _, doc_type in similarity_ranking[-10:]]
    bottom_10_count = bottom_10_types.count("irrelevant")
    print(f"Bottom 10 least similar documents to outlier: {10-bottom_10_count} relevant, {bottom_10_count} irrelevant")
    
    # 10. Analysis of similarity percentiles
    print("\n--- PERCENTILE ANALYSIS ---")
    
    # Find where outlier would rank among ALL documents
    all_docs = simulation_df.copy()
    all_texts = []
    
    # Get all documents except outlier for embedding
    for _, row in all_docs[all_docs['openalex_id'] != outlier_id].iterrows():
        title = str(row.get('title', '') or '')
        abstract = str(row.get('abstract', '') or '')
        combined_text = f"{title}. {abstract}".strip()
        if combined_text and len(combined_text) > 10:
            all_texts.append(combined_text)
        else:
            all_texts.append("No text available")
    
    # Take a random sample to speed up computation if needed
    sample_size = min(200, len(all_texts))
    random_indices = random.sample(range(len(all_texts)), sample_size)
    sample_texts = [all_texts[i] for i in random_indices]
    
    # Create embeddings
    sample_embeddings = model.encode(sample_texts, convert_to_numpy=True)
    
    # Calculate similarities
    sample_similarities = cosine_similarity(
        outlier_embedding.reshape(1, -1), 
        sample_embeddings
    )[0]
    
    # Sort and find percentile
    sorted_similarities = sorted(sample_similarities, reverse=True)
    outlier_percentile = (len(sorted_similarities) - 1) * 100 / len(sorted_similarities)
    
    print(f"Among a random sample of {sample_size} documents (excluding outlier):")
    print(f"Outlier's similarity would rank at approximately the {outlier_percentile:.1f}th percentile")
    print(f"Max similarity in sample: {max(sorted_similarities):.4f}")
    print(f"Min similarity in sample: {min(sorted_similarities):.4f}")
    
    # 11. Check for truncation or data issues
    print("\n--- DATA QUALITY CHECK ---")
    
    # Check for truncation in the content
    expected_abstract_length = len(outlier_abstract)
    expected_title_length = len(outlier_title)
    
    original_row = simulation_df[simulation_df['openalex_id'] == outlier_id].iloc[0]
    original_title = str(original_row.get('title', '') or '')
    original_abstract = str(original_row.get('abstract', '') or '')
    
    title_match = original_title == outlier_title
    abstract_match = original_abstract == outlier_abstract
    
    print(f"Title data integrity check: {'PASS' if title_match else 'FAIL'}")
    print(f"Abstract data integrity check: {'PASS' if abstract_match else 'FAIL'}")
    
    # Check for non-standard characters or encoding issues
    unusual_chars_title = [c for c in outlier_title if ord(c) > 127]
    unusual_chars_abstract = [c for c in outlier_abstract if ord(c) > 127]
    
    print(f"Non-ASCII characters in title: {len(unusual_chars_title)}")
    print(f"Non-ASCII characters in abstract: {len(unusual_chars_abstract)}")
    
    # 12. Additional insight - most similar documents to outlier
    print("\n--- MOST SIMILAR DOCUMENTS TO OUTLIER ---")
    
    # Find top 3 most similar relevant documents
    top_indices = np.argsort(similarities_to_relevant)[-3:][::-1]
    
    for i, idx in enumerate(top_indices):
        row = relevant_docs.iloc[idx]
        similarity = similarities_to_relevant[idx]
        print(f"\nRelevant document #{i+1} - Similarity: {similarity:.4f}")
        print(f"Title: {row.get('title', '')}")
        print(f"Abstract (first 100 chars): {str(row.get('abstract', ''))[:100]}...")

    # 13. Check for implementation issues in main scoring functions
    print("\n--- IMPLEMENTATION CHECK ---")
    # Simulate how similarity is calculated in the main implementation
    outlier_centroid_similarity = cosine_similarity(
        outlier_embedding.reshape(1, -1), 
        np.mean(relevant_embeddings, axis=0).reshape(1, -1)
    )[0, 0]
    
    print(f"Centroid similarity calculation check: {outlier_centroid_similarity:.4f}")
    
    # Calculate the raw max similarity as used in many scoring methods
    raw_max_similarity = np.max(similarities_to_relevant)
    print(f"Raw max similarity calculation check: {raw_max_similarity:.4f}")
    
    # 14. Conclusion
    print("\n--- CONCLUSION ---")
    if np.max(similarities_to_irrelevant) > np.max(similarities_to_relevant):
        print("ISSUE DETECTED: Outlier is more similar to some irrelevant docs than to any relevant docs!")
    else:
        print("Outlier is more similar to relevant docs than to irrelevant docs, as expected.")
        
    if np.mean(similarities_to_irrelevant) > np.mean(similarities_to_relevant):
        print("ISSUE DETECTED: Outlier is more similar to irrelevant docs on average!")
    else:
        print("Outlier is more similar to relevant docs than irrelevant docs on average, as expected.")

if __name__ == "__main__":
    debug_jeyaraman_outlier() 