#!/usr/bin/env python3
"""
Simple ASReview Simulation using Python API
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_synergy_dataset(project_root: str, dataset_name: str) -> pd.DataFrame:
    """Load Synergy dataset from CSV file."""
    # Load config
    config_path = os.path.join(project_root, 'data', 'datasets.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    synergy_name = config[dataset_name]['synergy_dataset_name']
    file_path = os.path.join(project_root, 'models', 'synergy_dataset', f'{synergy_name}.csv')
    
    logger.info(f"Loading {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} records")
    
    return df

def simple_simulation(df: pd.DataFrame, max_irrelevant: int = 100, seed: int = 42) -> list:
    """Run a simple random simulation with stopping rule for CONSECUTIVE irrelevant documents."""
    np.random.seed(seed)
    
    labels = df['label_included'].values
    n_docs = len(labels)
    
    # Start with random prior documents
    included_idx = np.where(labels == 1)[0]
    excluded_idx = np.where(labels == 0)[0]
    
    labeled = []
    if len(included_idx) > 0:
        labeled.append(included_idx[0])
    if len(excluded_idx) > 0:
        labeled.append(excluded_idx[0])
    
    consecutive_irrelevant = 0
    iteration = 0
    
    logger.info(f"Starting simulation with {max_irrelevant} CONSECUTIVE irrelevant stopping rule")
    
    while iteration < 5000:  # Max iterations to prevent infinite loop
        # Get unlabeled documents
        unlabeled = [i for i in range(n_docs) if i not in labeled]
        if not unlabeled:
            logger.info("All documents labeled")
            break
        
        # Random selection
        next_doc = np.random.choice(unlabeled)
        true_label = labels[next_doc]
        labeled.append(next_doc)
        iteration += 1
        
        # Track CONSECUTIVE irrelevant documents
        if true_label == 0:  # Irrelevant document
            consecutive_irrelevant += 1
            if iteration % 50 == 0 or consecutive_irrelevant % 10 == 0:
                logger.info(f"Iteration {iteration}: Found irrelevant document, consecutive count: {consecutive_irrelevant}")
        else:  # Relevant document found - reset counter
            if consecutive_irrelevant > 0:
                logger.info(f"Iteration {iteration}: Found RELEVANT document! Resetting consecutive count from {consecutive_irrelevant} to 0")
            consecutive_irrelevant = 0
        
        # Check stopping rule - stop when we hit the consecutive limit
        if consecutive_irrelevant >= max_irrelevant:
            logger.info(f"STOPPING: Found {consecutive_irrelevant} consecutive irrelevant documents (limit: {max_irrelevant})")
            break
        
        if iteration % 100 == 0:
            relevant_found = sum(1 for idx in labeled if labels[idx] == 1)
            logger.info(f"Iteration {iteration}: {relevant_found} relevant found, consecutive irrelevant: {consecutive_irrelevant}")
    
    relevant_found = sum(1 for idx in labeled if labels[idx] == 1)
    irrelevant_found = len(labeled) - relevant_found
    logger.info(f"Simulation done: {len(labeled)} documents labeled ({relevant_found} relevant, {irrelevant_found} irrelevant)")
    logger.info(f"Final consecutive irrelevant count: {consecutive_irrelevant}")
    return labeled

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--max-irrelevant', type=int, default=100)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Load dataset
    df = load_synergy_dataset(project_root, args.dataset)
    
    # Run simulation  
    labeled_indices = simple_simulation(df, args.max_irrelevant)
    
    # Load original dataset for mapping
    orig_path = os.path.join(project_root, 'data', 'simulations', f'{args.dataset}.csv')
    orig_df = pd.read_csv(orig_path, on_bad_lines='skip')
    
    # Find unlabeled
    all_indices = set(range(len(orig_df)))
    unlabeled_indices = all_indices - set(labeled_indices)
    unlabeled_df = orig_df.iloc[list(unlabeled_indices)]
    
    print(f"Dataset: {args.dataset}")
    print(f"Total documents: {len(orig_df)}")
    print(f"Documents labeled: {len(labeled_indices)}")
    print(f"Documents unlabeled: {len(unlabeled_df)}")
    
    if args.output:
        unlabeled_df.to_csv(args.output, index=False)
        print(f"Unlabeled documents saved to: {args.output}")

if __name__ == "__main__":
    main() 