"""
Dataset utility functions for ASReview models.

This module provides utilities for loading and processing datasets for ASReview models.
"""

import json
import os
import sys
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    # The project root is one level up from the models directory
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_datasets_config() -> Dict[str, Any]:
    """
    Load datasets configuration from datasets.json.
    
    Returns:
        Dict with dataset configurations
    """
    config_path = os.path.join(get_project_root(), 'data', 'datasets.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def get_available_datasets() -> List[str]:
    """
    Get list of available dataset names.
    
    Returns:
        List of dataset names
    """
    return list(load_datasets_config().keys())


def prompt_dataset_selection() -> str:
    """
    Prompt user to select a dataset.
    
    Returns:
        Selected dataset name
    """
    datasets = get_available_datasets()
    
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets):
        print(f"{i+1}. {dataset}")
    
    while True:
        try:
            selection = int(input("\nSelect dataset (enter number): "))
            if 1 <= selection <= len(datasets):
                return datasets[selection-1]
            else:
                print(f"Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("Please enter a valid number")


def load_dataset(dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load dataset by name.
    
    Args:
        dataset_name: Name of the dataset to load
    
    Returns:
        Tuple of (DataFrame with simulation data, dataset config)
    """
    config = load_datasets_config()
    if dataset_name not in config:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
    
    dataset_config = config[dataset_name]
    simulation_file = os.path.join(get_project_root(), 'data', dataset_config['simulation_file'])
    
    if not os.path.exists(simulation_file):
        raise FileNotFoundError(f"Simulation file not found: {simulation_file}")
    
    df = pd.read_csv(simulation_file)
    return df, dataset_config


def get_outlier_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get outlier information for a dataset.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Dict with outlier information
    """
    config = load_datasets_config()
    if dataset_name not in config:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
    
    return {
        'record_ids': config[dataset_name]['outlier_ids'],
        'synergy_dataset_name': config[dataset_name]['synergy_dataset_name']
    }


def identify_outlier_in_simulation(simulation_df: pd.DataFrame, dataset_config: Dict[str, Any]) -> pd.Series:
    """
    Identify the outlier record in the simulation data.
    
    Args:
        simulation_df: DataFrame with simulation data
        dataset_config: Dataset configuration
    
    Returns:
        Series with outlier record data
    """
    outlier_ids = dataset_config['outlier_ids']
    if not outlier_ids:
        raise ValueError("No outlier IDs specified in dataset config")
    
    # Try to find the outlier by record_id
    outlier_row = simulation_df[simulation_df['record_id'].isin(outlier_ids)]
    
    if outlier_row.empty:
        raise ValueError(f"Outlier with record_id in {outlier_ids} not found in simulation data")
    
    return outlier_row.iloc[0]


def create_training_data(simulation_df: pd.DataFrame, outlier_openalex_id: str) -> pd.DataFrame:
    """
    Create training data excluding the outlier.
    
    Args:
        simulation_df: DataFrame with simulation data
        outlier_openalex_id: OpenAlex ID of the outlier document
    
    Returns:
        DataFrame with training data (outlier excluded)
    """
    training_data = simulation_df.copy()
    
    # Check which columns are available in the dataset
    available_columns = training_data.columns.tolist()
    print(f"Available columns in dataset: {', '.join(available_columns)}")
    
    # Mark all documents as irrelevant by default
    training_data['label_included'] = 0
    
    # Use dataset-specific logic to identify relevant documents
    # Different datasets may have different column structures
    
    # 1. If asreview_prior exists, use it to mark prior knowledge
    if 'asreview_prior' in available_columns:
        prior_mask = training_data['asreview_prior'] == 1
        training_data.loc[prior_mask, 'label_included'] = 1
        print("Used asreview_prior column to identify prior knowledge")
    
    # 2. Mark documents that are labeled as included in the original dataset (except the outlier)
    if 'label_included' in available_columns:
        # Make sure we don't include the outlier in training
        relevant_mask = (simulation_df['label_included'] == 1) & (simulation_df['openalex_id'] != outlier_openalex_id)
        training_data.loc[relevant_mask, 'label_included'] = 1
        print("Used label_included column to identify relevant documents")
    
    # 3. If asreview_label exists, use it as additional source of relevance
    if 'asreview_label' in available_columns:
        relevant_mask = (simulation_df['asreview_label'] == 1) & (simulation_df['openalex_id'] != outlier_openalex_id)
        training_data.loc[relevant_mask, 'label_included'] = 1
        print("Used asreview_label column to identify relevant documents")
    
    # Count relevant documents for reporting
    num_relevant = training_data['label_included'].sum()
    print(f"Training dataset contains {num_relevant} relevant documents (excluding outlier)")
    
    # Sanity check - make sure the outlier is NOT in the training set
    outlier_in_training = training_data[training_data['openalex_id'] == outlier_openalex_id]['label_included'].iloc[0]
    if outlier_in_training == 1:
        print("WARNING: Outlier was incorrectly included in training set - fixing")
        training_data.loc[training_data['openalex_id'] == outlier_openalex_id, 'label_included'] = 0
    
    return training_data


def get_search_pool(simulation_df: pd.DataFrame, outlier_openalex_id: str) -> List[str]:
    """
    Create search pool with the outlier and all irrelevant documents.
    
    Args:
        simulation_df: DataFrame with simulation data
        outlier_openalex_id: OpenAlex ID of the outlier document
    
    Returns:
        List of OpenAlex IDs in the search pool
    """
    # Start with all documents
    all_docs = simulation_df['openalex_id'].tolist()
    
    # Find relevant documents (excluding the outlier)
    relevant_docs = simulation_df[
        (simulation_df['label_included'] == 1) & 
        (simulation_df['openalex_id'] != outlier_openalex_id)
    ]['openalex_id'].tolist()
    
    # Create search pool: outlier + all non-relevant documents
    search_pool = [outlier_openalex_id]
    
    # Add all non-relevant documents
    for doc_id in all_docs:
        if doc_id != outlier_openalex_id and doc_id not in relevant_docs:
            search_pool.append(doc_id)
    
    print(f"Total documents in dataset: {len(all_docs)}")
    print(f"Relevant documents (excluding outlier): {len(relevant_docs)}")
    print(f"Documents in search pool (outlier + irrelevant): {len(search_pool)}")
    
    return search_pool
