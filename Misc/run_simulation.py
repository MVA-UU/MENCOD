#!/usr/bin/env python3
"""
ASReview Simulation Runner

This script runs ASReview simulations using the API to test outlier detection performance.
It uses datasets from the Synergy dataset library and applies a stopping rule of 100 
consecutive irrelevant documents.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import argparse

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils import (
    load_datasets_config, 
    prompt_dataset_selection,
    get_available_datasets,
    load_simulation_data,
    load_synergy_dataset as utils_load_synergy_dataset
)

# ASReview imports
try:
    import asreview as asr
    from asreview.models.stoppers import NConsecutiveIrrelevant
    from asreview.models.classifiers import NaiveBayes, SVM, RandomForest, Logistic
    from asreview.models.feature_extractors import Tfidf
    from asreview.models.queriers import Max
    from asreview.models.balancers import Balanced
except ImportError as e:
    print(f"Error importing ASReview: {e}")
    print("Please make sure ASReview is installed: pip install asreview")
    sys.exit(1)

# ASReview Insights imports (optional)
try:
    from asreviewcontrib.insights.metrics import time_to_discovery
    INSIGHTS_AVAILABLE = True
except ImportError:
    INSIGHTS_AVAILABLE = False
    logger.warning("ASReview Insights not available. Install with: pip install asreview-insights")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ASReviewSimulationRunner:
    """Class for running ASReview simulations with specified configurations."""
    
    def __init__(self):
        self.datasets_config = load_datasets_config()
        self.output_dir = os.path.join(project_root, 'simulation_exports')
        self._ensure_output_directory()
    
    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def list_available_datasets(self):
        """Display available datasets and their information."""
        print("\nAvailable datasets for simulation:")
        print("="*60)
        
        for i, dataset_name in enumerate(get_available_datasets(), 1):
            config = self.datasets_config[dataset_name]
            synergy_name = config.get('synergy_dataset_name', 'Unknown')
            outlier_ids = config.get('outlier_ids', [])
            
            print(f"{i}. {dataset_name}")
            print(f"   Synergy Dataset: {synergy_name}")
            print(f"   Known Outliers: {len(outlier_ids)}")
            print()
    
    def load_synergy_dataset(self, dataset_name: str):
        """
        Load a dataset from the SYNERGY collection using synergy-dataset package.
        
        Args:
            dataset_name: Name of the dataset from our configuration
            
        Returns:
            ASReview dataset object
        """
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        
        synergy_name = self.datasets_config[dataset_name]['synergy_dataset_name']
        logger.info(f"Loading Synergy dataset: {synergy_name}")
        
        try:
            # Import synergy_dataset package
            from synergy_dataset import Dataset
            
            # Load dataset using synergy_dataset package
            synergy_dataset = Dataset(synergy_name)
            df = synergy_dataset.to_frame()
            
            # Get the OpenAlex IDs from the synergy dataset
            synergy_dict = synergy_dataset.to_dict()
            openalex_ids = list(synergy_dict.keys())
            
            # Add record_id and openalex_id columns to the DataFrame
            df['record_id'] = range(len(df))
            df['openalex_id'] = openalex_ids
            
            # Store the original DataFrame with OpenAlex IDs for later use in exports
            self._synergy_df_with_openalex = df.copy()
            
            logger.info(f"Successfully loaded dataset: {synergy_name}")
            logger.info(f"Dataset shape: {len(df)} records")
            logger.info(f"Dataset columns: {list(df.columns)}")
            logger.info(f"Sample OpenAlex IDs: {openalex_ids[:3]}")
            
            # Convert to ASReview dataset format by creating a temporary CSV
            import tempfile
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                    temp_file_path = tmp_file.name
                    df.to_csv(tmp_file.name, index=False)
                
                # Load the dataset from the temporary file
                dataset = asr.load_dataset(temp_file_path)
                
            finally:
                # Clean up the temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass  # Ignore cleanup errors
            
            return dataset
            
        except ImportError:
            logger.error("synergy_dataset package not found. Please install it with: pip install synergy-dataset")
            raise
        except Exception as e:
            logger.error(f"Failed to load dataset {synergy_name}: {e}")
            raise
    
    def setup_simulation_config(self, classifier_type: str = 'nb', random_state: int = 42):
        """
        Set up the simulation configuration with models and stopping rule.
        
        Args:
            classifier_type: Type of classifier to use ('nb', 'svm', 'rf', 'logistic')
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with simulation configuration
        """
        # Map classifier types to classes with random state where supported
        classifier_map = {
            'nb': NaiveBayes(),
            'svm': SVM(random_state=random_state),
            'rf': RandomForest(random_state=random_state),
            'logistic': Logistic(random_state=random_state)
        }
        
        if classifier_type not in classifier_map:
            logger.warning(f"Unknown classifier type '{classifier_type}', using Naive Bayes")
            classifier_type = 'nb'
        
        config = {
            'classifier': classifier_map[classifier_type],
            'feature_extractor': Tfidf(),
            'querier': Max(),
            'balancer': Balanced(),
            'stopper': NConsecutiveIrrelevant(100)  # Stop after 100 consecutive irrelevant
        }
        
        logger.info(f"Simulation configuration:")
        logger.info(f"  Classifier: {classifier_type}")
        logger.info(f"  Feature Extractor: TF-IDF")
        logger.info(f"  Querier: Max")
        logger.info(f"  Balancer: Balanced")
        logger.info(f"  Stopper: 100 consecutive irrelevant")
        
        return config
    
    def run_simulation(self, dataset_name: str, classifier_type: str = 'nb', 
                      random_state: int = 42):
        """
        Run an ASReview simulation on the specified dataset.
        
        Args:
            dataset_name: Name of the dataset to simulate
            classifier_type: Type of classifier to use
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Starting simulation for dataset: {dataset_name}")
        
        # Load dataset
        dataset = self.load_synergy_dataset(dataset_name)
        
        # Convert to DataFrame for easier manipulation
        dataset_df = dataset.get_df()
        logger.info(f"Dataset contains {len(dataset_df)} documents")
        
        # Debug: Check what columns are available
        logger.info(f"Dataset columns: {list(dataset_df.columns)}")
        logger.info(f"First few rows:\n{dataset_df.head()}")
        
        # Check for different possible label column names
        label_column = None
        for col in ['included', 'label_included', 'label', 'relevant']:
            if col in dataset_df.columns:
                label_column = col
                break
        
        if label_column is None:
            raise ValueError("No suitable label column found in dataset")
        
        logger.info(f"Using label column: {label_column}")
        logger.info(f"Relevant documents: {dataset_df[label_column].sum()}")
        logger.info(f"Label distribution:\n{dataset_df[label_column].value_counts()}")
        
        # Additional debugging for labels
        unique_labels = dataset_df[label_column].unique()
        logger.info(f"Unique label values: {unique_labels}")
        logger.info(f"Label data type: {dataset_df[label_column].dtype}")
        
        # Ensure labels are binary integers
        if len(unique_labels) != 2 or not all(label in [0, 1] for label in unique_labels):
            logger.error(f"Labels are not binary (0/1). Found: {unique_labels}")
            raise ValueError("Labels must be binary (0 for irrelevant, 1 for relevant)")
        
        # Convert to proper binary labels
        labels_binary = dataset_df[label_column].astype(int)
        logger.info(f"Converted labels - unique values: {labels_binary.unique()}")
        
        # Prepare features and labels
        X = dataset_df[['title', 'abstract']].fillna('')
        X = X['title'] + ' ' + X['abstract']  # Combine title and abstract
        y = dataset_df[label_column].astype(int)
        
        # Set up simulation configuration
        config = self.setup_simulation_config(classifier_type, random_state)
        
        try:
            # Initialize active learning cycle with the configured models
            logger.info("Initializing ASReview active learning cycle...")
            
            # Create ActiveLearningCycle with the configured models
            learning_cycle = asr.ActiveLearningCycle(
                querier=config['querier'],
                classifier=config['classifier'],
                feature_extractor=config['feature_extractor'],
                balancer=config['balancer']
            )
            
            # Initialize simulation with the active learning cycle and stopper
            logger.info("Initializing ASReview simulation...")
            simulation = asr.Simulate(
                dataset_df,  # Pass the full dataset DataFrame
                labels_binary,  # Pass the binary labels
                learning_cycle,  # Pass the single learning cycle
                stopper=config['stopper']  # Pass stopper to Simulate constructor
            )
            
            # Add initial priors to bootstrap the simulation
            logger.info("Adding initial priors...")
            # Find some relevant and irrelevant documents to start with
            relevant_indices = dataset_df[dataset_df[label_column] == 1].index[:2].tolist()
            irrelevant_indices = dataset_df[dataset_df[label_column] == 0].index[:3].tolist()
            initial_indices = relevant_indices + irrelevant_indices
            
            # Label these initial documents
            simulation.label(initial_indices)
            logger.info(f"Added {len(initial_indices)} initial labels ({len(relevant_indices)} relevant, {len(irrelevant_indices)} irrelevant)")
            
            logger.info("Running simulation...")
            
            # Run the simulation
            simulation.review()
            
            logger.info("Simulation completed successfully")
            
            # Get results from simulation
            results_df = simulation._results
            
            # Extract simulation statistics
            simulation_stats = {
                'dataset_name': dataset_name,
                'synergy_dataset_name': self.datasets_config[dataset_name]['synergy_dataset_name'],
                'classifier_type': classifier_type,
                'random_state': random_state,
                'total_documents': len(dataset_df),
                'total_relevant': int(y.sum()),
                'total_irrelevant': int(len(y) - y.sum()),
                'documents_reviewed': len(results_df),
                'relevant_found': int(results_df['label'].sum()),
                'stopping_rule_triggered': True,  # Since we use NConsecutiveIrrelevant
                'timestamp': datetime.now().isoformat()
            }
            
            # Debug: Check what the ASReview simulation results contain
            logger.info(f"ASReview simulation results type: {type(simulation._results)}")
            logger.info(f"ASReview simulation results columns: {list(simulation._results.columns)}")
            logger.info(f"ASReview simulation results shape: {simulation._results.shape}")
            logger.info(f"First few rows of results:")
            logger.info(f"{simulation._results.head()}")
            
            # Create detailed results DataFrame with original dataset information
            enriched_results_df = self._create_enriched_results_dataframe(
                dataset_df, simulation._results, simulation_stats
            )
            
            logger.info(f"Simulation statistics:")
            logger.info(f"  Documents reviewed: {simulation_stats['documents_reviewed']}")
            logger.info(f"  Relevant found: {simulation_stats['relevant_found']}")
            logger.info(f"  Coverage: {simulation_stats['relevant_found'] / simulation_stats['total_relevant']:.2%}")
            
            return {
                'stats': simulation_stats,
                'results_df': enriched_results_df,
                'simulation_object': simulation
            }
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
    
    def _create_enriched_results_dataframe(self, dataset_df: pd.DataFrame, 
                                          simulation_results, 
                                          stats: dict) -> pd.DataFrame:
        """
        Create a detailed results DataFrame by combining original dataset with simulation results.
        
        Args:
            dataset_df: Original dataset DataFrame
            simulation_results: ASReview simulation results object
            stats: Simulation statistics dictionary
            
        Returns:
            DataFrame with detailed simulation results
        """
        # Create enriched results DataFrame
        results_df = dataset_df.copy()
        
        # Add simulation-specific columns
        results_df['reviewed'] = False
        results_df['review_order'] = -1
        results_df['predicted_score'] = 0.0
        results_df['found_before_stopping'] = False
        
        # simulation_results is a DataFrame where each row represents a document that was queried
        # The index of the row represents the order in which it was queried
        # The 'record_id' column tells us which document index in the dataset was queried
        for review_idx, (_, row) in enumerate(simulation_results.iterrows()):
            doc_index = row['record_id']  # This should be the index in the original dataset
            
            if doc_index < len(results_df):
                results_df.loc[doc_index, 'reviewed'] = True
                results_df.loc[doc_index, 'review_order'] = review_idx
                results_df.loc[doc_index, 'found_before_stopping'] = True
                
                # Check if this document was labeled as relevant during simulation
                # The 'label' column in simulation results should contain the labels assigned
                if 'label' in row and row['label'] == 1:
                    logger.info(f"Found relevant document at dataset index {doc_index}, review order {review_idx}")
                # Also check ground truth to verify
                elif results_df.loc[doc_index, 'included'] == 1:
                    logger.info(f"Ground truth relevant document at dataset index {doc_index}, review order {review_idx}")
        
        # Count how many relevant documents were found during simulation
        found_relevant = results_df[(results_df['reviewed'] == True) & (results_df['included'] == 1)]
        logger.info(f"Total relevant documents found during simulation: {len(found_relevant)}")
        
        # Add metadata columns
        for key, value in stats.items():
            results_df[f'sim_{key}'] = value
        
        # Sort by review order (reviewed documents first, then original order)
        results_df = results_df.sort_values(['reviewed', 'review_order'], 
                                          ascending=[False, True])
        
        return results_df
    
    def export_results(self, results: dict, dataset_name: str) -> str:
        """
        Export simulation results to CSV file.
        
        Args:
            results: Dictionary with simulation results
            dataset_name: Name of the dataset
            
        Returns:
            Path to the exported CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        classifier_type = results['stats']['classifier_type']
        
        filename = f"{dataset_name}_{classifier_type}_simulation_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Export main results
        results['results_df'].to_csv(filepath, index=False)
        
        # Export summary statistics
        stats_filename = f"{dataset_name}_{classifier_type}_stats_{timestamp}.json"
        stats_filepath = os.path.join(self.output_dir, stats_filename)
        
        import json
        with open(stats_filepath, 'w') as f:
            json.dump(results['stats'], f, indent=2)
        
        logger.info(f"Results exported to: {filepath}")
        logger.info(f"Statistics exported to: {stats_filepath}")
        
        return filepath
    
    def print_simulation_summary(self, results: dict):
        """Print a summary of the simulation results."""
        stats = results['stats']
        
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        print(f"Dataset: {stats['dataset_name']}")
        print(f"Synergy Dataset: {stats['synergy_dataset_name']}")
        print(f"Classifier: {stats['classifier_type']}")
        print(f"Random State: {stats['random_state']}")
        print()
        print(f"Total Documents: {stats['total_documents']:,}")
        print(f"Total Relevant: {stats['total_relevant']:,}")
        print(f"Total Irrelevant: {stats['total_irrelevant']:,}")
        print()
        print(f"Documents Reviewed: {stats['documents_reviewed']:,}")
        print(f"Relevant Found: {stats['relevant_found']:,}")
        print(f"Coverage: {stats['relevant_found'] / stats['total_relevant']:.2%}")
        print(f"Review Efficiency: {stats['documents_reviewed'] / stats['total_documents']:.2%}")
        print()
        
        # Check for known outliers
        dataset_config = self.datasets_config.get(stats['dataset_name'], {})
        outlier_ids = dataset_config.get('outlier_ids', [])
        
        if outlier_ids:
            print(f"Known Outliers: {len(outlier_ids)}")
            # Note: More detailed outlier analysis would require mapping record_id to dataset indices
        
        print("="*60)
    
    def export_leftover_documents(self, results: dict, dataset_name: str) -> str:
        """
        Export relevant documents + unreviewed documents (leftover after stopping rule).
        Format: record_id,openalex_id,doi,title,abstract,label_included
        
        Includes:
        - Documents found relevant during simulation (label_included=1)
        - Documents NOT reviewed during simulation - leftover after stopping rule (label_included=0)
        
        Excludes:
        - Documents reviewed but found irrelevant during simulation
        
        Args:
            results: Dictionary with simulation results
            dataset_name: Name of the dataset
            
        Returns:
            Path to the exported leftover documents CSV file
        """
        try:
            # Get the results from simulation
            results_df = results['results_df']
            logger.info(f"Results dataframe shape: {results_df.shape}")
            
            # Filter documents to include only:
            # 1. Documents found relevant during simulation (reviewed=True AND included=1)
            # 2. Documents NOT reviewed during simulation (reviewed=False) - the leftover documents
            relevant_docs = results_df[(results_df['reviewed'] == True) & (results_df['included'] == 1)]
            unreviewed_docs = results_df[results_df['reviewed'] == False]
            
            logger.info(f"Found relevant documents: {len(relevant_docs)}")
            logger.info(f"Unreviewed documents (leftover): {len(unreviewed_docs)}")
            
            # Combine the two sets
            selected_docs = pd.concat([relevant_docs, unreviewed_docs])
            logger.info(f"Total documents to export: {len(selected_docs)}")
            
            # Use the stored synergy DataFrame with OpenAlex IDs
            if not hasattr(self, '_synergy_df_with_openalex'):
                raise ValueError("Synergy dataset with OpenAlex IDs not loaded. Run simulation first.")
            
            synergy_df = self._synergy_df_with_openalex
            logger.info(f"Synergy dataset shape: {synergy_df.shape}")
            logger.info(f"Synergy dataset columns: {list(synergy_df.columns)}")
            
            # Debug: Check if openalex_id is in the synergy dataset
            logger.info(f"'openalex_id' in synergy_df.columns: {'openalex_id' in synergy_df.columns}")
            if 'openalex_id' in synergy_df.columns:
                logger.info(f"First few OpenAlex IDs from synergy_df: {synergy_df['openalex_id'].head().tolist()}")
            else:
                logger.info(f"Available columns in synergy_df: {list(synergy_df.columns)}")
            
            # Create the output dataframe in the required format
            output_data = []
            
            for idx, row in selected_docs.iterrows():
                # Get the record_id from the ASReview dataset
                asreview_record_id = row.get('record_id', idx)
                
                # Debug logging for first few records
                if len(output_data) < 5:
                    logger.info(f"Processing record {len(output_data)}: asreview_record_id={asreview_record_id}, synergy_df.shape={synergy_df.shape}")
                
                # Find matching row in synergy dataset by record_id
                base_openalex_id = ""
                doi = row.get('doi', '')
                
                # Match by record_id in synergy dataset
                if 'openalex_id' in synergy_df.columns and asreview_record_id < len(synergy_df):
                    # Use the record_id as an index into the synergy dataset
                    synergy_row = synergy_df.iloc[asreview_record_id]
                    base_openalex_id = synergy_row.get('openalex_id', '')
                    if not doi:
                        doi = synergy_row.get('doi', '')
                    if len(output_data) < 5:
                        logger.info(f"Found synergy match for record_id {asreview_record_id}: OpenAlex={base_openalex_id}")
                else:
                    if len(output_data) < 5:
                        logger.info(f"Debug info for record_id {asreview_record_id}:")
                        logger.info(f"  'openalex_id' in synergy_df.columns: {'openalex_id' in synergy_df.columns}")
                        logger.info(f"  asreview_record_id < len(synergy_df): {asreview_record_id} < {len(synergy_df)} = {asreview_record_id < len(synergy_df)}")
                    logger.warning(f"No synergy match found for record_id {asreview_record_id}")
                
                # Convert to full OpenAlex URL if available
                openalex_id = ""
                if base_openalex_id and not base_openalex_id.startswith('https://'):
                    if base_openalex_id.startswith('W'):
                        openalex_id = f"https://openalex.org/{base_openalex_id}"
                    else:
                        openalex_id = f"https://openalex.org/W{base_openalex_id}"
                else:
                    openalex_id = base_openalex_id
                
                # Get title and abstract from the row
                title = row.get('title', '')
                abstract = row.get('abstract', '')
                
                # Set label_included based on what ASReview actually found:
                # 1 if the document was reviewed and found relevant during simulation
                # 0 if the document was not reviewed (leftover after stopping rule)
                if row.get('reviewed', False) and row.get('included', 0) == 1:
                    label_included = 1  # Found relevant during simulation
                else:
                    label_included = 0  # Not reviewed (leftover documents)
                
                output_data.append({
                    'record_id': asreview_record_id,
                    'openalex_id': openalex_id,
                    'doi': doi,
                    'title': title,
                    'abstract': abstract,
                    'label_included': label_included
                })
            
            # Create DataFrame with the required format
            output_df = pd.DataFrame(output_data)
            
            # Export leftover documents
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            classifier_type = results['stats']['classifier_type']
            
            filename = f"{dataset_name}_{classifier_type}_simulation_labeled_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            output_df.to_csv(filepath, index=False)
            
            # Count the labeled documents
            found_relevant = len(output_df[output_df['label_included'] == 1])
            leftover_docs = len(output_df[output_df['label_included'] == 0])
            
            logger.info(f"Relevant + leftover documents exported to: {filepath}")
            logger.info(f"Total documents: {len(output_df)}")
            logger.info(f"Found relevant (label_included=1): {found_relevant}")
            logger.info(f"Leftover unreviewed (label_included=0): {leftover_docs}")
            logger.info(f"Format: record_id,openalex_id,doi,title,abstract,label_included")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting leftover documents: {e}")
            raise
    
    def export_found_documents(self, results: dict, dataset_name: str) -> str:
        """
        Export documents that were found relevant during simulation.
        Format: record_id,openalex_id,doi,title,abstract,label_included
        
        These documents have label_included=1 since they were found relevant before stopping rule.
        
        Args:
            results: Dictionary with simulation results
            dataset_name: Name of the dataset
            
        Returns:
            Path to the exported found documents CSV file
        """
        try:
            # Get the results from simulation
            results_df = results['results_df']
            
            # Get documents that WERE reviewed and found relevant
            found_docs = results_df[(results_df['reviewed'] == True) & (results_df['included'] == 1)].copy()
            
            # Use the stored synergy DataFrame with OpenAlex IDs
            if not hasattr(self, '_synergy_df_with_openalex'):
                raise ValueError("Synergy dataset with OpenAlex IDs not loaded. Run simulation first.")
            
            synergy_df = self._synergy_df_with_openalex
            logger.info(f"Synergy dataset shape: {synergy_df.shape}")
            logger.info(f"Synergy dataset columns: {list(synergy_df.columns)}")
            
            # Create the output dataframe in the required format
            output_data = []
            
            for idx, row in found_docs.iterrows():
                # Get the record_id from the ASReview dataset
                record_id = row.get('record_id', idx)
                
                # Find matching row in synergy dataset by record_id
                base_openalex_id = ""
                doi = row.get('doi', '')
                
                # Match by record_id in synergy dataset
                if 'openalex_id' in synergy_df.columns and record_id < len(synergy_df):
                    # Use the record_id as an index into the synergy dataset
                    synergy_row = synergy_df.iloc[record_id]
                    base_openalex_id = synergy_row.get('openalex_id', '')
                    if not doi:
                        doi = synergy_row.get('doi', '')
                    logger.debug(f"Found synergy match for record_id {record_id}: OpenAlex={base_openalex_id}")
                else:
                    logger.warning(f"No synergy match found for record_id {record_id}")
                
                # Convert to full OpenAlex URL if needed
                if base_openalex_id and not base_openalex_id.startswith('https://'):
                    if base_openalex_id.startswith('W'):
                        openalex_id = f"https://openalex.org/{base_openalex_id}"
                    else:
                        openalex_id = f"https://openalex.org/W{base_openalex_id}"
                else:
                    openalex_id = base_openalex_id
                    
                    # Get title and abstract from the row
                    title = row.get('title', '')
                    abstract = row.get('abstract', '')
                    
                    # label_included should be 1 for documents found relevant during simulation
                    label_included = 1
                    
                    output_data.append({
                        'record_id': record_id,
                        'openalex_id': openalex_id,
                        'doi': doi,
                        'title': title,
                        'abstract': abstract,
                        'label_included': label_included
                    })
            
            # Create DataFrame with the required format
            output_df = pd.DataFrame(output_data)
            
            # Export found documents
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            classifier_type = results['stats']['classifier_type']
            
            filename = f"{dataset_name}_{classifier_type}_found_docs_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            output_df.to_csv(filepath, index=False)
            
            logger.info(f"Found documents exported to: {filepath}")
            logger.info(f"Total found documents: {len(output_df)}")
            logger.info(f"All found documents have label_included=1 (found relevant during simulation)")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting found documents: {e}")
            raise
    
    def get_outlier_original_rank(self, dataset_name: str, classifier_type: str = 'nb', 
                                random_state: int = 42) -> int:
        """
        Get the rank of the known outlier in the original ASReview ranking.
        Runs a full simulation without stopping rule to get complete ranking.
        
        Args:
            dataset_name: Name of the dataset
            classifier_type: Type of classifier to use
            random_state: Random state for reproducibility
            
        Returns:
            Rank of the outlier in original ranking (1-based), or -1 if not found
        """
        # Get known outlier IDs for this dataset
        if dataset_name not in self.datasets_config:
            logger.warning(f"Dataset {dataset_name} not found in configuration")
            return -1
        
        outlier_record_ids = self.datasets_config[dataset_name].get('outlier_ids', [])
        if not outlier_record_ids:
            logger.warning(f"No known outliers defined for dataset {dataset_name}")
            return -1
        
        # For simplicity, use the first outlier ID
        target_outlier_id = outlier_record_ids[0]
        logger.info(f"Looking for outlier with record_id: {target_outlier_id}")
        
        try:
            # Load dataset
            dataset = self.load_synergy_dataset(dataset_name)
            dataset_df = dataset.get_df()
            
            # Load simulation data to map record_id to dataset indices
            simulation_data = load_simulation_data(dataset_name)
            
            # Find the dataset index for the target outlier
            outlier_rows = simulation_data[simulation_data['record_id'] == target_outlier_id]
            if outlier_rows.empty:
                logger.error(f"Outlier with record_id {target_outlier_id} not found in simulation data")
                return -1
            
            # Get the dataset index of the outlier (in the original simulation data)
            outlier_sim_index = outlier_rows.index[0]
            
            # Find the corresponding row in the ASReview dataset by matching OpenAlex ID
            target_openalex_id = simulation_data.iloc[outlier_sim_index]['openalex_id']
            logger.info(f"Target outlier OpenAlex ID: {target_openalex_id}")
            
            # Reset index to get positional indices
            dataset_df_reset = dataset_df.reset_index(drop=True)
            
            # Find the outlier in the ASReview dataset using OpenAlex ID or other identifier
            # Since ASReview dataset might not have openalex_id, we'll use record_id mapping
            # or fall back to title matching
            outlier_asreview_index = None
            
            # Try to find by record position if datasets are aligned
            if outlier_sim_index < len(dataset_df_reset):
                outlier_asreview_index = outlier_sim_index
                logger.info(f"Using position-based mapping: ASReview index {outlier_asreview_index}")
            else:
                logger.error(f"Outlier simulation index {outlier_sim_index} out of range for ASReview dataset")
                return -1
            
            logger.info(f"Running full simulation to get complete ranking...")
            
            # Set up simulation configuration WITHOUT stopping rule
            config = self.setup_simulation_config(classifier_type, random_state)
            config['stopper'] = None  # Remove stopping rule
            
            # Prepare data
            label_column = 'included'
            if label_column not in dataset_df.columns:
                logger.error(f"Label column '{label_column}' not found in dataset")
                return -1
            
            # Add initial priors
            relevant_indices = dataset_df[dataset_df[label_column] == 1].index[:2].tolist()
            irrelevant_indices = dataset_df[dataset_df[label_column] == 0].index[:3].tolist()
            initial_indices = relevant_indices + irrelevant_indices
            
            # Create ActiveLearningCycle without stopper
            cycle = asr.ActiveLearningCycle(
                querier=config['querier'],
                classifier=config['classifier'],
                feature_extractor=config['feature_extractor'],
                balancer=config['balancer']
            )
            
            # Initialize simulation without stopper - run until all documents reviewed
            simulation = asr.Simulate(
                dataset_df,
                dataset_df[label_column],
                [cycle]
            )
            
            # Label initial documents
            simulation.label(initial_indices)
            logger.info(f"Added {len(initial_indices)} initial labels")
            
            # Run simulation without stopping rule
            logger.info("Running full simulation to completion...")
            simulation.review()
            
            # Get query order
            query_order = simulation._results.query_order
            
            # Find the outlier in the query order
            if outlier_asreview_index in query_order:
                # The outlier was found during simulation
                rank = np.where(query_order == outlier_asreview_index)[0][0] + 1
                logger.info(f"Outlier found at rank: {rank}")
                return rank
            else:
                logger.error(f"Outlier not found in simulation results")
                return -1
                
        except Exception as e:
            logger.error(f"Error computing outlier original rank: {e}")
            return -1


def main():
    """Main function to run the simulation."""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="ASReview Simulation Runner for Outlier Detection Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py                           # Interactive mode
  python run_simulation.py --get-leftover-docs      # Export leftover documents (label_included=0)
  python run_simulation.py --get-found-docs         # Export found documents (label_included=1)
  python run_simulation.py --get-original-rank      # Get outlier original rank
  python run_simulation.py --dataset appenzeller --get-leftover-docs  # Non-interactive export
        """
    )
    
    parser.add_argument(
        '--get-leftover-docs', 
        action='store_true',
        help='Export documents not reviewed during simulation (label_included=0 for outlier detection)'
    )
    
    parser.add_argument(
        '--get-found-docs', 
        action='store_true',
        help='Export documents found relevant during simulation (label_included=1)'
    )
    
    parser.add_argument(
        '--get-original-rank', 
        action='store_true',
        help='Get the rank of the known outlier in the original ASReview ranking (runs full simulation without stopping rule)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name (if not provided, will prompt for selection)'
    )
    
    parser.add_argument(
        '--classifier',
        type=str,
        choices=['nb', 'svm', 'rf', 'logistic'],
        default='nb',
        help='Classifier type (default: nb)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("ASReview Simulation Runner")
    print("="*60)
    
    try:
        runner = ASReviewSimulationRunner()
        
        # Determine dataset
        if args.dataset:
            dataset_name = args.dataset
            # Validate dataset exists
            available_datasets = get_available_datasets()
            if dataset_name not in available_datasets:
                print(f"Error: Dataset '{dataset_name}' not found.")
                print(f"Available datasets: {', '.join(available_datasets)}")
                sys.exit(1)
        else:
            # Show available datasets and let user select
            runner.list_available_datasets()
            dataset_name = prompt_dataset_selection()
        
        # Set parameters
        classifier_type = args.classifier
        random_state = args.random_state
        
        # If special modes are requested, run simulation first if needed
        if args.get_leftover_docs or args.get_found_docs or args.get_original_rank:
            print(f"\nRunning simulation for analysis...")
            print(f"  Dataset: {dataset_name}")
            print(f"  Classifier: {classifier_type}")
            print(f"  Random State: {random_state}")
            
            # Run simulation (only if needed for leftover/found docs)
            if args.get_leftover_docs or args.get_found_docs:
                results = runner.run_simulation(
                    dataset_name=dataset_name,
                    classifier_type=classifier_type,
                    random_state=random_state
                )
                
                if args.get_leftover_docs:
                    print(f"\nExporting relevant + leftover documents...")
                    leftover_file = runner.export_leftover_documents(results, dataset_name)
                    print(f"Relevant + leftover documents exported to: {leftover_file}")
                
                if args.get_found_docs:
                    print(f"\nExporting found documents...")
                    found_file = runner.export_found_documents(results, dataset_name)
                    print(f"Found documents exported to: {found_file}")
            
            if args.get_original_rank:
                print(f"\nGetting original outlier rank...")
                outlier_rank = runner.get_outlier_original_rank(dataset_name, classifier_type, random_state)
                if outlier_rank != -1:
                    print(f"Outlier original rank: {outlier_rank}")
                else:
                    print("Could not determine outlier original rank")
            
            return
        
        # Regular interactive mode
        if not args.dataset:
            # Ask for classifier type interactively
            print("\nAvailable classifiers:")
            print("1. Naive Bayes (nb) - Default")
            print("2. SVM (svm)")
            print("3. Random Forest (rf)")
            print("4. Logistic Regression (logistic)")
            
            classifier_choice = input(f"\nSelect classifier (1-4, or press Enter for {classifier_type}): ").strip()
            classifier_map = {'1': 'nb', '2': 'svm', '3': 'rf', '4': 'logistic'}
            if classifier_choice in classifier_map:
                classifier_type = classifier_map[classifier_choice]
            
            # Ask for random state
            random_state_input = input(f"\nEnter random state (or press Enter for {random_state}): ").strip()
            if random_state_input:
                random_state = int(random_state_input)
        
        print(f"\nStarting simulation with:")
        print(f"  Dataset: {dataset_name}")
        print(f"  Classifier: {classifier_type}")
        print(f"  Random State: {random_state}")
        print(f"  Stopping Rule: 100 consecutive irrelevant documents")
        
        # Run simulation
        results = runner.run_simulation(
            dataset_name=dataset_name,
            classifier_type=classifier_type,
            random_state=random_state
        )
        
        # Print summary
        runner.print_simulation_summary(results)
        
        # Export results
        output_file = runner.export_results(results, dataset_name)
        
        print(f"\nSimulation completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 