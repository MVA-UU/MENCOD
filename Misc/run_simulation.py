#!/usr/bin/env python3
"""
ASReview Simulation Runner

This script runs ASReview simulations using the API to test outlier detection performance.
It uses datasets from the Synergy dataset library and applies a stopping rule of 100 
consecutive irrelevant documents.
"""

import os
import sys
import logging
import argparse
import time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils import (
    load_datasets_config, 
    prompt_dataset_selection,
    get_available_datasets,
    load_simulation_data,
    load_synergy_dataset as utils_load_synergy_dataset
)

# ASReview imports
from asreview import load_dataset, Project, ActiveLearningCycle
from asreview.models.classifiers import SVM, NaiveBayes, RandomForest, Logistic
from asreview.models.queriers import Max, Uncertainty, Random, TopDown
from asreview.models.balancers import Balanced
from asreview.models.feature_extractors import Tfidf
from asreview.models.stoppers import NConsecutiveIrrelevant, IsFittable, LastRelevant
from asreview.simulation.simulate import Simulate
from asreview.datasets import DatasetManager

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
                dataset = load_dataset(temp_file_path)
                
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
    
    def setup_simulation_config(self, classifier_type: str = 'svm', random_state: int = 42):
        """
        Set up the simulation configuration with models (no main stopper here).
        
        Args:
            classifier_type: Type of classifier to use ('nb', 'svm', 'rf', 'logistic')
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with simulation configuration (without main stopper)
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
            'balancer': Balanced()
            # Note: No main stopper here - set separately in simulation methods
        }
        
        logger.info(f"Simulation configuration:")
        logger.info(f"  Classifier: {classifier_type}")
        logger.info(f"  Feature Extractor: TF-IDF")
        logger.info(f"  Querier: Max")
        logger.info(f"  Balancer: Balanced")
        
        return config
    
    def run_simulation(self, dataset_name: str, classifier_type: str = 'svm', 
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
            # Use the SAME two-cycle approach as ASReview frontend
            logger.info("Initializing ASReview simulation with frontend-compatible two-cycle approach...")
            
            # Import additional required components for frontend compatibility
            from asreview.models.queriers import TopDown
            from asreview.models.stoppers import IsFittable
            
            # Create TWO cycles exactly like the frontend
            cycles = [
                # Cycle 1: TopDown querier with IsFittable stopper (seeding phase)
                asr.ActiveLearningCycle(
                    querier=TopDown(),
                    stopper=IsFittable(),  # Stops after finding 1 relevant + 1 irrelevant
                ),
                # Cycle 2: Machine learning cycle (main active learning phase)  
                asr.ActiveLearningCycle(
                    querier=config['querier'],
                    classifier=config['classifier'],
                    feature_extractor=config['feature_extractor'],
                    balancer=config['balancer']
                )
            ]
            
            # Initialize simulation with two cycles and main stopper
            logger.info("Creating two-cycle simulation (TopDown seeding + ML active learning)")
            simulation = asr.Simulate(
                dataset_df,  # Pass the full dataset DataFrame
                labels_binary,  # Pass the binary labels
                cycles,  # Pass the TWO cycles
                stopper=NConsecutiveIrrelevant(100)  # Main stopper applies to overall simulation
            )
            
            # Frontend behavior: NO priors when none manually added
            logger.info("Using frontend-compatible approach: NO initial priors")
            logger.info("TopDown cycle will start with completely unlabeled dataset")
            # Don't label any documents initially - let TopDown find them naturally
            
            logger.info("Running two-phase simulation...")
            logger.info("  Phase 1: TopDown seeding (finds more relevant/irrelevant for ML training)")
            logger.info("  Phase 2: SVM active learning with stopping rule")
            
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
    
    def get_outlier_original_rank(self, dataset_name: str, classifier_type: str = 'svm', 
                                random_state: int = 42) -> dict:
        """
        Get the rank of the known outlier in both full dataset and leftover data.
        Runs simulations to get both rankings with comprehensive statistics.
        
        Args:
            dataset_name: Name of the dataset
            classifier_type: Type of classifier to use
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with both rankings and statistics
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
        target_outlier_record_id = outlier_record_ids[0]
        logger.info(f"Looking for outlier with record_id: {target_outlier_record_id}")
        
        try:
            # Load dataset
            dataset = self.load_synergy_dataset(dataset_name)
            dataset_df = dataset.get_df()
            
            # Prepare data - find the correct label column
            label_column = None
            for col in ['included', 'label_included', 'label', 'relevant']:
                if col in dataset_df.columns:
                    label_column = col
                    break
            
            if label_column is None:
                logger.error(f"No suitable label column found in dataset. Available columns: {list(dataset_df.columns)}")
                return {"error": "No suitable label column found"}
            
            logger.info(f"Using label column: {label_column}")
            labels_binary = dataset_df[label_column].astype(int)
            
            # Set up simulation components
            classifier_map = {
                'nb': NaiveBayes(),
                'svm': SVM(random_state=random_state),
                'rf': RandomForest(random_state=random_state),
                'logistic': Logistic(random_state=random_state)
            }
            
            if classifier_type not in classifier_map:
                logger.warning(f"Unknown classifier type '{classifier_type}', using SVM")
                classifier_type = 'svm'
            
            # Common initial priors
            relevant_indices = dataset_df[dataset_df[label_column] == 1].index[:1].tolist()
            irrelevant_indices = dataset_df[dataset_df[label_column] == 0].index[:1].tolist()
            initial_indices = relevant_indices + irrelevant_indices
            
            # ============================================================================
            # 1. RUN FULL SIMULATION (NO STOPPING RULE) FOR FULL DATASET RANKING
            # ============================================================================
            logger.info("="*60)
            logger.info("RUNNING FULL SIMULATION (NO STOPPING RULE)")
            logger.info("="*60)
            
            cycle_full = asr.ActiveLearningCycle(
                querier=Max(),
                classifier=classifier_map[classifier_type],
                feature_extractor=Tfidf(),
                balancer=Balanced()
            )
            
            simulation_full = asr.Simulate(
                dataset_df,
                labels_binary,
                cycle_full,
                stopper=-1  # Force review of ALL documents
            )
            
            simulation_full.label(initial_indices)
            logger.info(f"Added {len(initial_indices)} initial labels for full simulation")
            logger.info("Running full simulation to completion...")
            simulation_full.review()
            
            # Process full simulation results
            full_results = simulation_full._results
            if 'time' in full_results.columns:
                full_ordered = full_results.sort_values('time').reset_index(drop=True)
            else:
                full_ordered = full_results.reset_index(drop=True)
            
            # Export full simulation to CSV
            export_path_full = os.path.join(project_root, "log_full_simulation.csv")
            full_ordered.to_csv(export_path_full, index=True)
            
            # Find outlier in full simulation
            full_outlier_mask = full_ordered['record_id'] == target_outlier_record_id
            full_outlier_positions = full_ordered[full_outlier_mask].index
            
            if len(full_outlier_positions) > 0:
                full_rank = full_outlier_positions[0] + 1
                logger.info(f"Full dataset: Outlier found at rank {full_rank} out of {len(full_ordered)}")
            else:
                full_rank = -1
                logger.error(f"Outlier not found in full simulation")
            
            # ============================================================================
            # 2. RUN NORMAL SIMULATION (WITH STOPPING RULE) FOR LEFTOVER DATA RANKING
            # ============================================================================
            logger.info("="*60)
            logger.info("RUNNING NORMAL SIMULATION (WITH STOPPING RULE)")
            logger.info("="*60)
            
            cycle_normal = asr.ActiveLearningCycle(
                querier=Max(),
                classifier=classifier_map[classifier_type],
                feature_extractor=Tfidf(),
                balancer=Balanced()
            )
            
            simulation_normal = asr.Simulate(
                dataset_df,
                labels_binary,
                cycle_normal,
                stopper=NConsecutiveIrrelevant(100)  # Normal stopping rule
            )
            
            simulation_normal.label(initial_indices)
            logger.info(f"Added {len(initial_indices)} initial labels for normal simulation")
            logger.info("Running normal simulation with stopping rule...")
            simulation_normal.review()
            
            # Process normal simulation results
            normal_results = simulation_normal._results
            if 'time' in normal_results.columns:
                normal_ordered = normal_results.sort_values('time').reset_index(drop=True)
            else:
                normal_ordered = normal_results.reset_index(drop=True)
            
            # Identify reviewed and leftover documents
            reviewed_record_ids = set(normal_ordered['record_id'].tolist())
            all_record_ids = set(range(len(dataset_df)))  # Assuming record_ids are 0-based indices
            leftover_record_ids = all_record_ids - reviewed_record_ids
            
            logger.info(f"Normal simulation: Reviewed {len(reviewed_record_ids)} documents, {len(leftover_record_ids)} left over")
            
            # Export normal simulation to CSV
            export_path_normal = os.path.join(project_root, "log_normal_simulation.csv")
            normal_ordered.to_csv(export_path_normal, index=True)
            
            # Find outlier rank in leftover data
            leftover_rank = -1
            leftover_total = len(leftover_record_ids)
            
            if target_outlier_record_id in leftover_record_ids:
                # Get ranking of leftover documents from full simulation
                leftover_mask = full_ordered['record_id'].isin(leftover_record_ids)
                leftover_from_full = full_ordered[leftover_mask].reset_index(drop=True)
                
                # Find outlier position in leftover data
                leftover_outlier_mask = leftover_from_full['record_id'] == target_outlier_record_id
                leftover_outlier_positions = leftover_from_full[leftover_outlier_mask].index
                
                if len(leftover_outlier_positions) > 0:
                    leftover_rank = leftover_outlier_positions[0] + 1
                    logger.info(f"Leftover data: Outlier found at rank {leftover_rank} out of {leftover_total}")
                else:
                    logger.error(f"Outlier not found in leftover data (unexpected)")
            else:
                logger.info(f"Outlier was already reviewed in normal simulation (not in leftover data)")
                # Find where it was reviewed
                outlier_in_normal = normal_ordered[normal_ordered['record_id'] == target_outlier_record_id]
                if not outlier_in_normal.empty:
                    reviewed_rank = outlier_in_normal.index[0] + 1
                    logger.info(f"Outlier was reviewed at rank {reviewed_rank} in normal simulation")
            
            # ============================================================================
            # 3. COMPILE RESULTS AND STATISTICS
            # ============================================================================
            
            results = {
                'dataset_name': dataset_name,
                'outlier_record_id': target_outlier_record_id,
                'classifier_type': classifier_type,
                'random_state': random_state,
                
                # Full dataset statistics
                'full_dataset_total_docs': len(dataset_df),
                'full_dataset_reviewed_docs': len(full_ordered),
                'full_dataset_outlier_rank': full_rank,
                'full_dataset_outlier_found': full_rank != -1,
                
                # Normal simulation statistics
                'normal_sim_total_docs': len(dataset_df),
                'normal_sim_reviewed_docs': len(reviewed_record_ids),
                'normal_sim_leftover_docs': len(leftover_record_ids),
                'normal_sim_outlier_in_leftover': target_outlier_record_id in leftover_record_ids,
                
                # Leftover data statistics
                'leftover_total_docs': leftover_total,
                'leftover_outlier_rank': leftover_rank,
                'leftover_outlier_found': leftover_rank != -1,
                
                # Export paths
                'full_simulation_export': export_path_full,
                'normal_simulation_export': export_path_normal
            }
            
            # Print summary
            logger.info("="*60)
            logger.info("OUTLIER RANKING SUMMARY")
            logger.info("="*60)
            logger.info(f"Dataset: {dataset_name}")
            logger.info(f"Outlier Record ID: {target_outlier_record_id}")
            logger.info(f"Classifier: {classifier_type}")
            logger.info("")
            logger.info("FULL DATASET RANKING:")
            logger.info(f"  Total documents: {results['full_dataset_total_docs']:,}")
            logger.info(f"  Outlier rank: {results['full_dataset_outlier_rank']:,}" + (" (FOUND)" if results['full_dataset_outlier_found'] else " (NOT FOUND)"))
            logger.info("")
            logger.info("NORMAL SIMULATION:")
            logger.info(f"  Documents reviewed: {results['normal_sim_reviewed_docs']:,}")
            logger.info(f"  Documents left over: {results['normal_sim_leftover_docs']:,}")
            logger.info(f"  Outlier in leftover: {results['normal_sim_outlier_in_leftover']}")
            logger.info("")
            logger.info("LEFTOVER DATA RANKING:")
            logger.info(f"  Total leftover documents: {results['leftover_total_docs']:,}")
            logger.info(f"  Outlier rank in leftover: {results['leftover_outlier_rank']:,}" + (" (FOUND)" if results['leftover_outlier_found'] else " (NOT FOUND)"))
            logger.info("")
            logger.info("EXPORTED FILES:")
            logger.info(f"  Full simulation: {results['full_simulation_export']}")
            logger.info(f"  Normal simulation: {results['normal_simulation_export']}")
            logger.info("="*60)
            
            return results
                
        except Exception as e:
            logger.error(f"Error computing outlier rankings: {e}")
            return {"error": str(e)}

    def run_frontend_compatible_simulation(self, dataset_name: str, classifier_type: str = 'svm', 
                                          random_state: int = 42):
        """
        Run a simulation that EXACTLY matches ASReview frontend behavior.
        
        This replicates the frontend's run_simulation() function from _tasks.py:
        - Uses LastRelevant() stopper (finds ALL relevant documents)
        - Two-cycle approach: TopDown + IsFittable, then main ML cycle
        - No consecutive irrelevant stopping rule
        
        Args:
            dataset_name: Name of the dataset to simulate
            classifier_type: Type of classifier to use
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Starting FRONTEND-COMPATIBLE simulation for dataset: {dataset_name}")
        
        # Load dataset
        dataset = self.load_synergy_dataset(dataset_name)
        dataset_df = dataset.get_df()
        logger.info(f"Dataset contains {len(dataset_df)} documents")
        
        # Find label column
        label_column = None
        for col in ['included', 'label_included', 'label', 'relevant']:
            if col in dataset_df.columns:
                label_column = col
                break
        
        if label_column is None:
            raise ValueError("No suitable label column found in dataset")
        
        logger.info(f"Using label column: {label_column}")
        logger.info(f"Total relevant documents: {dataset_df[label_column].sum()}")
        
        # Convert to proper binary labels
        labels_binary = dataset_df[label_column].astype(int)
        
        # Set up configuration (without main stopper!)
        config = self.setup_simulation_config(classifier_type, random_state)
        
        try:
            logger.info("=== FRONTEND-COMPATIBLE SIMULATION ===")
            logger.info("Using EXACT same approach as ASReview frontend:")
            logger.info("1. TopDown seeding cycle with IsFittable stopper")
            logger.info("2. ML active learning cycle") 
            logger.info("3. LastRelevant() main stopper (finds ALL relevant documents)")
            
            # Import required components
            from asreview.models.queriers import TopDown
            from asreview.models.stoppers import IsFittable
            
            # Create EXACT frontend cycles
            cycles = [
                # Cycle 1: TopDown seeding (stops after finding 1 relevant + 1 irrelevant)
                asr.ActiveLearningCycle(
                    querier=TopDown(),
                    stopper=IsFittable(),
                ),
                # Cycle 2: Main ML cycle (no individual stopper)
                asr.ActiveLearningCycle(
                    classifier=config['classifier'],
                    querier=config['querier'],
                    balancer=config['balancer'],
                    feature_extractor=config['feature_extractor']
                )
            ]
            
            # Create simulation with NO main stopper (defaults to LastRelevant)
            logger.info("Creating simulation with LastRelevant() stopper...")
            simulation = asr.Simulate(
                dataset_df,
                labels_binary,
                cycles,  
                stopper=None,  # CRITICAL: No stopper = LastRelevant() = finds ALL relevant documents
                print_progress=True
            )
            
            # NO initial priors (like frontend default)
            logger.info("Running simulation (no initial priors, LastRelevant stopper)...")
            logger.info("Expected behavior: Find ALL relevant documents except outliers")
            
            # Run the simulation
            simulation.review()
            
            logger.info("Simulation completed successfully")
            
            # Get results
            results_df = simulation._results
            
            # Calculate statistics
            simulation_stats = {
                'dataset_name': dataset_name,
                'synergy_dataset_name': self.datasets_config[dataset_name]['synergy_dataset_name'],
                'classifier_type': classifier_type,
                'random_state': random_state,
                'total_documents': len(dataset_df),
                'total_relevant': int(labels_binary.sum()),
                'total_irrelevant': int(len(labels_binary) - labels_binary.sum()),
                'documents_reviewed': len(results_df),
                'relevant_found': int(results_df['label'].sum()),
                'stopping_rule': 'LastRelevant (finds ALL relevant documents)',
                'frontend_compatible': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Create enriched results
            enriched_results_df = self._create_enriched_results_dataframe(
                dataset_df, simulation._results, simulation_stats
            )
            
            logger.info(f"=== FRONTEND-COMPATIBLE SIMULATION RESULTS ===")
            logger.info(f"  Documents reviewed: {simulation_stats['documents_reviewed']}")
            logger.info(f"  Relevant found: {simulation_stats['relevant_found']}")
            logger.info(f"  Total relevant: {simulation_stats['total_relevant']}")
            logger.info(f"  Coverage: {simulation_stats['relevant_found'] / simulation_stats['total_relevant']:.2%}")
            logger.info(f"  Missing relevant: {simulation_stats['total_relevant'] - simulation_stats['relevant_found']}")
            
            return {
                'stats': simulation_stats,
                'results_df': enriched_results_df,
                'simulation_object': simulation
            }
            
        except Exception as e:
            logger.error(f"Frontend-compatible simulation failed: {e}")
            raise

    def run_frontend_stopping_rule_simulation(self, dataset_name: str, classifier_type: str = 'svm', 
                                              random_state: int = 42, stopping_rule: int = 100):
        """
        Run a simulation that EXACTLY matches ASReview frontend WITH stopping rule.
        
        This should find 95/96 relevant documents (stopping before the outlier) to match
        the frontend behavior with a stopping rule of 100 consecutive irrelevant documents.
        
        Args:
            dataset_name: Name of the dataset to simulate
            classifier_type: Type of classifier to use
            random_state: Random seed for reproducibility
            stopping_rule: Number of consecutive irrelevant documents before stopping
            
        Returns:
            Dictionary with simulation results including leftover documents for MENCOD
        """
        logger.info(f"Starting FRONTEND STOPPING RULE simulation for dataset: {dataset_name}")
        logger.info(f"Target: Find 95/96 relevant documents (stopping before outlier)")
        
        # Load dataset
        dataset = self.load_synergy_dataset(dataset_name)
        dataset_df = dataset.get_df()
        logger.info(f"Dataset contains {len(dataset_df)} documents")
        
        # Find label column
        label_column = None
        for col in ['included', 'label_included', 'label', 'relevant']:
            if col in dataset_df.columns:
                label_column = col
                break
        
        if label_column is None:
            raise ValueError("No suitable label column found in dataset")
        
        logger.info(f"Using label column: {label_column}")
        logger.info(f"Total relevant documents: {dataset_df[label_column].sum()}")
        
        # Convert to proper binary labels
        labels_binary = dataset_df[label_column].astype(int)
        
        # Set up configuration
        config = self.setup_simulation_config(classifier_type, random_state)
        
        try:
            logger.info("=== FRONTEND STOPPING RULE SIMULATION ===")
            logger.info("Replicating frontend with stopping rule:")
            logger.info("1. TopDown seeding cycle with IsFittable stopper")
            logger.info("2. ML active learning cycle") 
            logger.info(f"3. NConsecutiveIrrelevant({stopping_rule}) main stopper")
            
            # Import required components
            from asreview.models.queriers import TopDown
            from asreview.models.stoppers import IsFittable
            
            # Create EXACT frontend cycles
            cycles = [
                # Cycle 1: TopDown seeding (stops after finding 1 relevant + 1 irrelevant)
                asr.ActiveLearningCycle(
                    querier=TopDown(),
                    stopper=IsFittable(),
                ),
                # Cycle 2: Main ML cycle (no individual stopper)
                asr.ActiveLearningCycle(
                    classifier=config['classifier'],
                    querier=config['querier'],
                    balancer=config['balancer'],
                    feature_extractor=config['feature_extractor']
                )
            ]
            
            # Create simulation with STOPPING RULE (should find 95/96)
            logger.info(f"Creating simulation with NConsecutiveIrrelevant({stopping_rule}) stopper...")
            simulation = asr.Simulate(
                dataset_df,
                labels_binary,
                cycles,  
                stopper=NConsecutiveIrrelevant(stopping_rule),  # Use stopping rule
                print_progress=True
            )
            
            # NO initial priors (like frontend default)
            logger.info("Running simulation with stopping rule...")
            logger.info("Expected behavior: Find 95/96 relevant documents, stop before outlier")
            
            # Run the simulation
            simulation.review()
            
            logger.info("Simulation completed successfully")
            
            # Get results
            results_df = simulation._results
            
            # Calculate statistics
            simulation_stats = {
                'dataset_name': dataset_name,
                'synergy_dataset_name': self.datasets_config[dataset_name]['synergy_dataset_name'],
                'classifier_type': classifier_type,
                'random_state': random_state,
                'total_documents': len(dataset_df),
                'total_relevant': int(labels_binary.sum()),
                'total_irrelevant': int(len(labels_binary) - labels_binary.sum()),
                'documents_reviewed': len(results_df),
                'relevant_found': int(results_df['label'].sum()),
                'stopping_rule': f'NConsecutiveIrrelevant({stopping_rule})',
                'frontend_compatible': True,
                'leftover_documents': len(dataset_df) - len(results_df),
                'timestamp': datetime.now().isoformat()
            }
            
            # Create enriched results
            enriched_results_df = self._create_enriched_results_dataframe(
                dataset_df, simulation._results, simulation_stats
            )
            
            logger.info(f"=== FRONTEND STOPPING RULE SIMULATION RESULTS ===")
            logger.info(f"  Documents reviewed: {simulation_stats['documents_reviewed']}")
            logger.info(f"  Relevant found: {simulation_stats['relevant_found']}")
            logger.info(f"  Total relevant: {simulation_stats['total_relevant']}")
            logger.info(f"  Coverage: {simulation_stats['relevant_found'] / simulation_stats['total_relevant']:.2%}")
            logger.info(f"  Missing relevant: {simulation_stats['total_relevant'] - simulation_stats['relevant_found']}")
            logger.info(f"  Leftover documents: {simulation_stats['leftover_documents']}")
            
            # Check if we got the expected 95/96 relevant documents
            expected_found = 95
            actual_found = simulation_stats['relevant_found']
            
            if actual_found == expected_found:
                logger.info(f"✅ SUCCESS: Found exactly {actual_found} relevant documents (expected {expected_found})")
                logger.info("🎯 Perfect match with frontend behavior!")
            else:
                logger.warning(f"⚠️  MISMATCH: Found {actual_found} relevant documents (expected {expected_found})")
                logger.info(f"Difference: {actual_found - expected_found} documents")
            
            return {
                'stats': simulation_stats,
                'results_df': enriched_results_df,
                'simulation_object': simulation
            }
            
        except Exception as e:
            logger.error(f"Frontend stopping rule simulation failed: {e}")
            raise

    def export_leftover_documents_for_mencod(self, results: dict, dataset_name: str) -> str:
        """
        Export ALL documents with ASReview results for MENCOD analysis.
        
        This includes:
        - All 95 relevant documents that were found by ASReview
        - All 596 leftover documents (including the 1 outlier)
        - ASReview labeling information and timing
        
        Args:
            results: Dictionary with simulation results
            dataset_name: Name of the dataset
            
        Returns:
            Path to the exported complete dataset file
        """
        results_df = results['results_df']
        stats = results['stats']
        data_store = results['data_store']
        
        # Get the original dataset DataFrame with OpenAlex IDs
        if hasattr(self, '_synergy_df_with_openalex'):
            # Use the stored DataFrame with OpenAlex IDs
            original_df = self._synergy_df_with_openalex.copy()
            logger.info(f"Using stored DataFrame with OpenAlex IDs: {len(original_df)} documents")
        else:
            # Fallback to regular dataset
            original_df = data_store.get_df()
            logger.warning("No stored DataFrame with OpenAlex IDs found, using regular dataset")
        
        # Create a complete dataset with all documents
        complete_dataset = original_df.copy()
        
        # Ensure record_id and label_included are properly set
        if 'record_id' not in complete_dataset.columns:
            complete_dataset['record_id'] = range(len(complete_dataset))
        if 'label_included' not in complete_dataset.columns:
            complete_dataset['label_included'] = data_store["included"]
        
        # Initialize ASReview columns
        complete_dataset['asreview_label'] = None
        complete_dataset['asreview_time'] = None
        
        # Add ASReview results for reviewed documents
        for _, row in results_df.iterrows():
            record_id = row['record_id']
            complete_dataset.loc[record_id, 'asreview_label'] = row['label']
            complete_dataset.loc[record_id, 'asreview_time'] = row['time']
        
        # Create the specific columns you requested
        export_columns = {
            'record_id': complete_dataset['record_id'].fillna(0).astype(int),  # Fill NaN with 0, then convert to int
            'openalex_id': complete_dataset.get('openalex_id', ''),  # Use empty string if not available
            'doi': complete_dataset.get('doi', ''),
            'title': complete_dataset.get('title', ''),
            'abstract': complete_dataset.get('abstract', ''),
            'label_included': complete_dataset['label_included'].fillna(0).astype(int),  # Fill NaN with 0, then convert to int
            'asreview_label': complete_dataset['asreview_label'],
            'asreview_time': complete_dataset['asreview_time']
        }
        
        # Create final export DataFrame
        export_df = pd.DataFrame(export_columns)
        
        # Export complete dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_complete_with_asreview_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        export_df.to_csv(filepath, index=False)
        
        # Count documents
        total_documents = len(export_df)
        reviewed_documents = export_df['asreview_label'].notna().sum()
        leftover_documents = total_documents - reviewed_documents
        relevant_reviewed = export_df[export_df['asreview_label'] == 1].shape[0] if reviewed_documents > 0 else 0
        relevant_leftover = export_df[(export_df['asreview_label'].isna()) & (export_df['label_included'] == 1)].shape[0]
        
        print(f"📤 Exported COMPLETE dataset to: {filename}")
        print(f"   📊 Total documents: {total_documents}")
        print(f"   ✅ Reviewed by ASReview: {reviewed_documents} (found {relevant_reviewed} relevant)")
        print(f"   📋 Leftover documents: {leftover_documents} (contains {relevant_leftover} relevant including outlier)")
        print(f"   🎯 Ready for MENCOD analysis!")
        
        return filepath

    def run_exact_frontend_simulation(self, dataset_name: str, random_state: int = 42, stopping_rule: int = 100):
        """
        Run a simulation using EXACT ELAS u4 configuration that frontend uses by default.
        
        This replicates the exact ELAS u4 model parameters that ASReview LAB frontend
        uses by default, which should give us the exact same results as the frontend.
        
        Args:
            dataset_name: Name of the dataset to simulate
            random_state: Random seed for reproducibility
            stopping_rule: Number of consecutive irrelevant documents to stop after
            
        Returns:
            Dictionary with simulation results and statistics
        """
        print(f"🎯 Running EXACT FRONTEND (ELAS u4) SIMULATION")
        print(f"   Dataset: {dataset_name}")
        print(f"   Stopping rule: {stopping_rule} consecutive irrelevant")
        print(f"   Random state: {random_state}")
        
        # Load dataset using synergy-dataset package to get OpenAlex IDs
        data_store = self.load_synergy_dataset(dataset_name)
        
        # EXACT ELAS u4 configuration from asreview/models/models.py
        elas_u4_config = {
            'querier': Max(),
            'classifier': SVM(
                C=0.11, 
                loss="squared_hinge", 
                    random_state=random_state
            ),
            'balancer': Balanced(ratio=9.8),
            'feature_extractor': Tfidf(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                max_df=0.95
            ),
            'stopper': NConsecutiveIrrelevant(stopping_rule)
        }
        
        print(f"🔧 EXACT ELAS u4 Configuration:")
        print(f"   Classifier: SVM(C=0.11, loss='squared_hinge')")
        print(f"   Balancer: Balanced(ratio=9.8)")
        print(f"   Feature Extractor: TF-IDF(ngram_range=(1,2), sublinear_tf=True, min_df=1, max_df=0.95)")
        print(f"   Querier: Maximum")
        print(f"   Stopper: {stopping_rule} consecutive irrelevant")
        
        # Two-cycle approach as used by frontend (_tasks.py:105-115)
        cycles = [
            ActiveLearningCycle(
                querier=TopDown(),
                stopper=IsFittable()
            ),
            ActiveLearningCycle(
                querier=elas_u4_config['querier'],
                classifier=elas_u4_config['classifier'],
                balancer=elas_u4_config['balancer'],
                feature_extractor=elas_u4_config['feature_extractor'],
                stopper=elas_u4_config['stopper']
            )
        ]
        
        # Create simulation
        sim = Simulate(
            data_store.get_df(),
            data_store["included"],
            cycles,
            print_progress=True
        )
        
        # Label priors (if any) - frontend gets priors from project state
        # For simulation, we might not have specific priors
        
        # Run simulation
        print("🚀 Starting simulation...")
        start_time = time.time()
        sim.review()
        end_time = time.time()
        
        # Get results
        results_df = sim._results
        
        # Calculate statistics
        stats = self.calculate_statistics(results_df, data_store)
        stats['simulation_time'] = end_time - start_time
        stats['model_config'] = 'ELAS u4 (exact frontend)'
        
        print(f"✅ Simulation completed in {stats['simulation_time']:.2f} seconds")
        
        return {
            'results_df': results_df,
            'stats': stats,
            'data_store': data_store,
            'simulation': sim
        }

    def get_dataset(self, dataset_name: str):
        """
        Get dataset from ASReview datasets with proper Synergy loading.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            DataStore object containing the dataset with enriched metadata
        """
        # Map dataset names to actual dataset IDs for Synergy
        dataset_map = {
            'jeyaraman': 'Jeyaraman_2020',
            'hall': 'Hall_2012', 
            'appenzeller': 'Appenzeller-Herzog_2019'
        }
        
        # Try direct dataset name first (for benchmark datasets)
        try:
            dataset = load_dataset(dataset_name)
            logger.info(f"Loaded benchmark dataset: {dataset_name}")
            return dataset
        except Exception as e:
            logger.debug(f"Failed to load as benchmark: {e}")
        
        # Try loading as Synergy dataset with proper ID
        if dataset_name in dataset_map:
            try:
                synergy_id = dataset_map[dataset_name]
                # Use the synergy: prefix to load via Synergy extension
                dataset = load_dataset(f"synergy:{synergy_id}")
                logger.info(f"Loaded Synergy dataset: {synergy_id}")
                return dataset
            except Exception as e:
                logger.debug(f"Failed to load Synergy dataset: {e}")
        
        # Try loading directly with synergy prefix
        try:
            dataset = load_dataset(f"synergy:{dataset_name}")
            logger.info(f"Loaded Synergy dataset: {dataset_name}")
            return dataset
        except Exception as e:
            logger.debug(f"Failed to load with synergy prefix: {e}")
        
        # Final fallback - try file loading
        if dataset_name in dataset_map:
            try:
                filename = dataset_map[dataset_name] + '.csv'
                dataset = load_dataset(f"data/synergy_dataset/{filename}")
                logger.info(f"Loaded dataset from file: {filename}")
                return dataset
            except Exception as e:
                logger.debug(f"Failed to load from file: {e}")
        
        # Final fallback
        raise FileNotFoundError(f"Could not find dataset '{dataset_name}'. Available datasets: {list(dataset_map.keys())}")

    def calculate_statistics(self, results_df, data_store):
        """
        Calculate simulation statistics.
        
        Args:
            results_df: DataFrame with simulation results
            data_store: DataStore object containing the dataset
            
        Returns:
            Dictionary with simulation statistics
        """
        total_documents = len(data_store["included"])
        total_relevant = sum(data_store["included"])
        documents_reviewed = len(results_df)
        relevant_found = sum(results_df["label"])
        leftover_documents = total_documents - documents_reviewed
        
        stats = {
            'total_documents': total_documents,
            'total_relevant': total_relevant,
            'documents_reviewed': documents_reviewed,
            'relevant_found': relevant_found,
            'leftover_documents': leftover_documents,
            'recall': (relevant_found / total_relevant * 100) if total_relevant > 0 else 0,
            'stopping_rule': '100'
        }
        
        return stats


def main():
    runner = ASReviewSimulationRunner()
    
    print("="*70)
    print("ASReview EXACT FRONTEND SIMULATION (ELAS u4)")
    print("="*70)
    print("Choose a Synergy dataset to simulate:")
    print()
    
    # Available datasets with their Synergy names and outlier info
    datasets_config = runner.datasets_config
    available_datasets = {
        '1': ('jeyaraman', datasets_config['jeyaraman']['synergy_dataset_name'], 
              f"Known outlier: record_id {datasets_config['jeyaraman']['outlier_ids'][0]}"),
        '2': ('hall', datasets_config['hall']['synergy_dataset_name'], 
              f"Known outlier: record_id {datasets_config['hall']['outlier_ids'][0]}"),
        '3': ('appenzeller', datasets_config['appenzeller']['synergy_dataset_name'], 
              f"Known outlier: record_id {datasets_config['appenzeller']['outlier_ids'][0]}"),
    }
    
    # Display options
    for key, (name, synergy_name, outlier_info) in available_datasets.items():
        print(f"{key}. {name.title()} ({synergy_name})")
        print(f"   {outlier_info}")
        print()
    
    # Get user choice
    while True:
        try:
            choice = input("Enter your choice (1-3, or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Exiting...")
                return
                
            if choice in available_datasets:
                dataset_name, synergy_name, outlier_info = available_datasets[choice]
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 'q' to quit.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    print("="*70)
    print(f"RUNNING EXACT FRONTEND SIMULATION (ELAS u4)")
    print("="*70)
    print(f"Selected Dataset: {dataset_name.title()} ({synergy_name})")
    print(f"Goal: Find most relevant documents using ELAS u4 (exact frontend match)")
    print(f"Export: Complete dataset with ASReview labels for MENCOD analysis")
    print("="*70)
    
    try:
        # Run exact frontend simulation with ELAS u4 configuration
        results = runner.run_exact_frontend_simulation(
            dataset_name=dataset_name,
            random_state=42,
            stopping_rule=100
        )
        
        stats = results['stats']
        print(f"\n🎯 EXACT FRONTEND SIMULATION RESULTS:")
        print(f"  📊 Total documents: {stats['total_documents']}")
        print(f"  ✅ Relevant documents found: {stats['relevant_found']}/{stats['total_relevant']} ({stats['recall']:.1f}%)")
        print(f"  📋 Documents reviewed: {stats['documents_reviewed']}")
        print(f"  🛑 Stopping reason: {stats['stopping_rule']} consecutive irrelevant")
        print(f"  ⏱️  Simulation time: {stats['simulation_time']:.2f} seconds")
        print(f"  🧠 Model: {stats['model_config']}")
        
        # Export leftover documents for MENCOD
        if stats['relevant_found'] < stats['total_relevant']:
            leftover_file = runner.export_leftover_documents_for_mencod(results, dataset_name)
            print(f"\n📤 LEFTOVER DOCUMENTS EXPORTED:")
            print(f"  📁 File: {leftover_file}")
            print(f"  📊 Contains: {stats['total_relevant'] - stats['relevant_found']} relevant documents (including outlier)")
            print(f"  🎯 Ready for MENCOD reranking!")
        else:
            print(f"\n⚠️  All relevant documents found - no leftover for MENCOD")
            
        # Calculate coverage and missing documents
        actual_relevant = stats['relevant_found']
        coverage_percent = (actual_relevant / stats['total_relevant']) * 100
        missing_relevant = stats['total_relevant'] - actual_relevant
        
        print(f"\n🎯 SIMULATION RESULTS:")
        print(f"   ✅ Found: {actual_relevant}/{stats['total_relevant']} relevant documents ({coverage_percent:.1f}%)")
        if missing_relevant > 0:
            print(f"   📋 Missing: {missing_relevant} relevant documents (potential outliers)")
            print(f"   🎯 Perfect for MENCOD reranking!")
        else:
            print(f"   🏆 Found ALL relevant documents!")
        
        # Show stopping rule effectiveness
        review_efficiency = (stats['documents_reviewed'] / stats['total_documents']) * 100
        print(f"   ⚡ Efficiency: Reviewed only {review_efficiency:.1f}% of dataset")
            
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print(f"\n" + "="*70)


if __name__ == "__main__":
    main() 