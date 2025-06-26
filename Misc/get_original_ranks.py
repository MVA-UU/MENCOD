#!/usr/bin/env python3
"""
Get Original Rankings for MENCOD Baseline Measurement

This script gets the original ranking of outliers in both:
1. Full dataset (ASReview without stopping rule)
2. Leftover dataset (documents that would be exported after stopping rule)

This provides baseline measurements for evaluating MENCOD's ranking improvement.
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

from utils import load_datasets_config

# ASReview imports (CORRECTED TO MATCH WORKING SCRIPT)
from asreview import load_dataset, ActiveLearningCycle
from asreview.models.classifiers import SVM
from asreview.models.queriers import Max, TopDown
from asreview.models.balancers import Balanced
from asreview.models.feature_extractors import Tfidf
from asreview.models.stoppers import NConsecutiveIrrelevant, IsFittable
from asreview.simulation.simulate import Simulate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OriginalRankingAnalyzer:
    """Get original rankings for MENCOD baseline measurement."""
    
    def __init__(self):
        self.datasets_config = load_datasets_config()
        self.output_dir = os.path.join(project_root, 'ranking_analysis')
        self._ensure_output_directory()
    
    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_synergy_dataset(self, dataset_name: str):
        """Load dataset using synergy-dataset package (same as main script)."""
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        
        synergy_name = self.datasets_config[dataset_name]['synergy_dataset_name']
        logger.info(f"Loading Synergy dataset: {synergy_name}")
        
        try:
            from synergy_dataset import Dataset
            
            synergy_dataset = Dataset(synergy_name)
            df = synergy_dataset.to_frame()
            
            synergy_dict = synergy_dataset.to_dict()
            openalex_ids = list(synergy_dict.keys())
            
            df['record_id'] = range(len(df))
            df['openalex_id'] = openalex_ids
            
            # Store for later use
            self._synergy_df_with_openalex = df.copy()
            
            logger.info(f"Successfully loaded dataset: {synergy_name}")
            logger.info(f"Dataset shape: {len(df)} records")
            
            # Convert to ASReview dataset format
            import tempfile
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                    temp_file_path = tmp_file.name
                    df.to_csv(tmp_file.name, index=False)
                
                dataset = load_dataset(temp_file_path)
                
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
            
            return dataset
            
        except ImportError:
            logger.error("synergy_dataset package not found. Please install it with: pip install synergy-dataset")
            raise
        except Exception as e:
            logger.error(f"Failed to load dataset {synergy_name}: {e}")
            raise
    
    def run_full_simulation(self, dataset_name: str, random_state: int = 42):
        """
        Run EXACT ELAS u4 simulation WITHOUT stopping rule to get full ranking.
        
        Uses the exact same parameters as the main script.
        """
        print(f"🚀 Running FULL SIMULATION (no stopping rule)")
        print(f"   Dataset: {dataset_name}")
        print(f"   Random state: {random_state}")
        print(f"   Goal: Get complete ASReview ranking")
        
        # Load dataset
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
            )
        }  # No stopper in config - let cycle handle it
        
        print(f"EXACT ELAS u4 Configuration:")
        print(f"   Classifier: SVM(C=0.11, loss='squared_hinge')")
        print(f"   Balancer: Balanced(ratio=9.8)")
        print(f"   Feature Extractor: TF-IDF(ngram_range=(1,2), sublinear_tf=True, min_df=1, max_df=0.95)")
        print(f"   Querier: Maximum")
        print(f"   Stopper: -1 (process ALL documents)")
        
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
                stopper=elas_u4_config.get('stopper')  # Use .get() to handle missing stopper gracefully
            )
        ]
        
        # Create simulation
        sim = Simulate(
            data_store.get_df(),
            data_store["included"],
            cycles,
            print_progress=True,
            stopper=-1
        )
        
        # Run simulation
        print("Starting full simulation...")
        start_time = time.time()
        sim.review()
        end_time = time.time()
        
        # Get results
        results_df = sim._results
        
        print(f"Full simulation completed in {end_time - start_time:.2f} seconds")
        print(f"Documents processed: {len(results_df)}")
        print(f"Total documents: {len(data_store['included'])}")
        
        return {
            'results_df': results_df,
            'data_store': data_store,
            'simulation_time': end_time - start_time
        }
    
    def get_outlier_rankings(self, dataset_name: str, random_state: int = 42):
        """
        Get outlier rankings in both full dataset and leftover dataset.
        
        Returns:
            Dictionary with both rankings and analysis
        """
        # Get outlier information
        outlier_id = self.datasets_config[dataset_name]['outlier_ids'][0]
        print(f"\n🎯 Analyzing outlier rankings for dataset: {dataset_name}")
        print(f"🎯 Target outlier: record_id {outlier_id}")
        
        # Run full simulation
        full_results = self.run_full_simulation(dataset_name, random_state)
        results_df = full_results['results_df']
        data_store = full_results['data_store']
        
        # Sort by ASReview time to get true ranking
        print(f"\n📊 Processing simulation results...")
        results_sorted = results_df.sort_values('time').reset_index(drop=True)
        
        # Export full ranking for reference
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_ranking_file = os.path.join(self.output_dir, f"{dataset_name}_full_ranking_{timestamp}.csv")
        results_sorted.to_csv(full_ranking_file, index=False)
        print(f"📁 Full ranking exported: {full_ranking_file}")
        
        # 1. Find outlier rank in FULL dataset
        full_outlier_rank = -1
        for idx, row in results_sorted.iterrows():
            if row['record_id'] == outlier_id:
                full_outlier_rank = idx + 1  # Convert to 1-based
                break
        
        print(f"\n🌍 FULL DATASET RANKING:")
        print(f"   📊 Total documents processed: {len(results_sorted)}")
        print(f"   🎯 Outlier rank: {full_outlier_rank} (position when reviewing ALL documents)")
        
        # 2. Simulate stopping rule to find leftover documents
        print(f"\n🛑 SIMULATING STOPPING RULE (100 consecutive irrelevant):")
        
        # Apply 100 consecutive irrelevant stopping rule
        consecutive_irrelevant = 0
        stopping_point = -1
        found_first_relevant = False
        
        for idx, row in results_sorted.iterrows():
            if row['label'] == 1:  # Relevant document found
                consecutive_irrelevant = 0
                found_first_relevant = True  # Start counting consecutive irrelevant only after first relevant
            elif found_first_relevant:  # Only count consecutive irrelevant after finding first relevant
                consecutive_irrelevant += 1
                
                if consecutive_irrelevant >= 100:
                    stopping_point = idx + 1  # Stop after this document
                    break
        
        if stopping_point == -1:
            stopping_point = len(results_sorted)  # Reviewed all documents
        
        print(f"   📋 Documents that would be reviewed: {stopping_point}")
        print(f"   📄 Documents that would be leftover: {len(data_store['included']) - stopping_point}")
        
        # Get reviewed document IDs (those before stopping point)
        reviewed_record_ids = set(results_sorted.iloc[:stopping_point]['record_id'].tolist())
        
        # Check if outlier was found before stopping
        outlier_found_before_stopping = outlier_id in reviewed_record_ids
        
        # 3. Find outlier rank in LEFTOVER dataset
        leftover_outlier_rank = -1
        leftover_total_docs = 0
        
        if outlier_found_before_stopping:
            print(f"   ⚠️  Outlier WAS found before stopping rule")
            print(f"   💡 MENCOD won't help this case - outlier already discovered")
        else:
            print(f"   ✅ Outlier was NOT found before stopping (in leftover)")
            
            # Get documents in two categories:
            # 1. Found relevant during simulation (before stopping)
            relevant_found = results_sorted[
                (results_sorted.index < stopping_point) & 
                (results_sorted['label'] == 1)
            ].copy()
            
            # 2. Not reviewed (after stopping point)
            not_reviewed = results_sorted.iloc[stopping_point:].copy()
            
            # Combine them in ASReview ranking order
            leftover_documents = pd.concat([relevant_found, not_reviewed], ignore_index=True)
            leftover_total_docs = len(leftover_documents)
            
            # Find outlier position in leftover documents
            for idx, row in leftover_documents.iterrows():
                if row['record_id'] == outlier_id:
                    # Position within leftover documents (1-based)
                    leftover_outlier_rank = idx + 1  # No need to adjust for stopping_point as idx is already relative to slice
                    break
            
            print(f"   🎯 Outlier rank in leftover: {leftover_outlier_rank} out of {leftover_total_docs}")
            
            # Export leftover ranking for reference
            leftover_ranking_file = os.path.join(self.output_dir, f"{dataset_name}_leftover_ranking_{timestamp}.csv")
            leftover_documents.to_csv(leftover_ranking_file, index=False)
            print(f"   📁 Leftover ranking exported: {leftover_ranking_file}")
        
        # 4. Create comprehensive results
        results = {
            'dataset_name': dataset_name,
            'outlier_record_id': outlier_id,
            'random_state': random_state,
            
            # Full dataset results
            'full_total_documents': len(results_sorted),
            'full_outlier_rank': full_outlier_rank,
            'full_outlier_found': full_outlier_rank != -1,
            
            # Stopping rule simulation
            'stopping_point': stopping_point,
            'documents_reviewed_before_stopping': stopping_point,
            'documents_leftover_after_stopping': len(data_store['included']) - stopping_point,
            'outlier_found_before_stopping': outlier_found_before_stopping,
            
            # Leftover dataset results
            'leftover_total_documents': leftover_total_docs,
            'leftover_outlier_rank': leftover_outlier_rank,
            'leftover_outlier_found': leftover_outlier_rank != -1,
            
            # Export files
            'full_ranking_file': full_ranking_file,
            'leftover_ranking_file': leftover_ranking_file if not outlier_found_before_stopping else None,
            
            # Metadata
            'simulation_method': 'ELAS u4 (exact match with main script)',
            'stopping_rule': '100 consecutive irrelevant',
            'timestamp': timestamp
        }
        
        return results
    
    def print_results_summary(self, results: dict):
        """Print comprehensive results summary."""
        print(f"\n" + "="*70)
        print("ORIGINAL RANKING ANALYSIS - MENCOD BASELINE MEASUREMENT")
        print("="*70)
        
        print(f"📊 Dataset: {results['dataset_name']}")
        print(f"🎯 Outlier Record ID: {results['outlier_record_id']}")
        print(f"🔧 Method: {results['simulation_method']}")
        print()
        
        # Full dataset ranking
        print(f"🌍 FULL DATASET RANKING (complete ASReview run):")
        print(f"   📈 Total documents: {results['full_total_documents']:,}")
        print(f"   🎯 Outlier rank: {results['full_outlier_rank']:,}")
        if results['full_outlier_found']:
            percentile = (results['full_outlier_rank'] / results['full_total_documents']) * 100
            print(f"   📊 Outlier percentile: {percentile:.1f}%")
        print()
        
        # Stopping rule simulation
        print(f"🛑 STOPPING RULE SIMULATION:")
        print(f"   📋 Documents reviewed before stopping: {results['documents_reviewed_before_stopping']:,}")
        print(f"   📄 Documents leftover after stopping: {results['documents_leftover_after_stopping']:,}")
        print(f"   🎯 Outlier found before stopping: {'Yes' if results['outlier_found_before_stopping'] else 'No'}")
        print()
        
        # Leftover dataset ranking (key for MENCOD)
        if not results['outlier_found_before_stopping']:
            print(f"📋 LEFTOVER DATASET RANKING (MENCOD baseline measurement):")
            print(f"   📄 Total leftover documents: {results['leftover_total_documents']:,}")
            print(f"   🎯 Outlier rank in leftover: {results['leftover_outlier_rank']:,}")
            if results['leftover_outlier_found']:
                percentile = (results['leftover_outlier_rank'] / results['leftover_total_documents']) * 100
                print(f"   📊 Outlier percentile in leftover: {percentile:.1f}%")
                print()
                print(f"🚀 MENCOD PERFORMANCE MEASUREMENT:")
                print(f"   📏 Baseline rank: {results['leftover_outlier_rank']:,} out of {results['leftover_total_documents']:,}")
                print(f"   🎯 MENCOD goal: Move outlier to top ranks")
                print(f"   📈 Success metric: (Baseline rank - MENCOD rank) / Baseline rank")
                print(f"   💡 Example: If MENCOD ranks outlier at position 10:")
                improvement = ((results['leftover_outlier_rank'] - 10) / results['leftover_outlier_rank']) * 100
                print(f"       Improvement = ({results['leftover_outlier_rank']} - 10) / {results['leftover_outlier_rank']} = {improvement:.1f}%")
        else:
            print(f"⚠️  OUTLIER ALREADY FOUND:")
            print(f"   The outlier was discovered before the stopping rule")
            print(f"   MENCOD won't improve this case")
        
        print()
        print(f"📁 EXPORTED FILES:")
        print(f"   📄 Full ranking: {results['full_ranking_file']}")
        if results['leftover_ranking_file']:
            print(f"   📄 Leftover ranking: {results['leftover_ranking_file']}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Get Original Rankings for MENCOD Baseline Measurement')
    parser.add_argument('--dataset', type=str, choices=['jeyaraman', 'hall', 'appenzeller'],
                       required=True, help='Dataset to analyze')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state (must match main script)')
    args = parser.parse_args()
    
    analyzer = OriginalRankingAnalyzer()
    
    print("="*70)
    print("ORIGINAL RANKING ANALYSIS FOR MENCOD BASELINE MEASUREMENT")
    print("="*70)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🎲 Random state: {args.random_state}")
    print(f"🎯 Goal: Get baseline rankings for MENCOD improvement measurement")
    print("="*70)
    
    try:
        # Get original rankings
        results = analyzer.get_outlier_rankings(args.dataset, args.random_state)
        
        # Print comprehensive summary
        analyzer.print_results_summary(results)
        
        # Export results as JSON for later use
        import json
        results_file = os.path.join(analyzer.output_dir, f"{args.dataset}_rankings_{results['timestamp']}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results exported to: {results_file}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 