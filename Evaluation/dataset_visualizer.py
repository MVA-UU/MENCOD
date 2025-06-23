"""
Dataset-specific visualization script for MENCOD outlier detection results.

This script creates professional visualizations for individual datasets including:
- KDE plots for each outlier detection score
- Feature importance plots
- Combined KDE plot showing outlier locations
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from MENCOD import CitationNetworkOutlierDetector
from utils import prompt_dataset_selection, load_simulation_data, load_datasets_config

# Set the plotting style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")

class DatasetVisualizer:
    """Professional visualizer for individual dataset analysis."""
    
    def __init__(self, dataset_name: str):
        """Initialize visualizer for a specific dataset."""
        self.dataset_name = dataset_name
        self.output_dir = os.path.join("Evaluation", f"results_{dataset_name}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data and run detection
        print(f"Loading and analyzing dataset: {dataset_name}")
        self.simulation_df = load_simulation_data(dataset_name)
        self.datasets_config = load_datasets_config()
        
        # Run outlier detection
        self.detector = CitationNetworkOutlierDetector(random_state=42)
        self.results = self.detector.fit_predict_outliers(
            self.simulation_df, dataset_name=dataset_name
        )
        
        # Get known outliers (convert from record IDs to OpenAlex IDs)
        from utils import convert_record_ids_to_openalex
        outlier_record_ids = self.datasets_config[dataset_name]['outlier_ids']
        outlier_openalex_ids = convert_record_ids_to_openalex(outlier_record_ids, self.simulation_df)
        self.known_outliers = set(str(oid) for oid in outlier_openalex_ids)
        print(f"Known outliers: {self.known_outliers}")
        
        # Get features
        self.features_df = self.detector.features_df.copy()
        
        print(f"Analysis complete. Results will be saved to: {self.output_dir}")
    
    def create_score_kde_plots(self):
        """Create KDE plots for each outlier detection score."""
        print("Creating individual KDE plots for each scoring method...")
        
        # Score methods and their display names (FIXED: LOF capitalization)
        score_methods = {
            'lof_embeddings_scores': 'LOF embeddings score',
            'lof_network_scores': 'LOF network features score',
            'lof_mixed_scores': 'LOF mixed features score',
            'isolation_forest_scores': 'Isolation forest score',
            'ensemble_scores': 'Ensemble score'
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (score_key, score_name) in enumerate(score_methods.items()):
            if score_key not in self.results:
                continue
                
            ax = axes[idx]
            scores = self.results[score_key]
            doc_ids = [str(doc_id) for doc_id in self.results['openalex_ids']]
            
            # Handle zero-scores issue for LOF embeddings (exclude rather than impute)
            if score_key == 'lof_embeddings_scores':
                zero_count = np.sum(scores == 0)
                total_count = len(scores)
                if zero_count > 0:
                    print(f"  Warning: {zero_count}/{total_count} ({zero_count/total_count*100:.1f}%) zero scores in LOF embeddings")
                    print(f"  This suggests missing embeddings - excluding zero values from visualization")
                    # Filter out zero scores and corresponding doc_ids for this method
                    non_zero_mask = scores > 0
                    scores = scores[non_zero_mask]
                    current_doc_ids = [doc_ids[i] for i in range(len(doc_ids)) if non_zero_mask[i]]
                    print(f"  Using {len(scores)} documents with valid embeddings for LOF embeddings plot")
                else:
                    current_doc_ids = doc_ids
            else:
                current_doc_ids = doc_ids
            
            # Create KDE plot with filled area
            sns.histplot(scores, kde=True, ax=ax, alpha=0.6, color='lightblue', bins=30, stat='density')
            
            # Highlight known outliers (use current_doc_ids for this method)
            outlier_scores = [scores[i] for i, doc_id in enumerate(current_doc_ids) 
                            if doc_id in self.known_outliers]
            
            if outlier_scores:
                for score in outlier_scores:
                    ax.axvline(score, color='red', linestyle='--', linewidth=2, alpha=0.8)
                    ax.text(score, ax.get_ylim()[1] * 0.9, 'Outlier', 
                           rotation=90, color='red', fontweight='bold')
            
            # FIXED: Title capitalization (sentence case) and dataset name capitalization
            ax.set_title(f'{score_name}\n({self.dataset_name.capitalize()} dataset)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Score', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(score_methods) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'kde_plots_individual.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Individual KDE plots saved")
    
    def create_feature_importance_plot(self):
        """Create feature importance visualization using Random Forest."""
        print("Creating feature importance analysis...")
        
        # Prepare feature matrix (exclude openalex_id and publication_year)
        exclude_cols = ['openalex_id']
        # FIXED: Exclude publication_year as it's always 0
        if 'publication_year' in self.features_df.columns:
            exclude_cols.append('publication_year')
            print("  Excluding publication_year (always 0)")
            
        feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        X = self.features_df[feature_cols].values
        
        # Use ensemble scores as target for importance
        y = self.results['ensemble_scores']
        
        # Handle any NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Fit Random Forest to determine feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        importance = rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Create visualization
        plt.figure(figsize=(12, max(8, len(feature_cols) * 0.4)))
        
        # Create horizontal bar plot  
        bars = plt.barh(range(len(feature_importance_df)), 
                       feature_importance_df['importance'], 
                       color=sns.color_palette("viridis", len(feature_importance_df)))
        
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
        plt.xlabel('Feature importance', fontsize=12, fontweight='bold')
        # FIXED: Title capitalization (sentence case) and dataset name capitalization
        plt.title(f'Feature importance for outlier detection\n({self.dataset_name.capitalize()} dataset)', 
                 fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature importance data
        feature_importance_df.to_csv(
            os.path.join(self.output_dir, 'feature_importance.csv'), 
            index=False
        )
        
        print(f"✓ Feature importance plot and data saved")
        return feature_importance_df
    
    def create_individual_kde_plots(self):
        """Create individual KDE plots for each scoring method."""
        print("Creating individual KDE plots for each scoring method...")
        
        # Score methods and their display names
        score_methods = {
            'lof_embeddings_scores': 'LOF embeddings',
            'lof_network_scores': 'LOF network',
            'lof_mixed_scores': 'LOF mixed',
            'isolation_forest_scores': 'Isolation forest',
            'ensemble_scores': 'Ensemble'
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        doc_ids = [str(doc_id) for doc_id in self.results['openalex_ids']]
        
        for idx, (score_key, method_name) in enumerate(score_methods.items()):
            if score_key not in self.results:
                continue
                
            ax = axes[idx]
            scores = self.results[score_key].copy()
            current_doc_ids = doc_ids.copy()
            
            # Handle zero-scores issue for LOF embeddings
            if score_key == 'lof_embeddings_scores':
                zero_count = np.sum(scores == 0)
                if zero_count > 0:
                    non_zero_mask = scores > 0
                    scores = scores[non_zero_mask]
                    current_doc_ids = [doc_ids[i] for i in range(len(doc_ids)) if non_zero_mask[i]]
            
            # Create filled KDE plot
            sns.kdeplot(scores, ax=ax, fill=True, alpha=0.6, color='lightblue', linewidth=2)
            
            # Highlight known outliers with better visibility
            outlier_scores = [scores[i] for i, doc_id in enumerate(current_doc_ids) 
                            if doc_id in self.known_outliers]
            
            if outlier_scores:
                for score in outlier_scores:
                    ax.axvline(score, color='darkred', linestyle='--', linewidth=3, alpha=0.9)
                    ax.text(score, ax.get_ylim()[1] * 0.9, '★ Outlier', 
                           rotation=90, color='darkred', fontweight='bold', fontsize=10)
            
            ax.set_title(f'{method_name}\n({self.dataset_name.capitalize()} dataset)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Score', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(score_methods) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'individual_kde_plots.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Individual KDE plots saved")

    def create_combined_kde_plot(self):
        """Create a combined KDE plot using rank-based normalization for consistent outlier positioning."""
        print("Creating combined rank-normalized KDE plot...")
        
        # Score methods and their display names
        score_methods = {
            'lof_embeddings_scores': 'LOF embeddings',
            'lof_network_scores': 'LOF network',
            'lof_mixed_scores': 'LOF mixed',
            'isolation_forest_scores': 'Isolation forest',
            'ensemble_scores': 'Ensemble'
        }
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        doc_ids = [str(doc_id) for doc_id in self.results['openalex_ids']]
        colors = sns.color_palette("Set2", len(score_methods))  # Better color palette
        
        # Store outlier percentiles for consistent positioning
        outlier_percentiles = []
        
        for idx, (score_key, method_name) in enumerate(score_methods.items()):
            if score_key not in self.results:
                continue
                
            scores = self.results[score_key].copy()
            current_doc_ids = doc_ids.copy()
            
            # Handle zero-scores issue for LOF embeddings
            if score_key == 'lof_embeddings_scores':
                zero_count = np.sum(scores == 0)
                if zero_count > 0:
                    print(f"  Excluding {zero_count} documents without embeddings from {method_name}")
                    non_zero_mask = scores > 0
                    scores = scores[non_zero_mask]
                    current_doc_ids = [doc_ids[i] for i in range(len(doc_ids)) if non_zero_mask[i]]
            
            # Convert to percentile ranks (0-100)
            from scipy.stats import rankdata
            percentile_ranks = rankdata(scores, method='average') / len(scores) * 100
            
            # Create filled KDE plot with transparency
            sns.kdeplot(percentile_ranks, ax=ax, label=method_name, 
                       color=colors[idx], linewidth=2.5, fill=True, alpha=0.3)
            
            # Find outlier percentiles for this method
            for i, doc_id in enumerate(current_doc_ids):
                if doc_id in self.known_outliers:
                    outlier_percentile = percentile_ranks[i]
                    outlier_percentiles.append((outlier_percentile, colors[idx], method_name))
        
        # Mark outliers with consistent positioning and better visibility
        for percentile, color, method in outlier_percentiles:
            ax.axvline(percentile, color=color, linestyle='--', linewidth=3, alpha=0.8)
            ax.text(percentile, ax.get_ylim()[1] * 0.95, '★', color=color, 
                   fontsize=14, ha='center', va='bottom', fontweight='bold')
        
        # Add a legend entry for outliers
        ax.axvline(-10, color='black', linestyle='--', linewidth=3, alpha=0.8, label='Known outliers')
        
        ax.set_title(f'Combined score distributions (percentile ranks)\n({self.dataset_name.capitalize()} dataset)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Percentile rank (0=lowest, 100=highest outlier score)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'combined_kde_plot.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Combined KDE plot saved")
        
        return outlier_percentiles
    
    def create_outlier_analysis_summary(self):
        """Create a summary table of outlier analysis results."""
        print("Creating outlier analysis summary...")
        
        doc_ids = [str(doc_id) for doc_id in self.results['openalex_ids']]
        
        # Find known outliers and their ranks
        summary_data = []
        
        for outlier_id in self.known_outliers:
            if outlier_id in doc_ids:
                idx = doc_ids.index(outlier_id)
                
                # Calculate ranks for each method
                for score_key in ['ensemble_scores', 'lof_embeddings_scores', 
                                'lof_network_scores', 'lof_mixed_scores', 
                                'isolation_forest_scores']:
                    if score_key in self.results:
                        scores = self.results[score_key].copy()
                        current_doc_ids = doc_ids.copy()
                        
                        # Handle zero-scores for LOF embeddings (exclude rather than impute)
                        if score_key == 'lof_embeddings_scores':
                            zero_count = np.sum(scores == 0)
                            if zero_count > 0:
                                # Filter out zero scores and corresponding doc_ids
                                non_zero_mask = scores > 0
                                scores = scores[non_zero_mask]
                                current_doc_ids = [doc_ids[i] for i in range(len(doc_ids)) if non_zero_mask[i]]
                                
                                # Check if outlier still exists after filtering
                                if outlier_id not in current_doc_ids:
                                    continue  # Skip this method for this outlier
                                
                                # Update index for filtered data
                                idx = current_doc_ids.index(outlier_id)
                        
                        rank = np.sum(scores > scores[idx]) + 1
                        percentile = (1 - rank / len(scores)) * 100
                        
                        summary_data.append({
                            'outlier_id': outlier_id,
                            'method': score_key.replace('_scores', ''),
                            'score': scores[idx],
                            'rank': rank,
                            'total_docs': len(scores),
                            'percentile': percentile
                        })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(
                os.path.join(self.output_dir, 'outlier_summary.csv'), 
                index=False
            )
            print(f"✓ Outlier analysis summary saved")
            return summary_df
        else:
            print("⚠ No known outliers found in dataset")
            return None
    
    def run_complete_analysis(self):
        """Run all visualization analyses for the dataset."""
        print(f"\n{'='*60}")
        print(f"RUNNING COMPLETE ANALYSIS FOR {self.dataset_name.upper()}")
        print(f"{'='*60}")
        
        self.create_score_kde_plots()  # Histograms with KDE overlay
        self.create_individual_kde_plots()  # Pure KDE plots for each method
        feature_importance = self.create_feature_importance_plot()
        outlier_percentiles = self.create_combined_kde_plot()  # Rank-normalized combined plot
        outlier_summary = self.create_outlier_analysis_summary()
        
        print(f"\n✅ Complete analysis finished!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Generated visualizations:")
        print(f"   • Individual histogram plots with KDE overlay (kde_plots_individual.png)")
        print(f"   • Individual pure KDE plots for each method (individual_kde_plots.png)")
        print(f"   • Feature importance analysis (feature_importance.png)")
        print(f"   • Combined rank-normalized KDE plot (combined_kde_plot.png)")
        print(f"   • Outlier analysis summary (outlier_summary.csv)")
        
        if outlier_percentiles:
            print(f"📈 Outlier performance across methods:")
            for percentile, _, method in outlier_percentiles:
                print(f"   • {method}: {percentile:.1f}th percentile")
        
        return {
            'feature_importance': feature_importance,
            'outlier_summary': outlier_summary,
            'outlier_percentiles': outlier_percentiles
        }

def main():
    """Main function to run dataset visualization."""
    print("MENCOD Dataset Visualization Tool")
    print("=" * 50)
    
    # Get dataset name from user or command line
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = prompt_dataset_selection()
    
    print(f"\nAnalyzing dataset: {dataset_name}")
    
    try:
        # Create visualizer and run analysis
        visualizer = DatasetVisualizer(dataset_name)
        results = visualizer.run_complete_analysis()
        
        print(f"\n🎉 Analysis complete for {dataset_name}!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 