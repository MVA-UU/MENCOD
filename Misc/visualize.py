#!/usr/bin/env python3
"""
ECINOD Model Performance Visualization Suite
==========================================

Creates meaningful visualizations showing:
1. Score distributions per outlier detection method
2. Known outlier positions within these distributions  
3. Method effectiveness comparison
4. Real simulation results

Author: Marco
Purpose: Master Thesis Visualization
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add ECINOD to path
sys.path.append('ECINOD')
sys.path.append('.')
from ECINOD import CitationNetworkOutlierDetector
from utils import load_simulation_data, load_datasets_config, prompt_dataset_selection

# Set professional styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure matplotlib for high-quality output
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
})

class EcinodVisualization:
    """Professional visualization suite for ECINOD outlier detection analysis."""
    
    def __init__(self, dataset_name=None):
        """Initialize with a single dataset to analyze."""
        self.output_dir = Path('visualization_output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Load dataset configurations
        self.datasets_config = load_datasets_config()
        
        # Get dataset to analyze
        if dataset_name is None:
            self.dataset_name = prompt_dataset_selection()
        else:
            self.dataset_name = dataset_name
        
        print(f"📊 Will analyze dataset: {self.dataset_name}")
        
        # Results storage
        self.results = {}
        self.outlier_positions = {}
    
    def run_ecinod_analysis(self):
        """Run ECINOD on the selected dataset and collect results."""
        print(f"\n🔬 Running ECINOD analysis on {self.dataset_name}...")
        
        detector = CitationNetworkOutlierDetector(random_state=42)
        
        try:
            # Load simulation data
            simulation_df = load_simulation_data(self.dataset_name)
            print(f"Loaded {len(simulation_df)} documents")
            
            # Run ECINOD
            results = detector.fit_predict_outliers(simulation_df, dataset_name=self.dataset_name)
            
            # Store results
            self.results[self.dataset_name] = {
                'lof_scores': results['lof_scores'],
                'isolation_forest_scores': results['isolation_forest_scores'],
                'ensemble_scores': results['ensemble_scores'],
                'openalex_ids': results['openalex_ids'],
                'simulation_df': simulation_df
            }
            
            # Find outlier positions
            outlier_ids = self.datasets_config[self.dataset_name]['outlier_ids']
            self.outlier_positions[self.dataset_name] = self._find_outlier_positions(
                results, outlier_ids
            )
            
            print(f"✅ Completed analysis for {self.dataset_name}")
            
        except Exception as e:
            print(f"❌ Error analyzing {self.dataset_name}: {e}")
            raise
    
    def _find_outlier_positions(self, results, outlier_ids):
        """Find where known outliers rank in each method."""
        positions = {}
        
        for outlier_id in outlier_ids:
            # Find outlier in results
            outlier_idx = None
            for idx, doc_id in enumerate(results['openalex_ids']):
                if str(doc_id) == str(outlier_id):
                    outlier_idx = idx
                    break
            
            if outlier_idx is not None:
                positions[outlier_id] = {
                    'lof_score': results['lof_scores'][outlier_idx],
                    'isolation_forest_score': results['isolation_forest_scores'][outlier_idx],
                    'ensemble_score': results['ensemble_scores'][outlier_idx],
                    'lof_rank': self._get_rank(results['lof_scores'], outlier_idx),
                    'if_rank': self._get_rank(results['isolation_forest_scores'], outlier_idx),
                    'ensemble_rank': self._get_rank(results['ensemble_scores'], outlier_idx)
                }
        
        return positions
    
    def _get_rank(self, scores, outlier_idx):
        """Get rank of outlier in score array (1-based, 1 = highest score)."""
        outlier_score = scores[outlier_idx]
        return np.sum(scores > outlier_score) + 1
    
    def create_score_distributions(self):
        """Create comprehensive score distribution visualizations."""
        if not self.results:
            print("No results to visualize")
            return
        
        # Create subplots for the single dataset (1 row, 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        fig.suptitle(f'ECINOD Score Distributions - {self.dataset_name.title()} Dataset', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        methods = ['lof_scores', 'isolation_forest_scores', 'ensemble_scores']
        method_names = ['LOF (Semantic)', 'Isolation Forest (Global)', 'Ensemble (Combined)']
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        # Get the single dataset results
        dataset_name, data = next(iter(self.results.items()))
        
        for method_idx, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
            ax = axes[method_idx]
            scores = data[method]
            
            # Create histogram
            n_bins = min(50, len(scores) // 5)
            ax.hist(scores, bins=n_bins, alpha=0.7, color=color, 
                   edgecolor='black', linewidth=0.5)
            
            # Highlight outlier positions
            if dataset_name in self.outlier_positions:
                print(f"DEBUG: Highlighting outliers for {dataset_name} in {method}")
                for outlier_id, positions in self.outlier_positions[dataset_name].items():
                    score_key = method.replace('_scores', '_score')
                    outlier_score = positions[score_key]
                    print(f"DEBUG: Outlier {outlier_id} has {score_key} = {outlier_score}")
                    
                    # Add vertical line for outlier
                    ax.axvline(outlier_score, color='red', linewidth=4, 
                             linestyle='--', alpha=0.9, 
                             label=f'Outlier {outlier_id}')
                    
                    # Add rank annotation
                    rank_key = method.replace('_scores', '_rank').replace('isolation_forest', 'if')
                    rank = positions[rank_key]
                    
                    # Position annotation at top of plot
                    ax.annotate(f'Outlier {outlier_id}\nRank: {rank}', 
                              xy=(outlier_score, ax.get_ylim()[1] * 0.8),
                              xytext=(20, -20), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.5', 
                                       facecolor='yellow', alpha=0.9, edgecolor='red'),
                              fontweight='bold', fontsize=11,
                              arrowprops=dict(arrowstyle='->', color='red', lw=2))
            else:
                print(f"DEBUG: No outlier positions found for {dataset_name}")
            
            # Formatting
            ax.set_title(f'{method_name}', fontweight='bold')
            ax.set_xlabel('Outlier Score')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            if dataset_name in self.outlier_positions:
                ax.legend()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(self.output_dir / 'score_distributions.png', bbox_inches='tight', dpi=300)
        plt.show()
    
    def create_method_comparison(self):
        """Create method effectiveness comparison."""
        if not self.outlier_positions:
            print("No outlier positions to compare")
            return
        
        # Prepare comparison data
        comparison_data = []
        for dataset_name, outliers in self.outlier_positions.items():
            for outlier_id, positions in outliers.items():
                comparison_data.append({
                    'Dataset': dataset_name.title(),
                    'Outlier_ID': outlier_id,
                    'LOF_Rank': positions['lof_rank'],
                    'IsolationForest_Rank': positions['if_rank'],
                    'Ensemble_Rank': positions['ensemble_rank'],
                    'Total_Documents': len(self.results[dataset_name]['lof_scores'])
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Ranking comparison
        methods = ['LOF_Rank', 'IsolationForest_Rank', 'Ensemble_Rank']
        method_labels = ['LOF\n(Semantic)', 'Isolation Forest\n(Global)', 'Ensemble\n(Combined)']
        
        x_pos = np.arange(len(methods))
        width = 0.35
        
        for i, (_, row) in enumerate(df.iterrows()):
            label = f"{row['Dataset']} (ID: {row['Outlier_ID']})"
            ranks = [row[method] for method in methods]
            
            ax1.bar(x_pos + i*width, ranks, width, alpha=0.8, label=label)
        
        ax1.set_title('Outlier Ranking by Method\n(Lower = Better)', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Detection Method')
        ax1.set_ylabel('Rank (1 = Best)')
        ax1.set_xticks(x_pos + width/2)
        ax1.set_xticklabels(method_labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Percentile performance
        for i, (_, row) in enumerate(df.iterrows()):
            total_docs = row['Total_Documents']
            percentiles = [(total_docs - row[method] + 1) / total_docs * 100 
                          for method in methods]
            
            label = f"{row['Dataset']} (ID: {row['Outlier_ID']})"
            ax2.bar(x_pos + i*width, percentiles, width, alpha=0.8, label=label)
        
        ax2.set_title('Outlier Detection Performance\n(Higher = Better)', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Detection Method')
        ax2.set_ylabel('Percentile Score (%)')
        ax2.set_xticks(x_pos + width/2)
        ax2.set_xticklabels(method_labels)
        ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95th Percentile')
        ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90th Percentile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'method_comparison.png', bbox_inches='tight', dpi=300)
        plt.show()
    
    def create_performance_summary(self):
        """Create comprehensive performance summary."""
        if not self.outlier_positions:
            print("No data for performance summary")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ECINOD Performance Summary\nOutlier Detection Effectiveness', 
                     fontsize=18, fontweight='bold')
        
        # Prepare summary data
        summary_data = []
        for dataset_name, outliers in self.outlier_positions.items():
            total_docs = len(self.results[dataset_name]['lof_scores'])
            
            for outlier_id, positions in outliers.items():
                summary_data.append({
                    'dataset': dataset_name,
                    'outlier_id': outlier_id,
                    'total_docs': total_docs,
                    'lof_rank': positions['lof_rank'],
                    'if_rank': positions['if_rank'],
                    'ensemble_rank': positions['ensemble_rank'],
                    'lof_percentile': (total_docs - positions['lof_rank'] + 1) / total_docs * 100,
                    'if_percentile': (total_docs - positions['if_rank'] + 1) / total_docs * 100,
                    'ensemble_percentile': (total_docs - positions['ensemble_rank'] + 1) / total_docs * 100
                })
        
        df = pd.DataFrame(summary_data)
        print("DEBUG: DataFrame columns:", df.columns.tolist())  # Debug info
        
        # 1. Rank comparison heatmap
        ax1 = axes[0, 0]
        rank_data = df[['lof_rank', 'if_rank', 'ensemble_rank']].values
        dataset_labels = [f"{row['dataset']}\n(ID: {row['outlier_id']})" for _, row in df.iterrows()]
        
        im = ax1.imshow(rank_data.T, cmap='RdYlGn_r', aspect='auto')
        ax1.set_title('Outlier Ranks by Method', fontweight='bold')
        ax1.set_xticks(range(len(dataset_labels)))
        ax1.set_xticklabels(dataset_labels, rotation=45, ha='right')
        ax1.set_yticks(range(3))
        ax1.set_yticklabels(['LOF', 'Isolation Forest', 'Ensemble'])
        
        # Add rank values as text
        for i in range(len(dataset_labels)):
            for j in range(3):
                ax1.text(i, j, f'{rank_data[i,j]:.0f}', ha='center', va='center', 
                        fontweight='bold', color='white' if rank_data[i,j] > rank_data.max()/2 else 'black')
        
        # 2. Documents saved visualization
        ax2 = axes[0, 1]
        # Assuming documents would be reviewed in rank order, show documents saved
        docs_saved = df['total_docs'] - df['ensemble_rank'] + 1
        bars = ax2.bar(dataset_labels, docs_saved, color='mediumseagreen', alpha=0.8)
        ax2.set_title('Documents Saved from Manual Review\n(Using Ensemble Method)', fontweight='bold')
        ax2.set_ylabel('Documents Saved')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, docs_saved):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Percentile performance radar-like plot
        ax3 = axes[1, 0]
        methods = ['LOF', 'Isolation Forest', 'Ensemble']
        for i, (_, row) in enumerate(df.iterrows()):
            percentiles = [row['lof_percentile'], row['if_percentile'], row['ensemble_percentile']]
            label = f"{row['dataset']} (ID: {row['outlier_id']})"
            ax3.plot(methods, percentiles, 'o-', linewidth=2, markersize=8, label=label)
        
        ax3.set_title('Detection Performance by Method', fontweight='bold')
        ax3.set_ylabel('Percentile Score (%)')
        ax3.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Excellent (95%)')
        ax3.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='Very Good (90%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # 4. Overall effectiveness metrics
        ax4 = axes[1, 1]
        effectiveness_metrics = {
            'Avg LOF Percentile': df['lof_percentile'].mean(),
            'Avg IF Percentile': df['if_percentile'].mean(),
            'Avg Ensemble Percentile': df['ensemble_percentile'].mean(),
            'Best Method (Avg)': 'Ensemble' if df['ensemble_percentile'].mean() == max(df['lof_percentile'].mean(), df['if_percentile'].mean(), df['ensemble_percentile'].mean()) else 'LOF' if df['lof_percentile'].mean() > df['if_percentile'].mean() else 'Isolation Forest',
            'Total Outliers Found': len(df),
            'Excellent Performance': len(df[df['ensemble_percentile'] >= 95]),
            'Good Performance': len(df[df['ensemble_percentile'] >= 90])
        }
        
        # Create metrics table
        y_pos = np.arange(len(effectiveness_metrics))
        ax4.barh(y_pos, [1] * len(effectiveness_metrics), color='lightblue', alpha=0.3)
        
        for i, (metric, value) in enumerate(effectiveness_metrics.items()):
            ax4.text(0.02, i, metric, ha='left', va='center', fontweight='bold')
            if isinstance(value, (int, float)) and metric != 'Total Outliers Found':
                display_value = f'{value:.1f}%' if 'Percentile' in metric else str(value)
            else:
                display_value = str(value)
            ax4.text(0.98, i, display_value, ha='right', va='center', fontweight='bold')
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(-0.5, len(effectiveness_metrics) - 0.5)
        ax4.set_yticks([])
        ax4.set_xticks([])
        ax4.set_title('Overall Performance Metrics', fontweight='bold')
        
        # Remove spines
        for spine in ax4.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_summary.png', bbox_inches='tight', dpi=300)
        plt.show()
    
    def generate_all_visualizations(self):
        """Generate all meaningful visualizations."""
        print("🎨 Generating ECINOD performance visualizations...")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        
        # Run analysis
        self.run_ecinod_analysis()
        
        if not self.results:
            print("❌ No successful analysis results to visualize")
            return
        
        print("\n1. Creating score distributions...")
        self.create_score_distributions()
        
        print("2. Creating method comparison...")
        self.create_method_comparison()
        
        print("3. Creating performance summary...")
        self.create_performance_summary()
        
        print(f"\n✅ All visualizations generated successfully!")
        print(f"📊 Files saved in: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        for file in self.output_dir.glob("*.png"):
            print(f"  - {file.name}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("ECINOD VISUALIZATION SUITE")
    print("Score Distributions & Outlier Detection Performance")
    print("=" * 80)
    
    try:
        # Initialize visualization suite - prompts for dataset selection
        viz = EcinodVisualization()
        
        # Generate all visualizations
        viz.generate_all_visualizations()
        
        print("\n" + "=" * 80)
        print("VISUALIZATION COMPLETE! 🎉")
        print("Meaningful performance charts ready for thesis presentation.")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 