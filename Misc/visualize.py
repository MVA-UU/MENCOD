#!/usr/bin/env python3
"""
ECINOD Visualization Suite - Master Thesis Quality Visualizations

This script generates comprehensive visualizations for the Extended Citation Network 
Outlier Detection (ECINOD) system, showcasing the performance of different outlier 
detection methods through professional KDE plots and comparative analyses.

Author: M.V.A. van Angeren
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils import (
    prompt_dataset_selection, load_simulation_data, 
    load_datasets_config, get_available_datasets
)
from ECINOD import CitationNetworkOutlierDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set professional plotting style with seaborn
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 300
})

class ECINODVisualizer:
    """Professional visualization suite for ECINOD outlier detection results."""
    
    def __init__(self, results: Dict, simulation_df: pd.DataFrame, 
                 dataset_name: str, outlier_info: Dict):
        """
        Initialize the visualizer with detection results.
        
        Args:
            results: Dictionary with outlier scores from ECINOD
            simulation_df: Original simulation DataFrame
            dataset_name: Name of the dataset
            outlier_info: Information about known outliers
        """
        self.results = results
        self.simulation_df = simulation_df
        self.dataset_name = dataset_name
        self.outlier_info = outlier_info
        
        # Method information for professional labeling
        self.method_info = {
            'lof_embeddings_scores': {
                'name': 'LOF on Embeddings',
                'description': 'Semantic Outlier Detection',
                'color': '#E74C3C',  # Red
                'linestyle': '-'
            },
            'lof_network_scores': {
                'name': 'LOF on Network Features', 
                'description': 'Structural Outlier Detection',
                'color': '#3498DB',  # Blue
                'linestyle': '-'
            },
            'lof_mixed_scores': {
                'name': 'LOF on Mixed Features',
                'description': 'Hybrid Outlier Detection', 
                'color': '#9B59B6',  # Purple
                'linestyle': '-'
            },
            'isolation_forest_scores': {
                'name': 'Isolation Forest',
                'description': 'Global Anomaly Detection',
                'color': '#F39C12',  # Orange
                'linestyle': '-'
            },
            'ensemble_scores': {
                'name': 'Multi-LOF Ensemble',
                'description': 'Weighted Combination',
                'color': '#27AE60',  # Green
                'linestyle': '-',
                'linewidth': 3
            }
        }
        
        # Debug: Print available result keys
        logger.info(f"Available result keys: {list(results.keys())}")
        
        # Create output directory
        self.output_dir = os.path.join(project_root, 'Misc', 'visualization_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized visualizer for {dataset_name} with {len(simulation_df)} documents")
    
    def create_all_visualizations(self) -> None:
        """Generate all visualizations sequentially."""
        logger.info("🎨 Starting comprehensive visualization generation...")
        self._create_visualizations_sequential()
        logger.info(f"✅ All visualizations saved to: {self.output_dir}")
    
    def _create_visualizations_sequential(self) -> None:
        """Generate visualizations sequentially."""
        visualization_functions = [
            ("Individual KDE Plots", self._create_individual_kde_plots),
            ("Comparative KDE Plot", self._create_comparative_kde_plot),
            ("Normalized Score Comparison", self._create_normalized_score_comparison),
            ("Method Correlation Heatmap", self._create_correlation_heatmap),
            ("Outlier Ranking Analysis", self._create_outlier_ranking_analysis),
            ("Score Distribution Analysis", self._create_score_distribution_analysis),
            ("Ensemble Weights Visualization", self._create_ensemble_weights_visualization)
        ]
        
        for name, func in visualization_functions:
            try:
                logger.info(f"🎨 Creating: {name}")
                func()
                logger.info(f"✅ Completed: {name}")
            except Exception as e:
                logger.error(f"❌ Failed {name}: {e}")
    
    def _create_individual_kde_plots(self) -> None:
        """Create individual KDE plots for each method with improved layout."""
        n_methods = len([k for k in self.results.keys() if k.endswith('_scores')])
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for method_key, method_info in self.method_info.items():
            if method_key in self.results and plot_idx < len(axes):
                ax = axes[plot_idx]
                scores = self.results[method_key]
                
                # Create KDE plot using seaborn
                try:
                    # Enhanced visual styling
                    alpha_fill = 0.5 if method_key == 'ensemble_scores' else 0.3
                    linewidth = 3 if method_key == 'ensemble_scores' else 2
                    
                    sns.kdeplot(data=scores, ax=ax, color=method_info['color'], 
                               fill=True, alpha=alpha_fill, linewidth=linewidth)
                    
                    # Highlight outlier position with better visibility
                    outlier_score = self._get_outlier_score(method_key)
                    if outlier_score is not None:
                        # Get KDE value at outlier position for better annotation placement
                        y_max = ax.get_ylim()[1]
                        
                        ax.axvline(x=outlier_score, color='darkred', linestyle='--', 
                                  linewidth=2.5, alpha=0.8, zorder=10)
                        ax.scatter([outlier_score], [y_max * 0.05], color='red', s=150, 
                                  marker='*', zorder=15, edgecolor='black', linewidth=2)
                        
                        # Better positioned annotation
                        ax.annotate('Known\nOutlier', 
                                   xy=(outlier_score, y_max * 0.05),
                                   xytext=(15, 15), textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', 
                                           alpha=0.8, edgecolor='red'),
                                   arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                                   fontsize=9, fontweight='bold', zorder=20)
                        
                        # Add score value
                        ax.text(0.02, 0.98, f'Outlier Score: {outlier_score:.3f}', 
                               transform=ax.transAxes, fontsize=9, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                               verticalalignment='top')
                    
                    ax.set_xlabel('Outlier Score', fontweight='bold', fontsize=11)
                    ax.set_ylabel('Density', fontweight='bold', fontsize=11)
                    ax.set_title(f"{method_info['name']}\n{method_info['description']}", 
                                fontweight='bold', fontsize=12, pad=15)
                    ax.grid(True, alpha=0.3)
                    
                    # Clean up spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    logger.info(f"Successfully plotted {method_key}")
                    
                except Exception as e:
                    logger.error(f"Error plotting {method_key}: {e}")
                    ax.text(0.5, 0.5, f'Error plotting\n{method_info["name"]}', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral'))
                
                plot_idx += 1
        
        # Remove unused subplots
        for idx in range(plot_idx, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'ECINOD Individual Method Analysis - {self.dataset_name.title()} Dataset', 
                    fontsize=18, fontweight='bold', y=0.96)
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # Leave space for suptitle
        
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_individual_kde_plots.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved improved individual KDE plots to {output_path}")
    
    def _create_comparative_kde_plot(self) -> None:
        """Create a single plot comparing all methods with improved overlap visualization."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 1])
        
        plotted_methods = 0
        outlier_scores = []
        
        # Main KDE plot with filled areas and better overlap handling
        for method_key, method_info in self.method_info.items():
            if method_key in self.results:
                scores = self.results[method_key]
                logger.info(f"Plotting {method_key}: {len(scores)} scores, std={np.std(scores):.4f}")
                
                try:
                    linewidth = method_info.get('linewidth', 2)
                    if method_key == 'ensemble_scores':
                        alpha_fill = 0.4
                        alpha_line = 1.0
                        zorder = 10
                    else:
                        alpha_fill = 0.2
                        alpha_line = 0.8
                        zorder = 5
                    
                    # Plot filled KDE with better alpha for overlap visualization
                    sns.kdeplot(data=scores, 
                               ax=ax1,
                               label=f"{method_info['name']}",
                               color=method_info['color'],
                               fill=True,
                               alpha=alpha_fill,
                               linewidth=linewidth,
                               linestyle='-',
                               zorder=zorder)
                    
                    # Store outlier score for this method
                    outlier_score = self._get_outlier_score(method_key)
                    if outlier_score is not None:
                        outlier_scores.append((method_key, method_info, outlier_score))
                    
                    plotted_methods += 1
                    logger.info(f"Successfully plotted {method_key}")
                    
                except Exception as e:
                    logger.error(f"Error plotting KDE for {method_key}: {e}")
        
        if plotted_methods == 0:
            ax1.text(0.5, 0.5, 'No methods with sufficient\nscore variation for KDE', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=16, bbox=dict(boxstyle='round', facecolor='lightgray'))
        else:
            # Single outlier reference line using ensemble method (best performing)
            if outlier_scores:
                # Use ensemble score as the primary outlier reference
                ensemble_outlier = None
                for method_key, method_info, score in outlier_scores:
                    if method_key == 'ensemble_scores':
                        ensemble_outlier = score
                        break
                
                # If no ensemble, use the first available outlier score
                if ensemble_outlier is None and outlier_scores:
                    ensemble_outlier = outlier_scores[0][2]
                
                if ensemble_outlier is not None:
                    ax1.axvline(x=ensemble_outlier, color='darkred', linestyle='--', 
                               linewidth=3, alpha=0.8, zorder=15)
                    
                    # Add annotation for the outlier line
                    y_max = ax1.get_ylim()[1]
                    ax1.annotate('Known Outlier', 
                                xy=(ensemble_outlier, y_max * 0.9),
                                xytext=(10, -10), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                                fontsize=10, fontweight='bold',
                                zorder=20)
        
        ax1.set_xlabel('Outlier Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax1.set_title(f'ECINOD Methods Comparison - {self.dataset_name.title()} Dataset\n'
                     f'Density Distributions with Overlap Visualization', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Improved legend positioning
        if plotted_methods > 0:
            ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                      frameon=True, fancybox=True, shadow=True, fontsize=10)
        
        ax1.grid(True, alpha=0.3)
        
        # Bottom subplot: Outlier score comparison across methods
        if outlier_scores:
            methods = [info['name'] for _, info, _ in outlier_scores]
            scores = [score for _, _, score in outlier_scores]
            colors = [info['color'] for _, info, _ in outlier_scores]
            
            bars = ax2.bar(methods, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add score labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(scores) * 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax2.set_ylabel('Outlier Score', fontsize=10, fontweight='bold')
            ax2.set_title('Outlier Scores by Method', fontsize=11, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45, labelsize=9)
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'No outlier scores available', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax2.set_title('Outlier Score Comparison', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_comparative_kde.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved improved comparative KDE plot with {plotted_methods} methods to {output_path}")
    
    def _create_normalized_score_comparison(self) -> None:
        """Create normalized score comparison to address scaling issues."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Normalize all scores to 0-1 range for fair comparison
        normalized_data = []
        outlier_positions = {}
        
        for method_key, method_info in self.method_info.items():
            if method_key in self.results:
                scores = self.results[method_key]
                
                # Min-max normalization
                min_score, max_score = scores.min(), scores.max()
                if max_score > min_score:
                    normalized_scores = (scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores = np.zeros_like(scores)
                
                # Store for plotting
                for score in normalized_scores:
                    normalized_data.append({
                        'Method': method_info['name'],
                        'Normalized_Score': score,
                        'Color': method_info['color']
                    })
                
                # Get normalized outlier position
                outlier_score = self._get_outlier_score(method_key)
                if outlier_score is not None:
                    normalized_outlier = (outlier_score - min_score) / (max_score - min_score) if max_score > min_score else 0
                    outlier_positions[method_info['name']] = normalized_outlier
        
        if not normalized_data:
            for ax in [ax1, ax2]:
                ax.text(0.5, 0.5, 'No data available for normalization', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            plt.suptitle(f'Normalized Score Comparison - {self.dataset_name.title()} Dataset', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, f'{self.dataset_name}_normalized_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        df_normalized = pd.DataFrame(normalized_data)
        
        # Left plot: Overlaid KDE plots with normalized scores
        for method_key, method_info in self.method_info.items():
            if method_key in self.results:
                method_data = df_normalized[df_normalized['Method'] == method_info['name']]
                if not method_data.empty:
                    alpha = 0.4 if method_key == 'ensemble_scores' else 0.25
                    linewidth = 3 if method_key == 'ensemble_scores' else 2
                    
                    sns.kdeplot(data=method_data['Normalized_Score'], 
                               ax=ax1,
                               label=method_info['name'],
                               color=method_info['color'],
                               fill=True,
                               alpha=alpha,
                               linewidth=linewidth)
        
        # Add single outlier reference line (using ensemble if available)
        ensemble_outlier = outlier_positions.get('Multi-LOF Ensemble')
        if ensemble_outlier is None and outlier_positions:
            ensemble_outlier = list(outlier_positions.values())[0]
        
        if ensemble_outlier is not None:
            ax1.axvline(x=ensemble_outlier, color='darkred', linestyle='--', 
                       linewidth=3, alpha=0.9, zorder=15)
            ax1.annotate('Known Outlier\n(Normalized)', 
                        xy=(ensemble_outlier, ax1.get_ylim()[1] * 0.9),
                        xytext=(10, -10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                        fontsize=10, fontweight='bold', zorder=20)
        
        ax1.set_xlabel('Normalized Score (0-1)', fontweight='bold')
        ax1.set_ylabel('Density', fontweight='bold')
        ax1.set_title('Normalized Score Distributions\n(Fair Method Comparison)', fontweight='bold', pad=15)
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, frameon=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)
        
        # Right plot: Enhanced box plot with better styling
        sns.boxplot(data=df_normalized, x='Method', y='Normalized_Score', ax=ax2, 
                   showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white", 
                                            "markeredgecolor": "black", "markersize": 6})
        
        # Add outlier positions as red stars
        for i, method in enumerate(df_normalized['Method'].unique()):
            if method in outlier_positions:
                ax2.scatter(i, outlier_positions[method], color='red', s=150, 
                           marker='*', zorder=10, edgecolor='black', linewidth=2)
        
        ax2.set_xlabel('Method', fontweight='bold')
        ax2.set_ylabel('Normalized Score (0-1)', fontweight='bold')
        ax2.set_title('Normalized Score Distributions\n(Red stars = Known outliers, Diamonds = Means)', 
                     fontweight='bold', pad=15)
        ax2.tick_params(axis='x', rotation=45, labelsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(-0.05, 1.05)  # Slightly extended range for better visibility
        
        plt.suptitle(f'Normalized Score Analysis - {self.dataset_name.title()} Dataset', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_normalized_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved normalized score comparison to {output_path}")
    
    def _create_correlation_heatmap(self) -> None:
        """Create correlation heatmap between different methods using seaborn."""
        # Prepare data for correlation analysis
        score_data = {}
        for method_key, method_info in self.method_info.items():
            if method_key in self.results:
                score_data[method_info['name']] = self.results[method_key]
        
        logger.info(f"Score data for correlation: {list(score_data.keys())}")
        
        if len(score_data) < 2:
            logger.warning("Not enough methods for correlation analysis")
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5, f'Correlation analysis requires\nat least 2 methods\n\nFound: {len(score_data)} methods', 
                   ha='center', va='center', transform=plt.gca().transAxes,
                   fontsize=16, bbox=dict(boxstyle='round', facecolor='lightgray'))
            plt.title(f'Method Correlation Matrix - {self.dataset_name.title()} Dataset', 
                     fontsize=14, fontweight='bold')
            output_path = os.path.join(self.output_dir, f'{self.dataset_name}_correlation_heatmap.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        df_scores = pd.DataFrame(score_data)
        correlation_matrix = df_scores.corr()
        
        # Create heatmap using seaborn
        plt.figure(figsize=(10, 8))
        
        try:
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            sns.heatmap(correlation_matrix, 
                       mask=mask,
                       annot=True, 
                       cmap='RdYlBu_r', 
                       center=0,
                       square=True,
                       fmt='.3f',
                       cbar_kws={"shrink": .8})
            
            plt.title(f'Method Correlation Matrix - {self.dataset_name.title()} Dataset\n'
                     f'Pearson Correlation Between Outlier Detection Methods', 
                     fontsize=14, fontweight='bold', pad=20)
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            plt.text(0.5, 0.5, f'Error creating correlation heatmap:\n{str(e)}', 
                   ha='center', va='center', transform=plt.gca().transAxes,
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral'))
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_correlation_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved correlation heatmap to {output_path}")
    
    def _create_outlier_ranking_analysis(self) -> None:
        """Create visualization showing outlier ranking performance using seaborn."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Outlier ranks across methods
        outlier_ranks = self._calculate_outlier_ranks()
        logger.info(f"Outlier ranks: {outlier_ranks}")
        
        if not outlier_ranks:
            ax1.text(0.5, 0.5, 'No outlier ranking\ndata available', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        else:
            methods = [self.method_info[k]['name'] for k in outlier_ranks.keys()]
            ranks = list(outlier_ranks.values())
            colors = [self.method_info[k]['color'] for k in outlier_ranks.keys()]
            
            # Use seaborn barplot
            rank_df = pd.DataFrame({'Method': methods, 'Rank': ranks, 'Color': colors})
            sns.barplot(data=rank_df, y='Method', x='Rank', palette=colors, ax=ax1, alpha=0.8)
            
            # Add rank annotations
            for i, (method, rank) in enumerate(zip(methods, ranks)):
                ax1.text(rank + max(ranks) * 0.01, i, f'#{rank}', 
                        ha='left', va='center', fontweight='bold')
        
        ax1.set_xlabel('Outlier Rank', fontsize=12, fontweight='bold')
        ax1.set_title(f'Known Outlier Ranking Performance\n{self.dataset_name.title()} Dataset', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Right plot: Score percentiles
        outlier_percentiles = self._calculate_outlier_percentiles()
        logger.info(f"Outlier percentiles: {outlier_percentiles}")
        
        if not outlier_percentiles:
            ax2.text(0.5, 0.5, 'No outlier percentile\ndata available', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        else:
            methods = [self.method_info[k]['name'] for k in outlier_percentiles.keys()]
            percentiles = list(outlier_percentiles.values())
            colors = [self.method_info[k]['color'] for k in outlier_percentiles.keys()]
            
            # Use seaborn barplot
            perc_df = pd.DataFrame({'Method': methods, 'Percentile': percentiles, 'Color': colors})
            sns.barplot(data=perc_df, y='Method', x='Percentile', palette=colors, ax=ax2, alpha=0.8)
            
            # Add percentile annotations
            for i, (method, percentile) in enumerate(zip(methods, percentiles)):
                ax2.text(percentile + 1, i, f'{percentile:.1f}%', 
                        ha='left', va='center', fontweight='bold')
        
        ax2.set_xlabel('Score Percentile', fontsize=12, fontweight='bold')
        ax2.set_title(f'Known Outlier Score Percentiles\n{self.dataset_name.title()} Dataset', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, 100)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_outlier_ranking.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved outlier ranking analysis to {output_path}")
    
    def _create_score_distribution_analysis(self) -> None:
        """Create detailed score distribution analysis using seaborn."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Prepare data
        score_data = []
        for method_key, method_info in self.method_info.items():
            if method_key in self.results:
                for score in self.results[method_key]:
                    score_data.append({
                        'Method': method_info['name'],
                        'Score': score,
                        'Color': method_info['color']
                    })
        
        if not score_data:
            for ax in axes:
                ax.text(0.5, 0.5, 'No score data available', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            plt.suptitle(f'Score Distribution Analysis - {self.dataset_name.title()} Dataset', 
                        fontsize=16, fontweight='bold', y=0.95)
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, f'{self.dataset_name}_score_distributions.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        df_scores = pd.DataFrame(score_data)
        
        # 1. Enhanced box plots with better styling
        sns.boxplot(data=df_scores, x='Method', y='Score', ax=axes[0],
                   showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white", 
                                            "markeredgecolor": "red", "markersize": 5})
        axes[0].set_title('Score Distribution Box Plots\n(Red diamonds = Means)', fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45, labelsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlabel('')  # Remove redundant label
        
        # 2. Enhanced violin plots with better colors
        sns.violinplot(data=df_scores, x='Method', y='Score', ax=axes[1], inner='quart')
        axes[1].set_title('Score Distribution Violin Plots\n(Inner lines = Quartiles)', fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45, labelsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel('')  # Remove redundant label
        
        # 3. Enhanced histogram comparison
        for method_key, method_info in self.method_info.items():
            if method_key in self.results:
                scores = self.results[method_key]
                alpha = 0.7 if method_key == 'ensemble_scores' else 0.5
                axes[2].hist(scores, alpha=alpha, label=method_info['name'], 
                           color=method_info['color'], bins=25, edgecolor='black', linewidth=0.5)
        
        axes[2].set_xlabel('Outlier Score', fontweight='bold')
        axes[2].set_ylabel('Frequency', fontweight='bold')
        axes[2].set_title('Score Distribution Histograms\n(Overlapping distributions)', fontweight='bold')
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[2].grid(True, alpha=0.3)
        
        # 4. Enhanced cumulative distribution
        for method_key, method_info in self.method_info.items():
            if method_key in self.results:
                scores = self.results[method_key]
                sorted_scores = np.sort(scores)
                cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
                
                linewidth = 3 if method_key == 'ensemble_scores' else 2
                alpha = 1.0 if method_key == 'ensemble_scores' else 0.8
                
                axes[3].plot(sorted_scores, cumulative, 
                           label=method_info['name'], 
                           color=method_info['color'],
                           linewidth=linewidth,
                           alpha=alpha)
                
                # Add outlier position marker
                outlier_score = self._get_outlier_score(method_key)
                if outlier_score is not None:
                    # Find position in CDF
                    idx = np.searchsorted(sorted_scores, outlier_score)
                    if idx < len(cumulative):
                        axes[3].scatter(outlier_score, cumulative[idx], 
                                      color=method_info['color'], s=50, 
                                      marker='o', zorder=10, edgecolor='black')
        
        axes[3].set_xlabel('Outlier Score', fontweight='bold')
        axes[3].set_ylabel('Cumulative Probability', fontweight='bold')
        axes[3].set_title('Cumulative Distribution Functions\n(Circles = Known outlier positions)', fontweight='bold')
        axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(f'Score Distribution Analysis - {self.dataset_name.title()} Dataset', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_score_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved score distribution analysis to {output_path}")
    
    def _create_ensemble_weights_visualization(self) -> None:
        """Create visualization showing ensemble method weights using seaborn."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Method performance (based on outlier ranking)
        outlier_ranks = self._calculate_outlier_ranks()
        outlier_percentiles = self._calculate_outlier_percentiles()
        
        if not outlier_ranks:
            ax1.text(0.5, 0.5, 'No ranking data\navailable', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        else:
            # Convert ranks to performance scores (lower rank = higher performance)
            max_rank = max(outlier_ranks.values()) if outlier_ranks else 1
            performance_scores = {k: (max_rank - v + 1) / max_rank for k, v in outlier_ranks.items()}
            
            # Plot 1: Method performance
            methods = [self.method_info[k]['name'] for k in performance_scores.keys()]
            scores = list(performance_scores.values())
            colors = [self.method_info[k]['color'] for k in performance_scores.keys()]
            
            perf_df = pd.DataFrame({'Method': methods, 'Performance': scores, 'Color': colors})
            sns.barplot(data=perf_df, x='Method', y='Performance', palette=colors, ax=ax1, alpha=0.8)
            
            # Add value annotations
            for i, (method, score) in enumerate(zip(methods, scores)):
                ax1.text(i, score + max(scores) * 0.01, f'{score:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Performance Score', fontweight='bold')
        ax1.set_title('Method Performance\n(Based on Outlier Ranking)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Enhanced score ranges with percentile information
        method_names = []
        score_ranges = []
        outlier_positions = []
        outlier_percentiles = []
        colors_list = []
        
        for i, (method_key, method_info) in enumerate(self.method_info.items()):
            if method_key in self.results:
                scores = self.results[method_key]
                outlier_score = self._get_outlier_score(method_key)
                
                method_names.append(method_info['name'])
                score_ranges.append([scores.min(), scores.max()])
                outlier_positions.append(outlier_score if outlier_score is not None else np.nan)
                colors_list.append(method_info['color'])
                
                # Calculate percentile
                if outlier_score is not None:
                    percentile = stats.percentileofscore(scores, outlier_score)
                    outlier_percentiles.append(percentile)
                else:
                    outlier_percentiles.append(np.nan)
        
        if method_names:
            # Plot score ranges as horizontal bars with enhanced styling
            for i, (name, score_range, color) in enumerate(zip(method_names, score_ranges, colors_list)):
                # Main range bar
                ax2.barh(i, score_range[1] - score_range[0], 
                        left=score_range[0], 
                        alpha=0.4, 
                        color=color,
                        height=0.6,
                        edgecolor='black',
                        linewidth=0.5)
                
                # Add quartile markers
                scores = self.results[list(self.method_info.keys())[i]]
                q25, q50, q75 = np.percentile(scores, [25, 50, 75])
                ax2.scatter([q25, q50, q75], [i, i, i], 
                           color='black', s=20, marker='|', zorder=5)
            
            # Plot outlier positions with enhanced markers
            for i, (outlier_pos, percentile) in enumerate(zip(outlier_positions, outlier_percentiles)):
                if not np.isnan(outlier_pos):
                    ax2.scatter(outlier_pos, i, 
                              color='red', 
                              s=120, 
                              marker='*', 
                              zorder=10,
                              edgecolor='black',
                              linewidth=1)
                    
                    # Add percentile annotation
                    if not np.isnan(percentile):
                        ax2.annotate(f'{percentile:.1f}%', 
                                    xy=(outlier_pos, i),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.2', 
                                            facecolor='yellow', alpha=0.7))
            
            ax2.set_yticks(range(len(method_names)))
            ax2.set_yticklabels(method_names, fontsize=10)
        
        ax2.set_xlabel('Score Range', fontweight='bold')
        ax2.set_title('Score Ranges and Outlier Performance\n(Red stars = Known outliers with percentiles)', 
                     fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_ensemble_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ensemble analysis to {output_path}")
    
    def _get_outlier_score(self, method_key_or_info) -> Optional[float]:
        """Get the outlier score for a specific method."""
        if isinstance(method_key_or_info, dict):
            method_key = None
            for k, v in self.method_info.items():
                if v == method_key_or_info:
                    method_key = k
                    break
        else:
            method_key = method_key_or_info
            
        if method_key not in self.results:
            return None
        
        # Find outlier ID
        outlier_ids = self.outlier_info.get('outlier_ids', [])
        if not outlier_ids:
            return None
        
        outlier_id = outlier_ids[0]  # Take first outlier
        
        # Find outlier score
        openalex_ids = self.results['openalex_ids']
        scores = self.results[method_key]
        
        for i, doc_id in enumerate(openalex_ids):
            # Match by record_id from simulation_df
            matching_rows = self.simulation_df[self.simulation_df['openalex_id'] == doc_id]
            if not matching_rows.empty:
                record_id = matching_rows.iloc[0]['record_id']
                if record_id == outlier_id:
                    return scores[i]
        
        return None
    
    def _calculate_outlier_ranks(self) -> Dict[str, int]:
        """Calculate ranks of known outliers for each method."""
        ranks = {}
        
        for method_key in self.method_info.keys():
            if method_key in self.results:
                scores = self.results[method_key]
                outlier_score = self._get_outlier_score(method_key)
                
                if outlier_score is not None:
                    # Rank (1 = highest score)
                    rank = np.sum(scores >= outlier_score)
                    ranks[method_key] = rank
        
        return ranks
    
    def _calculate_outlier_percentiles(self) -> Dict[str, float]:
        """Calculate percentiles of known outliers for each method."""
        percentiles = {}
        
        for method_key in self.method_info.keys():
            if method_key in self.results:
                scores = self.results[method_key]
                outlier_score = self._get_outlier_score(method_key)
                
                if outlier_score is not None:
                    percentile = stats.percentileofscore(scores, outlier_score)
                    percentiles[method_key] = percentile
        
        return percentiles


def run_ecinod_analysis(dataset_name: str) -> Tuple[Dict, pd.DataFrame, Dict]:
    """
    Run ECINOD analysis on the specified dataset.
    
    Args:
        dataset_name: Name of the dataset to analyze
        
    Returns:
        Tuple of (results, simulation_df, outlier_info)
    """
    logger.info(f"🔬 Running ECINOD analysis on {dataset_name} dataset...")
    
    # Load data
    simulation_df = load_simulation_data(dataset_name)
    datasets_config = load_datasets_config()
    outlier_info = datasets_config[dataset_name]
    
    logger.info(f"📊 Loaded {len(simulation_df)} documents for analysis")
    
    # Run ECINOD
    detector = CitationNetworkOutlierDetector(random_state=42)
    results = detector.fit_predict_outliers(simulation_df, dataset_name=dataset_name)
    
    logger.info("✅ ECINOD analysis completed successfully")
    
    return results, simulation_df, outlier_info


def main():
    """Main function for the visualization script."""
    print("=" * 70)
    print("🎨 ECINOD VISUALIZATION SUITE - MASTER THESIS QUALITY")
    print("=" * 70)
    print("Extended Citation Network Outlier Detection Visualizations")
    print("Author: M.V.A. van Angeren")
    print("=" * 70)
    
    try:
        # Dataset selection
        print("\n📁 Available datasets:")
        available_datasets = get_available_datasets()
        for i, dataset in enumerate(available_datasets, 1):
            print(f"  {i}. {dataset}")
        
        dataset_name = prompt_dataset_selection()
        
        print(f"\n🔬 Selected dataset: {dataset_name}")
        print("🚀 Starting analysis and visualization generation...")
        
        # Run analysis
        results, simulation_df, outlier_info = run_ecinod_analysis(dataset_name)
        
        # Create visualizations
        print("\n🎨 Generating professional visualizations...")
        visualizer = ECINODVisualizer(results, simulation_df, dataset_name, outlier_info)
        
        visualizer.create_all_visualizations()
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ VISUALIZATION GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📊 Dataset: {dataset_name}")
        print(f"📝 Documents analyzed: {len(simulation_df)}")
        print(f"🎯 Known outliers: {len(outlier_info.get('outlier_ids', []))}")
        print(f"🔍 Methods visualized: {len([k for k in results.keys() if k.endswith('_scores')])}")
        print(f"📁 Output directory: {visualizer.output_dir}")
        print("\n🎨 Generated visualizations:")
        print("  • Individual KDE plots for each method (improved layout)")
        print("  • Comparative KDE plot with overlap visualization")
        print("  • Normalized score comparison (addresses scaling)")
        print("  • Method correlation heatmap")  
        print("  • Outlier ranking analysis")
        print("  • Score distribution analysis")
        print("  • Ensemble weights visualization")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user.")
    except Exception as e:
        logger.error(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 