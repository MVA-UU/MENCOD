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
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

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
        
        # Create output directory
        self.output_dir = os.path.join(project_root, 'Misc', 'visualization_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized visualizer for {dataset_name} with {len(simulation_df)} documents")
    
    def create_all_visualizations(self, use_multithreading: bool = True) -> None:
        """
        Generate all visualizations using multithreading for performance.
        
        Args:
            use_multithreading: Whether to use parallel processing
        """
        logger.info("🎨 Starting comprehensive visualization generation...")
        
        if use_multithreading:
            self._create_visualizations_parallel()
        else:
            self._create_visualizations_sequential()
        
        logger.info(f"✅ All visualizations saved to: {self.output_dir}")
    
    def _create_visualizations_parallel(self) -> None:
        """Generate visualizations using multithreading."""
        tasks = [
            ('individual_kde_plots', self._create_individual_kde_plots),
            ('comparative_kde_plot', self._create_comparative_kde_plot), 
            ('method_correlation_heatmap', self._create_correlation_heatmap),
            ('outlier_ranking_analysis', self._create_outlier_ranking_analysis),
            ('score_distribution_analysis', self._create_score_distribution_analysis),
            ('ensemble_weights_visualization', self._create_ensemble_weights_visualization)
        ]
        
        with ThreadPoolExecutor(max_workers=min(6, len(tasks))) as executor:
            future_to_task = {
                executor.submit(task_func): task_name 
                for task_name, task_func in tasks
            }
            
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    future.result()
                    logger.info(f"✅ Completed: {task_name}")
                except Exception as e:
                    logger.error(f"❌ Failed {task_name}: {e}")
    
    def _create_visualizations_sequential(self) -> None:
        """Generate visualizations sequentially."""
        visualization_functions = [
            ("Individual KDE Plots", self._create_individual_kde_plots),
            ("Comparative KDE Plot", self._create_comparative_kde_plot),
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
        """Create individual KDE plots for each method."""
        n_methods = len([k for k in self.results.keys() if k.endswith('_scores')])
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for method_key, method_info in self.method_info.items():
            if method_key in self.results and plot_idx < len(axes):
                ax = axes[plot_idx]
                scores = self.results[method_key]
                
                # Create KDE plot
                self._plot_single_kde(ax, scores, method_info, show_outlier=True)
                
                plot_idx += 1
        
        # Remove unused subplots
        for idx in range(plot_idx, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'ECINOD Method Comparison - {self.dataset_name.title()} Dataset', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_individual_kde_plots.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comparative_kde_plot(self) -> None:
        """Create a single plot comparing all methods."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Plot KDEs for all methods
        for method_key, method_info in self.method_info.items():
            if method_key in self.results:
                scores = self.results[method_key]
                
                # Create KDE
                if len(scores) > 1 and np.std(scores) > 0:
                    kde = stats.gaussian_kde(scores)
                    x_range = np.linspace(scores.min(), scores.max(), 1000)
                    density = kde(x_range)
                    
                    linewidth = method_info.get('linewidth', 2)
                    if method_key == 'ensemble_scores':
                        alpha = 1.0
                        zorder = 10
                    else:
                        alpha = 0.7
                        zorder = 5
                    
                    ax.plot(x_range, density, 
                           color=method_info['color'],
                           linestyle=method_info['linestyle'],
                           linewidth=linewidth,
                           alpha=alpha,
                           label=f"{method_info['name']}\n{method_info['description']}",
                           zorder=zorder)
        
        # Highlight outlier position
        self._add_outlier_indicators(ax)
        
        ax.set_xlabel('Outlier Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'ECINOD Methods Comparison - {self.dataset_name.title()} Dataset\n'
                    f'Density Distributions of Outlier Scores', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Professional legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 frameon=True, fancybox=True, shadow=True)
        
        # Grid and styling
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_comparative_kde.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_correlation_heatmap(self) -> None:
        """Create correlation heatmap between different methods."""
        # Prepare data for correlation analysis
        score_data = {}
        for method_key, method_info in self.method_info.items():
            if method_key in self.results:
                score_data[method_info['name']] = self.results[method_key]
        
        if len(score_data) < 2:
            logger.warning("Not enough methods for correlation analysis")
            return
        
        df_scores = pd.DataFrame(score_data)
        correlation_matrix = df_scores.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8},
                   ax=ax)
        
        ax.set_title(f'Method Correlation Matrix - {self.dataset_name.title()} Dataset\n'
                    f'Pearson Correlation Between Outlier Detection Methods', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_correlation_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_outlier_ranking_analysis(self) -> None:
        """Create visualization showing outlier ranking performance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Outlier ranks across methods
        outlier_ranks = self._calculate_outlier_ranks()
        
        methods = list(outlier_ranks.keys())
        ranks = list(outlier_ranks.values())
        colors = [self.method_info[k]['color'] for k in self.method_info.keys() 
                 if k in outlier_ranks]
        
        bars = ax1.barh(methods, ranks, color=colors, alpha=0.8)
        
        # Add rank annotations
        for bar, rank in zip(bars, ranks):
            width = bar.get_width()
            ax1.text(width + max(ranks) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'#{rank}', ha='left', va='center', fontweight='bold')
        
        ax1.set_xlabel('Outlier Rank', fontsize=12, fontweight='bold')
        ax1.set_title(f'Known Outlier Ranking Performance\n{self.dataset_name.title()} Dataset', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Right plot: Score percentiles
        outlier_percentiles = self._calculate_outlier_percentiles()
        
        methods = list(outlier_percentiles.keys())
        percentiles = list(outlier_percentiles.values())
        colors = [self.method_info[k]['color'] for k in self.method_info.keys() 
                 if k in outlier_percentiles]
        
        bars = ax2.barh(methods, percentiles, color=colors, alpha=0.8)
        
        # Add percentile annotations
        for bar, percentile in zip(bars, percentiles):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{percentile:.1f}%', ha='left', va='center', fontweight='bold')
        
        ax2.set_xlabel('Score Percentile', fontsize=12, fontweight='bold')
        ax2.set_title(f'Known Outlier Score Percentiles\n{self.dataset_name.title()} Dataset', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, 100)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_outlier_ranking.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_score_distribution_analysis(self) -> None:
        """Create detailed score distribution analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 1. Box plots
        ax1 = axes[0]
        score_data = []
        method_names = []
        
        for method_key, method_info in self.method_info.items():
            if method_key in self.results:
                score_data.append(self.results[method_key])
                method_names.append(method_info['name'])
        
        bp = ax1.boxplot(score_data, labels=method_names, patch_artist=True)
        
        # Color the boxes
        colors = [self.method_info[k]['color'] for k in self.method_info.keys() 
                 if k in self.results and k.endswith('_scores')]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Score Distribution Box Plots', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Violin plots
        ax2 = axes[1]
        parts = ax2.violinplot(score_data, positions=range(1, len(score_data) + 1))
        
        for i, (part, color) in enumerate(zip(parts['bodies'], colors)):
            part.set_facecolor(color)
            part.set_alpha(0.7)
        
        ax2.set_xticks(range(1, len(method_names) + 1))
        ax2.set_xticklabels(method_names, rotation=45)
        ax2.set_title('Score Distribution Violin Plots', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Histogram comparison
        ax3 = axes[2]
        for method_key, method_info in self.method_info.items():
            if method_key in self.results:
                scores = self.results[method_key]
                ax3.hist(scores, alpha=0.6, label=method_info['name'], 
                        color=method_info['color'], bins=30)
        
        ax3.set_xlabel('Outlier Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Score Distribution Histograms', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative distribution
        ax4 = axes[3]
        for method_key, method_info in self.method_info.items():
            if method_key in self.results:
                scores = self.results[method_key]
                sorted_scores = np.sort(scores)
                cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
                ax4.plot(sorted_scores, cumulative, 
                        label=method_info['name'], 
                        color=method_info['color'],
                        linewidth=2)
        
        ax4.set_xlabel('Outlier Score')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution Functions', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Score Distribution Analysis - {self.dataset_name.title()} Dataset', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_score_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_ensemble_weights_visualization(self) -> None:
        """Create visualization showing ensemble method weights."""
        # This would require access to the actual weights from the ensemble
        # For now, create a placeholder showing method importance
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Method performance (based on outlier ranking)
        outlier_ranks = self._calculate_outlier_ranks()
        outlier_percentiles = self._calculate_outlier_percentiles()
        
        # Convert ranks to performance scores (lower rank = higher performance)
        max_rank = max(outlier_ranks.values()) if outlier_ranks else 1
        performance_scores = {k: (max_rank - v + 1) / max_rank for k, v in outlier_ranks.items()}
        
        # Plot 1: Method performance
        methods = list(performance_scores.keys())
        scores = list(performance_scores.values())
        colors = [self.method_info[k]['color'] for k in methods]
        
        bars = ax1.bar(methods, scores, color=colors, alpha=0.8)
        ax1.set_ylabel('Performance Score', fontweight='bold')
        ax1.set_title('Method Performance\n(Based on Outlier Ranking)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value annotations
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Score ranges and outlier positions
        ax2_twin = ax2.twinx()
        
        for i, (method_key, method_info) in enumerate(self.method_info.items()):
            if method_key in self.results:
                scores = self.results[method_key]
                outlier_score = self._get_outlier_score(method_key)
                
                # Plot score range
                ax2.barh(i, scores.max() - scores.min(), 
                        left=scores.min(), 
                        alpha=0.3, 
                        color=method_info['color'],
                        height=0.6)
                
                # Plot outlier position
                if outlier_score is not None:
                    ax2.scatter(outlier_score, i, 
                              color='red', 
                              s=100, 
                              marker='*', 
                              zorder=10)
        
        ax2.set_yticks(range(len([k for k in self.method_info.keys() if k in self.results])))
        ax2.set_yticklabels([self.method_info[k]['name'] for k in self.method_info.keys() 
                            if k in self.results])
        ax2.set_xlabel('Score Range', fontweight='bold')
        ax2.set_title('Score Ranges and Outlier Positions\n(Red stars = Known outliers)', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_ensemble_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_single_kde(self, ax, scores: np.ndarray, method_info: Dict, 
                        show_outlier: bool = True) -> None:
        """Plot KDE for a single method with outlier highlighting."""
        if len(scores) <= 1 or np.std(scores) == 0:
            ax.text(0.5, 0.5, 'Insufficient data\nfor KDE', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, alpha=0.7)
            ax.set_title(f"{method_info['name']}\n{method_info['description']}", 
                        fontweight='bold')
            return
        
        # Create KDE
        kde = stats.gaussian_kde(scores)
        x_range = np.linspace(scores.min(), scores.max(), 1000)
        density = kde(x_range)
        
        # Plot KDE
        ax.fill_between(x_range, density, alpha=0.3, color=method_info['color'])
        ax.plot(x_range, density, color=method_info['color'], 
               linewidth=method_info.get('linewidth', 2), 
               linestyle=method_info['linestyle'])
        
        # Highlight outlier position
        if show_outlier:
            outlier_score = self._get_outlier_score(method_info)
            if outlier_score is not None:
                outlier_density = kde(outlier_score)[0]
                ax.scatter(outlier_score, outlier_density, 
                          color='red', s=100, marker='*', 
                          zorder=10, edgecolor='black', linewidth=1)
                
                # Add outlier annotation
                ax.annotate('Known\nOutlier', 
                           xy=(outlier_score, outlier_density),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Outlier Score', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title(f"{method_info['name']}\n{method_info['description']}", 
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
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
    
    def _add_outlier_indicators(self, ax) -> None:
        """Add outlier indicators to comparative plots."""
        outlier_positions = {}
        
        for method_key, method_info in self.method_info.items():
            if method_key in self.results:
                outlier_score = self._get_outlier_score(method_key)
                if outlier_score is not None:
                    outlier_positions[method_info['name']] = outlier_score
        
        if outlier_positions:
            y_max = ax.get_ylim()[1]
            for i, (method_name, score) in enumerate(outlier_positions.items()):
                ax.axvline(x=score, color='red', linestyle='--', alpha=0.7)
                ax.text(score, y_max * (0.9 - i * 0.05), 
                       f'Outlier\n({method_name})', 
                       rotation=90, va='top', ha='right',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7),
                       fontsize=8)


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
        
        # Ask about multithreading
        use_multithreading = True
        try:
            response = input("\n🔧 Use multithreading for faster generation? [Y/n]: ").strip().lower()
            if response in ['n', 'no']:
                use_multithreading = False
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
        
        visualizer.create_all_visualizations(use_multithreading=use_multithreading)
        
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
        print("  • Individual KDE plots for each method")
        print("  • Comparative KDE plot (all methods)")
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