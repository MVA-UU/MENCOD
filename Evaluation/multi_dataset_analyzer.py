"""
Multi-dataset analysis script for MENCOD outlier detection results.

This script creates comprehensive comparisons across all datasets including:
- Feature importance comparison table
- Rank improvement analysis table
- Dataset characteristics comparison table
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from MENCOD import CitationNetworkOutlierDetector
from utils import load_simulation_data, load_datasets_config, get_available_datasets

# Set the plotting style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")

class MultiDatasetAnalyzer:
    """Comprehensive analyzer for multi-dataset comparisons."""
    
    def __init__(self):
        """Initialize multi-dataset analyzer."""
        self.output_dir = os.path.join("Evaluation", "multi_dataset_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.datasets_config = load_datasets_config()
        self.available_datasets = get_available_datasets()
        
        # Storage for results
        self.dataset_results = {}
        self.feature_importance_data = {}
        self.rank_improvement_data = {}
        self.dataset_statistics = {}
        
        print(f"Multi-dataset analyzer initialized")
        print(f"Available datasets: {self.available_datasets}")
        print(f"Results will be saved to: {self.output_dir}")
    
    def analyze_all_datasets(self):
        """Run analysis on all available datasets."""
        print(f"\n{'='*70}")
        print("ANALYZING ALL DATASETS")
        print(f"{'='*70}")
        
        for dataset_name in self.available_datasets:
            print(f"\nProcessing dataset: {dataset_name}")
            print("-" * 50)
            
            try:
                # Load data and run detection
                simulation_df = load_simulation_data(dataset_name)
                detector = CitationNetworkOutlierDetector(random_state=42)
                results = detector.fit_predict_outliers(simulation_df, dataset_name=dataset_name)
                
                # Store results
                self.dataset_results[dataset_name] = {
                    'simulation_df': simulation_df,
                    'results': results,
                    'features_df': detector.features_df.copy(),
                    'detector': detector
                }
                
                # Extract feature importance
                self._extract_feature_importance(dataset_name)
                
                # Calculate rank improvements
                self._calculate_rank_improvements(dataset_name)
                
                # Extract dataset statistics
                self._extract_dataset_statistics(dataset_name)
                
                print(f"✓ {dataset_name} analysis complete")
                
            except Exception as e:
                print(f"✗ Error analyzing {dataset_name}: {e}")
                continue
        
        print(f"\n{'='*70}")
        print("ALL DATASETS ANALYZED")
        print(f"{'='*70}")
    
    def _extract_feature_importance(self, dataset_name: str):
        """Extract feature importance for a dataset."""
        from sklearn.ensemble import RandomForestRegressor
        
        try:
            data = self.dataset_results[dataset_name]
            features_df = data['features_df']
            results = data['results']
            
            # Prepare feature matrix
            feature_cols = [col for col in features_df.columns if col != 'openalex_id']
            X = features_df[feature_cols].values
            y = results['ensemble_scores']
            
            # Handle NaN and infinite values
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Fit Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            # Store feature importance
            importance_dict = dict(zip(feature_cols, rf.feature_importances_))
            self.feature_importance_data[dataset_name] = importance_dict
            
        except Exception as e:
            print(f"  Warning: Could not extract feature importance for {dataset_name}: {e}")
            self.feature_importance_data[dataset_name] = {}
    
    def _calculate_rank_improvements(self, dataset_name: str):
        """Calculate rank improvements for known outliers."""
        try:
            data = self.dataset_results[dataset_name] 
            results = data['results']
            
            known_outliers = set(str(oid) for oid in self.datasets_config[dataset_name]['outlier_ids'])
            original_rank = self.datasets_config[dataset_name].get('original_rank', None)
            
            doc_ids = [str(doc_id) for doc_id in results['openalex_ids']]
            
            improvements = {}
            
            for outlier_id in known_outliers:
                if outlier_id in doc_ids:
                    idx = doc_ids.index(outlier_id)
                    
                    # Calculate new rank based on ensemble scores
                    ensemble_scores = results['ensemble_scores']
                    new_rank = np.sum(ensemble_scores > ensemble_scores[idx]) + 1
                    
                    if original_rank:
                        rank_improvement = original_rank - new_rank
                        relative_improvement = (rank_improvement / original_rank) * 100
                        
                        improvements[outlier_id] = {
                            'original_rank': original_rank,
                            'new_rank': new_rank,
                            'rank_improvement': rank_improvement,
                            'relative_improvement': relative_improvement,
                            'total_documents': len(doc_ids),
                            'new_percentile': (1 - new_rank / len(doc_ids)) * 100
                        }
            
            self.rank_improvement_data[dataset_name] = improvements
            
        except Exception as e:
            print(f"  Warning: Could not calculate rank improvements for {dataset_name}: {e}")
            self.rank_improvement_data[dataset_name] = {}
    
    def _extract_dataset_statistics(self, dataset_name: str):
        """Extract comprehensive dataset statistics."""
        try:
            data = self.dataset_results[dataset_name]
            simulation_df = data['simulation_df']
            features_df = data['features_df']
            results = data['results']
            
            # Basic statistics
            total_docs = len(simulation_df)
            relevant_docs = len(simulation_df[simulation_df['label_included'] == 1])
            irrelevant_docs = total_docs - relevant_docs
            
            # Network statistics (from features)
            feature_cols = [col for col in features_df.columns if col != 'openalex_id']
            network_features = features_df[feature_cols]
            
            # Score statistics
            score_stats = {}
            for score_type in ['ensemble_scores', 'lof_embeddings_scores', 'lof_network_scores']:
                if score_type in results:
                    scores = results[score_type]
                    score_stats[score_type] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores)
                    }
            
            self.dataset_statistics[dataset_name] = {
                'total_documents': total_docs,
                'relevant_documents': relevant_docs,
                'irrelevant_documents': irrelevant_docs,
                'relevance_ratio': relevant_docs / total_docs,
                'mean_degree': network_features.get('degree', pd.Series([0])).mean(),
                'mean_pagerank': network_features.get('pagerank', pd.Series([0])).mean(),
                'mean_clustering': network_features.get('clustering', pd.Series([0])).mean(),
                'network_density': len(features_df[features_df['degree'] > 0]) / len(features_df),
                'score_statistics': score_stats
            }
            
        except Exception as e:
            print(f"  Warning: Could not extract statistics for {dataset_name}: {e}")
            self.dataset_statistics[dataset_name] = {}
    
    def create_feature_importance_table(self):
        """Create comprehensive feature importance comparison table."""
        print("Creating feature importance comparison table...")
        
        # Collect all unique features
        all_features = set()
        for dataset_importance in self.feature_importance_data.values():
            all_features.update(dataset_importance.keys())
        
        # Create comparison matrix
        comparison_data = []
        for feature in sorted(all_features):
            row = {'feature': feature}
            
            feature_importances = []
            for dataset_name in self.available_datasets:
                importance = self.feature_importance_data.get(dataset_name, {}).get(feature, 0.0)
                row[f'{dataset_name}_importance'] = importance
                feature_importances.append(importance)
                
                # Convert to percentage
                row[f'{dataset_name}_percentage'] = importance * 100
            
            # Calculate overall statistics
            row['mean_importance'] = np.mean(feature_importances)
            row['std_importance'] = np.std(feature_importances)
            row['mean_percentage'] = np.mean(feature_importances) * 100
            
            comparison_data.append(row)
        
        # Create DataFrame and sort by mean importance
        importance_df = pd.DataFrame(comparison_data)
        importance_df = importance_df.sort_values('mean_importance', ascending=False)
        
        # Save detailed table
        importance_df.to_csv(
            os.path.join(self.output_dir, 'feature_importance_comparison.csv'),
            index=False
        )
        
        # Create visualization
        self._visualize_feature_importance_comparison(importance_df)
        
        print("✓ Feature importance comparison table created")
        return importance_df
    
    def _visualize_feature_importance_comparison(self, importance_df: pd.DataFrame):
        """Create feature importance comparison visualization."""
        
        # Select top features for visualization
        top_features = importance_df.head(15)
        
        # Prepare data for heatmap
        heatmap_data = []
        for _, row in top_features.iterrows():
            heatmap_row = []
            for dataset_name in self.available_datasets:
                heatmap_row.append(row[f'{dataset_name}_percentage'])
            heatmap_data.append(heatmap_row)
        
        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=top_features['feature'],
            columns=[name.capitalize() for name in self.available_datasets]
        )
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Feature importance (%)'})
        plt.title('Feature importance comparison across datasets', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Dataset', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_poster_feature_importance_plot(self):
        """Create a poster-friendly feature importance visualization with top features only."""
        print("Creating poster-friendly feature importance visualization...")
        
        if not self.feature_importance_data:
            print("⚠ No feature importance data available")
            return None
        
        # Get top 5 features for each dataset
        top_features_per_dataset = {}
        all_top_features = set()
        
        for dataset_name, importance_dict in self.feature_importance_data.items():
            if importance_dict:
                # Sort features by importance and get top 5
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                top_5 = sorted_features[:5]
                top_features_per_dataset[dataset_name] = dict(top_5)
                all_top_features.update([feat[0] for feat in top_5])
        
        if not all_top_features:
            print("⚠ No top features found")
            return None
        
        # Create a matrix with top features across all datasets
        feature_matrix = []
        feature_names = []
        
        for feature in all_top_features:
            row = []
            for dataset_name in self.available_datasets:
                if dataset_name in top_features_per_dataset:
                    importance = top_features_per_dataset[dataset_name].get(feature, 0)
                    # Convert to percentage for better readability
                    row.append(importance * 100)
                else:
                    row.append(0)
            
            # Only include features that are important in at least one dataset
            if max(row) > 0:
                feature_matrix.append(row)
                # Clean up feature names for better readability
                clean_name = feature.replace('_', ' ').replace('neighbor', 'neighb.').title()
                if len(clean_name) > 20:
                    clean_name = clean_name[:17] + "..."
                feature_names.append(clean_name)
        
        # Create DataFrame for heatmap
        heatmap_df = pd.DataFrame(
            feature_matrix,
            index=feature_names,
            columns=[name.capitalize() for name in self.available_datasets]
        )
        
        # Sort by maximum importance across datasets
        heatmap_df['max_importance'] = heatmap_df.max(axis=1)
        heatmap_df = heatmap_df.sort_values('max_importance', ascending=True)
        heatmap_df = heatmap_df.drop('max_importance', axis=1)
        
        # Create the poster plot
        plt.figure(figsize=(10, 8))
        
        # Use a more vibrant color scheme for poster visibility
        cmap = sns.color_palette("rocket_r", as_cmap=True)
        
        # Create heatmap with larger fonts for poster readability
        ax = sns.heatmap(heatmap_df, 
                        annot=True, 
                        fmt='.1f', 
                        cmap=cmap,
                        cbar_kws={'label': 'Feature Importance (%)', 'shrink': 0.8},
                        linewidths=0.5,
                        linecolor='white',
                        square=False,
                        annot_kws={'size': 11, 'weight': 'bold'})
        
        # Customize for poster presentation
        plt.title('Top Feature Importance Across Datasets', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Dataset', fontsize=14, fontweight='bold')
        plt.ylabel('Features', fontsize=14, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=0, fontsize=12, fontweight='bold')
        plt.yticks(rotation=0, fontsize=11, fontweight='bold')
        
        # Adjust colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=11)
        cbar.set_label('Feature Importance (%)', fontsize=12, fontweight='bold')
        
        # Make the plot more compact and poster-friendly
        plt.tight_layout()
        
        # Save with high DPI for poster quality
        plt.savefig(os.path.join(self.output_dir, 'poster_feature_importance.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Also create a summary table for reference
        summary_data = []
        for i, feature in enumerate(heatmap_df.index):
            row = {'feature': feature}
            for j, dataset in enumerate(heatmap_df.columns):
                row[f'{dataset.lower()}_importance'] = heatmap_df.iloc[i, j]
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            os.path.join(self.output_dir, 'poster_feature_importance_data.csv'),
            index=False
        )
        
        print("✓ Poster-friendly feature importance plot created")
        print(f"📊 Visualization shows top {len(feature_names)} features across {len(self.available_datasets)} datasets")
        print(f"💾 Files saved:")
        print(f"   • poster_feature_importance.png (main visualization)")
        print(f"   • poster_feature_importance_data.csv (data table)")
        
        return heatmap_df
    
    def create_rank_improvement_table(self):
        """Create rank improvement analysis table."""
        print("Creating rank improvement analysis table...")
        
        improvement_data = []
        
        for dataset_name in self.available_datasets:
            dataset_improvements = self.rank_improvement_data.get(dataset_name, {})
            
            for outlier_id, improvement_info in dataset_improvements.items():
                improvement_data.append({
                    'dataset': dataset_name,
                    'outlier_id': outlier_id,
                    'original_rank': improvement_info['original_rank'],
                    'new_rank': improvement_info['new_rank'],
                    'rank_improvement': improvement_info['rank_improvement'],
                    'relative_improvement_percent': improvement_info['relative_improvement'],
                    'total_documents': improvement_info['total_documents'],
                    'new_percentile': improvement_info['new_percentile']
                })
        
        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            improvement_df = improvement_df.sort_values('relative_improvement_percent', ascending=False)
            
            # Save table
            improvement_df.to_csv(
                os.path.join(self.output_dir, 'rank_improvement_analysis.csv'),
                index=False
            )
            
            # Create visualization
            self._visualize_rank_improvements(improvement_df)
            
            print("✓ Rank improvement analysis table created")
            return improvement_df
        else:
            print("⚠ No rank improvement data available")
            return None
    
    def _visualize_rank_improvements(self, improvement_df: pd.DataFrame):
        """Create rank improvement visualization."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Rank improvements by dataset
        sns.barplot(data=improvement_df, x='dataset', y='rank_improvement', ax=ax1, palette='viridis')
        ax1.set_title('Rank improvements by dataset', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Rank improvement', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, bar in enumerate(ax1.containers[0]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Relative improvements 
        sns.barplot(data=improvement_df, x='dataset', y='relative_improvement_percent', ax=ax2, palette='plasma')
        ax2.set_title('Relative improvements by dataset', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Relative improvement (%)', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, bar in enumerate(ax2.containers[0]):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rank_improvements_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_dataset_statistics_table(self):
        """Create comprehensive dataset characteristics table."""
        print("Creating dataset statistics comparison table...")
        
        stats_data = []
        
        for dataset_name in self.available_datasets:
            dataset_stats = self.dataset_statistics.get(dataset_name, {})
            
            if dataset_stats:
                row = {
                    'dataset': dataset_name,
                    'total_documents': dataset_stats.get('total_documents', 0),
                    'relevant_documents': dataset_stats.get('relevant_documents', 0),
                    'irrelevant_documents': dataset_stats.get('irrelevant_documents', 0),
                    'relevance_ratio': dataset_stats.get('relevance_ratio', 0.0),
                    'mean_degree': dataset_stats.get('mean_degree', 0.0),
                    'mean_pagerank': dataset_stats.get('mean_pagerank', 0.0),
                    'mean_clustering': dataset_stats.get('mean_clustering', 0.0),
                    'network_density': dataset_stats.get('network_density', 0.0),
                }
                
                # Add score statistics
                for score_type, score_stats in dataset_stats.get('score_statistics', {}).items():
                    row[f'{score_type}_mean'] = score_stats.get('mean', 0.0)
                    row[f'{score_type}_std'] = score_stats.get('std', 0.0)
                
                stats_data.append(row)
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            
            # Save table
            stats_df.to_csv(
                os.path.join(self.output_dir, 'dataset_statistics_comparison.csv'),
                index=False
            )
            
            # Create visualization
            self._visualize_dataset_statistics(stats_df)
            
            print("✓ Dataset statistics comparison table created")
            return stats_df
        else:
            print("⚠ No dataset statistics available")
            return None
    
    def _visualize_dataset_statistics(self, stats_df: pd.DataFrame):
        """Create dataset statistics visualization."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Document counts
        ax1 = axes[0, 0]
        x_pos = np.arange(len(stats_df))
        width = 0.35
        
        ax1.bar(x_pos - width/2, stats_df['relevant_documents'], width, 
               label='Relevant', color='lightgreen', alpha=0.8)
        ax1.bar(x_pos + width/2, stats_df['irrelevant_documents'], width,
               label='Irrelevant', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Dataset', fontweight='bold')
        ax1.set_ylabel('Number of documents', fontweight='bold')
        ax1.set_title('Document composition by dataset', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(stats_df['dataset'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Relevance ratio
        ax2 = axes[0, 1]
        bars = ax2.bar(stats_df['dataset'], stats_df['relevance_ratio'], 
                      color='skyblue', alpha=0.8)
        ax2.set_xlabel('Dataset', fontweight='bold')
        ax2.set_ylabel('Relevance ratio', fontweight='bold')
        ax2.set_title('Relevance ratio by dataset', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Network characteristics
        ax3 = axes[1, 0]
        ax3.bar(stats_df['dataset'], stats_df['mean_degree'], 
               color='orange', alpha=0.8)
        ax3.set_xlabel('Dataset', fontweight='bold')
        ax3.set_ylabel('Mean degree', fontweight='bold')
        ax3.set_title('Mean degree by dataset', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Network density
        ax4 = axes[1, 1]
        bars = ax4.bar(stats_df['dataset'], stats_df['network_density'], 
                      color='purple', alpha=0.8)
        ax4.set_xlabel('Dataset', fontweight='bold')
        ax4.set_ylabel('Network density', fontweight='bold')
        ax4.set_title('Network density by dataset', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dataset_statistics_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_summary_report(self):
        """Create a comprehensive summary report."""
        print("Creating comprehensive summary report...")
        
        report_lines = []
        report_lines.append("MENCOD Multi-Dataset Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Dataset overview
        report_lines.append("Dataset Overview:")
        report_lines.append("-" * 20)
        for dataset_name in self.available_datasets:
            stats = self.dataset_statistics.get(dataset_name, {})
            total_docs = stats.get('total_documents', 0)
            relevant_docs = stats.get('relevant_documents', 0)
            report_lines.append(f"• {dataset_name}: {total_docs} documents ({relevant_docs} relevant)")
        report_lines.append("")
        
        # Rank improvements summary
        report_lines.append("Rank Improvements Summary:")
        report_lines.append("-" * 30)
        for dataset_name in self.available_datasets:
            improvements = self.rank_improvement_data.get(dataset_name, {})
            if improvements:
                for outlier_id, info in improvements.items():
                    improvement = info['rank_improvement']
                    relative = info['relative_improvement']
                    report_lines.append(f"• {dataset_name} outlier {outlier_id}: {improvement} ranks ({relative:.1f}% improvement)")
        report_lines.append("")
        
        # Top features summary
        report_lines.append("Top Important Features (across all datasets):")
        report_lines.append("-" * 45)
        if self.feature_importance_data:
            # Calculate average importance across datasets
            feature_averages = defaultdict(list)
            for dataset_importance in self.feature_importance_data.values():
                for feature, importance in dataset_importance.items():
                    feature_averages[feature].append(importance)
            
            avg_importance = {feature: np.mean(importances) 
                            for feature, importances in feature_averages.items()}
            
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, avg_imp in sorted_features[:10]:
                report_lines.append(f"• {feature}: {avg_imp:.4f}")
        report_lines.append("")
        
        # Save report
        with open(os.path.join(self.output_dir, 'comprehensive_summary_report.txt'), 'w') as f:
            f.write('\n'.join(report_lines))
        
        print("✓ Comprehensive summary report created")
    
    def run_complete_analysis(self):
        """Run complete multi-dataset analysis."""
        print(f"\n{'='*80}")
        print("RUNNING COMPLETE MULTI-DATASET ANALYSIS")
        print(f"{'='*80}")
        
        # Analyze all datasets
        self.analyze_all_datasets()
        
        # Create comparison tables and visualizations
        feature_importance_df = self.create_feature_importance_table()
        poster_feature_plot = self.create_poster_feature_importance_plot()  # New poster-friendly plot
        rank_improvement_df = self.create_rank_improvement_table()
        dataset_stats_df = self.create_dataset_statistics_table()
        
        # Create comprehensive summary
        self.create_comprehensive_summary_report()
        
        print(f"\n✅ Complete multi-dataset analysis finished!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Generated analyses:")
        print(f"   • Feature importance comparison across datasets")
        print(f"   • Poster-friendly feature importance visualization")
        print(f"   • Rank improvement analysis and visualizations")
        print(f"   • Dataset statistics comparison")
        print(f"   • Comprehensive summary report")
        
        return {
            'feature_importance': feature_importance_df,
            'rank_improvement': rank_improvement_df,
            'dataset_statistics': dataset_stats_df
        }


def main():
    """Main execution function."""
    print("🎯 MENCOD Multi-Dataset Analysis Tool")
    print("="*60)
    
    # Create analyzer and run complete analysis
    analyzer = MultiDatasetAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print(f"\n🎉 Multi-dataset analysis complete!")


if __name__ == "__main__":
    main() 