"""
Complete MENCOD Evaluation Suite

This script runs the complete evaluation suite including:
- Individual dataset visualizations (KDE plots, feature importance, combined plots)
- Multi-dataset comparison analysis (feature importance, rank improvements, statistics)
- Professional thesis-ready visualizations
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import get_available_datasets
from dataset_visualizer import DatasetVisualizer
from multi_dataset_analyzer import MultiDatasetAnalyzer

def print_header():
    """Print professional header."""
    print("\n" + "="*80)
    print("MENCOD COMPLETE EVALUATION SUITE")
    print("Master's Thesis - Citation Network Outlier Detection")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def run_individual_dataset_analyses():
    """Run analysis for each individual dataset."""
    print("🔍 PHASE 1: INDIVIDUAL DATASET ANALYSES")
    print("-" * 50)
    
    available_datasets = get_available_datasets()
    individual_results = {}
    
    for i, dataset_name in enumerate(available_datasets, 1):
        print(f"\n[{i}/{len(available_datasets)}] Analyzing {dataset_name}...")
        
        try:
            start_time = time.time()
            visualizer = DatasetVisualizer(dataset_name)
            results = visualizer.run_complete_analysis()
            end_time = time.time()
            
            individual_results[dataset_name] = results
            print(f"✅ {dataset_name} completed in {end_time - start_time:.1f}s")
            
        except Exception as e:
            print(f"❌ Error analyzing {dataset_name}: {e}")
            continue
    
    print(f"\n✅ Individual dataset analyses complete!")
    print(f"📊 Analyzed {len(individual_results)} datasets")
    return individual_results

def run_multi_dataset_analysis():
    """Run multi-dataset comparison analysis."""
    print("\n🔗 PHASE 2: MULTI-DATASET COMPARISON ANALYSIS")
    print("-" * 50)
    
    try:
        start_time = time.time()
        analyzer = MultiDatasetAnalyzer()
        results = analyzer.run_complete_analysis()
        end_time = time.time()
        
        print(f"✅ Multi-dataset analysis completed in {end_time - start_time:.1f}s")
        return results
        
    except Exception as e:
        print(f"❌ Error in multi-dataset analysis: {e}")
        return None

def create_evaluation_summary():
    """Create overall evaluation summary."""
    print("\n📋 PHASE 3: CREATING EVALUATION SUMMARY")
    print("-" * 50)
    
    try:
        available_datasets = get_available_datasets()
        
        summary_lines = []
        summary_lines.append("MENCOD Complete Evaluation Summary")
        summary_lines.append("="*50)
        summary_lines.append(f"Evaluation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        summary_lines.append("Generated Analyses:")
        summary_lines.append("-"*20)
        summary_lines.append("Individual Dataset Analyses:")
        for dataset in available_datasets:
            summary_lines.append(f"  • {dataset.capitalize()}")
            summary_lines.append(f"    - KDE plots for all scoring methods")
            summary_lines.append(f"    - Feature importance analysis")
            summary_lines.append(f"    - Combined normalized KDE plot")
            summary_lines.append(f"    - Outlier performance summary")
        
        summary_lines.append("")
        summary_lines.append("Multi-Dataset Comparisons:")
        summary_lines.append("  • Feature importance comparison across datasets")
        summary_lines.append("  • Rank improvement analysis and visualization")
        summary_lines.append("  • Dataset characteristics comparison")
        summary_lines.append("  • Comprehensive summary report")
        
        summary_lines.append("")
        summary_lines.append("Output Structure:")
        summary_lines.append("-"*15)
        summary_lines.append("Evaluation/")
        for dataset in available_datasets:
            summary_lines.append(f"├── results_{dataset}/")
            summary_lines.append(f"│   ├── kde_plots_individual.png")
            summary_lines.append(f"│   ├── feature_importance.png")
            summary_lines.append(f"│   ├── combined_kde_plot.png")
            summary_lines.append(f"│   ├── feature_importance.csv")
            summary_lines.append(f"│   └── outlier_summary.csv")
        summary_lines.append("└── multi_dataset_results/")
        summary_lines.append("    ├── feature_importance_comparison.csv")
        summary_lines.append("    ├── feature_importance_heatmap.png")
        summary_lines.append("    ├── rank_improvement_analysis.csv")
        summary_lines.append("    ├── rank_improvements_visualization.png")
        summary_lines.append("    ├── dataset_statistics_comparison.csv")
        summary_lines.append("    ├── dataset_statistics_visualization.png")
        summary_lines.append("    └── comprehensive_summary_report.txt")
        
        summary_lines.append("")
        summary_lines.append("🎓 Ready for Master's Thesis Integration!")
        
        # Save summary
        evaluation_dir = "Evaluation"
        with open(os.path.join(evaluation_dir, "evaluation_summary.txt"), 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print("✅ Evaluation summary created")
        return summary_lines
        
    except Exception as e:
        print(f"❌ Error creating summary: {e}")
        return None

def print_final_results():
    """Print final results and instructions."""
    print("\n" + "="*80)
    print("🎉 MENCOD EVALUATION SUITE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\n📁 Results Location:")
    print("   • Individual dataset results: Evaluation/results_[dataset_name]/")
    print("   • Multi-dataset comparisons: Evaluation/multi_dataset_results/")
    print("   • Overall summary: Evaluation/evaluation_summary.txt")
    
    print("\n📊 Generated Visualizations:")
    print("   ✓ KDE plots for each scoring method (per dataset)")
    print("   ✓ Feature importance plots (per dataset)")
    print("   ✓ Combined normalized KDE plots (per dataset)")
    print("   ✓ Feature importance comparison heatmap (multi-dataset)")
    print("   ✓ Rank improvement visualizations (multi-dataset)")
    print("   ✓ Dataset statistics comparisons (multi-dataset)")
    
    print("\n📋 Generated Tables:")
    print("   ✓ Feature importance comparison table")
    print("   ✓ Rank improvement analysis table")
    print("   ✓ Dataset statistics comparison table")
    print("   ✓ Individual outlier summaries")
    
    print("\n🎓 Thesis Integration Tips:")
    print("   • All plots are high-resolution (300 DPI) and thesis-ready")
    print("   • Tables are in CSV format for easy LaTeX integration")
    print("   • Use individual dataset plots for detailed analysis")
    print("   • Use multi-dataset comparisons for overall conclusions")
    
    print("\n" + "="*80)

def main():
    """Main execution function."""
    print_header()
    
    total_start = time.time()
    
    # Phase 1: Individual dataset analyses
    individual_results = run_individual_dataset_analyses()
    
    # Phase 2: Multi-dataset analysis
    multi_dataset_results = run_multi_dataset_analysis()
    
    # Phase 3: Create evaluation summary
    summary = create_evaluation_summary()
    
    total_end = time.time()
    
    # Print final results
    print_final_results()
    
    print(f"\n⏱️  Total execution time: {total_end - total_start:.1f} seconds")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main() 