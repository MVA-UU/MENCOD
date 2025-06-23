# MENCOD Evaluation Suite

Professional visualization and analysis toolkit for the MENCOD (Multi-method Extended Citation Network Outlier Detection) system, designed for master's thesis integration.

## Overview

This evaluation suite provides comprehensive analysis and visualization tools for MENCOD's outlier detection results across multiple datasets. It generates publication-ready figures and detailed statistical comparisons suitable for academic research.

## Features

### Individual Dataset Analysis
- **KDE plots** for each scoring method (LOF embeddings, LOF network, LOF mixed, Isolation Forest, Ensemble)
- **Feature importance** analysis using Random Forest regression
- **Combined normalized KDE plots** showing outlier positions across all methods
- **Outlier performance summaries** with ranking and percentile information

### Multi-Dataset Comparison
- **Feature importance heatmap** comparing importance across all datasets
- **Rank improvement analysis** showing algorithm performance gains
- **Dataset statistics comparison** highlighting dataset characteristics
- **Comprehensive summary reports** for thesis integration

## Scripts

### 1. `dataset_visualizer.py`
Analyzes individual datasets with interactive dataset selection.

```bash
python Evaluation/dataset_visualizer.py
```

**Outputs for each dataset:**
- `results_[dataset]/kde_plots_individual.png` - Individual KDE plots
- `results_[dataset]/feature_importance.png` - Feature importance visualization
- `results_[dataset]/combined_kde_plot.png` - Combined normalized KDE plot
- `results_[dataset]/feature_importance.csv` - Feature importance data
- `results_[dataset]/outlier_summary.csv` - Outlier performance summary

### 2. `multi_dataset_analyzer.py`
Performs comprehensive comparison across all available datasets.

```bash
python Evaluation/multi_dataset_analyzer.py
```

**Outputs:**
- `multi_dataset_results/feature_importance_comparison.csv` - Feature importance comparison table
- `multi_dataset_results/feature_importance_heatmap.png` - Feature importance heatmap
- `multi_dataset_results/rank_improvement_analysis.csv` - Rank improvement analysis
- `multi_dataset_results/rank_improvements_visualization.png` - Rank improvement plots
- `multi_dataset_results/dataset_statistics_comparison.csv` - Dataset characteristics table
- `multi_dataset_results/dataset_statistics_visualization.png` - Dataset statistics plots
- `multi_dataset_results/comprehensive_summary_report.txt` - Summary report

### 3. `run_complete_evaluation.py`
Master script that runs all analyses automatically.

```bash
python Evaluation/run_complete_evaluation.py
```

**Complete workflow:**
1. Analyzes each dataset individually
2. Performs multi-dataset comparisons
3. Generates comprehensive summary
4. Creates thesis-ready documentation

## Output Structure

```
Evaluation/
├── results_jeyaraman/
│   ├── kde_plots_individual.png
│   ├── feature_importance.png
│   ├── combined_kde_plot.png
│   ├── feature_importance.csv
│   └── outlier_summary.csv
├── results_appenzeller/
│   ├── kde_plots_individual.png
│   ├── feature_importance.png
│   ├── combined_kde_plot.png
│   ├── feature_importance.csv
│   └── outlier_summary.csv
├── results_hall/
│   ├── kde_plots_individual.png
│   ├── feature_importance.png
│   ├── combined_kde_plot.png
│   ├── feature_importance.csv
│   └── outlier_summary.csv
├── multi_dataset_results/
│   ├── feature_importance_comparison.csv
│   ├── feature_importance_heatmap.png
│   ├── rank_improvement_analysis.csv
│   ├── rank_improvements_visualization.png
│   ├── dataset_statistics_comparison.csv
│   ├── dataset_statistics_visualization.png
│   └── comprehensive_summary_report.txt
└── evaluation_summary.txt
```

## Visualization Details

### KDE Plots
- **Individual plots**: Show distribution of each scoring method with outliers highlighted
- **Combined plots**: Normalized comparison of all methods on the same scale
- **Professional styling**: Clean, thesis-ready appearance with proper labels

### Feature Importance
- **Random Forest-based**: Uses ensemble scores as target variable
- **Horizontal bar charts**: Easy-to-read importance rankings
- **Percentage values**: Industry-standard importance metrics
- **Cross-dataset heatmap**: Comparative importance across datasets

### Statistical Comparisons
- **Rank improvements**: Shows algorithm performance gains
- **Dataset characteristics**: Document counts, relevance ratios, network properties
- **Performance metrics**: Detailed outlier detection statistics

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- MENCOD system (local)

## Usage Tips

### For Thesis Writing
1. Use individual dataset plots for detailed method analysis
2. Use multi-dataset comparisons for general conclusions
3. Reference CSV files for exact statistical values
4. All plots are 300 DPI for publication quality

### Customization
- Modify color schemes in script headers
- Adjust figure sizes in plotting functions
- Change statistical methods in analyzer classes
- Add new datasets by updating `data/datasets.json`

## Scoring Methods Analyzed

1. **LOF Embeddings**: Semantic outlier detection using SPECTER2 embeddings
2. **LOF Network**: Structural outlier detection using network features
3. **LOF Mixed**: Hybrid approach combining embeddings and network features
4. **Isolation Forest**: Global anomaly detection
5. **Ensemble**: Weighted combination of all methods

## Network Features

The system analyzes comprehensive network features including:
- Degree centrality measures (in-degree, out-degree, total degree)
- Advanced centrality metrics (PageRank, betweenness, closeness, eigenvector)
- Local network properties (clustering coefficient, neighbor diversity)
- Citation patterns and semantic relationships

## Academic Integration

This evaluation suite is designed for academic research and thesis integration:
- Publication-quality figures (300 DPI, clean styling)
- Statistical rigor (Random Forest feature importance, normalized comparisons)
- Comprehensive documentation and reproducibility
- Professional presentation suitable for academic defense

---

**Note**: This evaluation suite is part of the MENCOD project for master's thesis research on citation network outlier detection. For questions or issues, refer to the main project documentation. 