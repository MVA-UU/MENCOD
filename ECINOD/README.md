# ECINOD - Extended CItation Network Outlier Detector

A streamlined outlier detection system for citation networks using Local Outlier Factor (LOF) on SPECTER2 embeddings and Isolation Forest for comprehensive anomaly detection.

## Overview

ECINOD combines semantic similarity analysis with network structure analysis to identify outlier documents in citation networks. It uses:

- **LOF on SPECTER2 embeddings** for semantic outlier detection
- **Isolation Forest** on network features for structural anomaly detection  
- **Ensemble methods** for robust final scoring

## Package Structure

```
ECINOD/
├── __init__.py          # Package initialization and public interface
├── core.py              # Main CitationNetworkOutlierDetector class
├── network.py           # Citation network construction (NetworkBuilder)
├── features.py          # Feature extraction (FeatureExtractor) 
├── detection.py         # Outlier detection algorithms (OutlierDetector)
├── evaluation.py        # Results analysis (ResultsAnalyzer)
├── main.py              # Standalone script
└── README.md            # This file
```

## Usage

### As a Package

```python
from ECINOD import CitationNetworkOutlierDetector
import pandas as pd

# Load your simulation data
simulation_df = pd.read_csv('your_data.csv')

# Initialize detector
detector = CitationNetworkOutlierDetector(random_state=42)

# Run outlier detection
results = detector.fit_predict_outliers(simulation_df, dataset_name='your_dataset')

# Get top outliers
outliers = detector.get_outlier_documents(method='ensemble', top_k=10)
```

### Convenience Function

```python
from ECINOD import detect_outliers

# Quick detection
results = detect_outliers(simulation_df, dataset_name='your_dataset')
```

### Standalone Script

```bash
python -m ECINOD.main
```

## Features

### Network Analysis
- Citation network construction from Synergy datasets
- Graph centrality measures (PageRank, betweenness, closeness, eigenvector)
- Neighborhood analysis and path-based features

### Semantic Analysis  
- SPECTER2 embedding-based similarity
- Cosine distance LOF for semantic outliers
- Isolation from relevant document clusters

### Ensemble Methods
- Variance-based weighting
- Normalized score combination
- Method comparison and analysis

## Components

### CitationNetworkOutlierDetector
Main orchestration class that coordinates all components.

### NetworkBuilder
Handles citation network construction from simulation data and Synergy datasets.

### FeatureExtractor
Extracts network features, semantic features, and citation metadata.

### OutlierDetector
Implements LOF and Isolation Forest algorithms with ensemble scoring.

### ResultsAnalyzer
Provides detailed analysis, ranking, and visualization of results.

## Requirements

- pandas, numpy
- scikit-learn
- networkx  
- tqdm
- scipy

## Output

The system provides:
- Individual method scores (LOF, Isolation Forest)
- Ensemble scores with data-driven weighting
- Detailed feature analysis for each document
- Performance evaluation against known outliers
- Formatted summaries and comparisons