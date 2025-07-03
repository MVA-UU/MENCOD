# MENCOD: Multi-modal ENsemble Citation Outlier Detection
Master thesis project by Marco van Angeren  
Utrecht University - Applied Data Science  
In collaboration with ASReview - https://www.asreview.nl

## Project Overview
This repository contains the implementation and evaluation of MENCOD (Multi-modal ENsemble Citation Outlier Detection), a novel approach to detecting outlier documents in citation networks using a combination of semantic and network-based features. The model reranks a dataset on the fly, aiming to prioritize the target relevant document. The project was developed as part of a Master's thesis in Applied Data Science at Utrecht University, in collaboration with ASReview.

## MENCOD Package
MENCOD is a Python package that combines semantic similarity analysis with network structure analysis to identify outlier documents in citation networks. It leverages:
- SPECTER2 embeddings for semantic document representation
- Local Outlier Factor (LOF) for semantic outlier detection
- Isolation Forest for structural anomaly detection
- Ensemble methods for robust final scoring

### Installation
```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt
```

### Usage
```python
from MENCOD import CitationNetworkOutlierDetector
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

## Project Structure
```
.
├── MENCOD/                     # Main package implementation
│   ├── core.py                # Main CitationNetworkOutlierDetector class
│   ├── network.py             # Citation network construction
│   ├── features.py            # Feature extraction
│   ├── detection.py           # Outlier detection algorithms
│   ├── evaluation.py          # Results analysis
│   └── main.py                # Standalone script
├── data/                      # Data directory
│   ├── datasets.json          # Dataset configurations
│   ├── embeddings/            # SPECTER2 embeddings
│   ├── simulations/           # Simulation results
│   └── synergy_dataset/       # Original datasets
└── Misc/                      # Evaluation and utility scripts
    ├── create_recall_plots.py # Create recall plots to illustrate improvements
    ├── generate_embeddings.py # Generate embeddings using SPECTER2 model
    ├── get_original_ranks.py  # Script to retrieve original rank of the outlier in the Synergy dataset
    ├── run_simulation.py      # Run ASReview simulation (in order to get the simulation left-over dataset)
    └── visualize.py           # Another visualization script
```

## Requirements
```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
networkx>=2.6.0
wordcloud>=1.8.0
synergy-dataset>=0.1.0
torch>=1.9.0
transformers>=4.21.0
adapters>=3.2.0
tqdm>=4.62.0
```

## Disclaimer
- This project actively utilized AI assistance in code generation and development. Specifically, large language models were used to assist in code writing, debugging, and documentation. Mainly Anthropic Claude Sonnet 3.7 and 4.0.
- The evaluation scripts in the `Misc/` directory are experimental and may contain irregularities. They were developed primarily for research purposes and may not follow strict software engineering practices.
- The main MENCOD package is properly structured and tested, but the evaluation scripts should be used with caution and may require modifications for different use cases. Due to time limitations, these are made specifically to fit the use-case of this thesis research.

## Acknowledgments
- Utrecht University
- ASReview Team
- Zeyu Ding
- Rens van de Schoot
