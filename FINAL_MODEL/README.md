# Hybrid Outlier Detection Model for Systematic Reviews

This is the final implementation of a hybrid outlier detection system designed to identify relevant documents that are missed by content-based ranking algorithms in systematic reviews. The system combines multiple detection approaches with adaptive weighting and continuous scaling for optimal generalizability.

## Features

### Core Components

1. **Citation Network Analysis** (`models/CitationNetwork/`)
   - GPU-accelerated graph processing using cuGraph
   - Semantic similarity integration via SPECTER2 embeddings
   - Multiple edge types: citations, co-citations, bibliographic coupling, semantic similarity
   - Advanced network metrics: centrality, clustering, PageRank

2. **Confidence Calibration** (`models/ConfidenceCalibration/`)
   - Ensemble-based overconfidence detection
   - Calibrated probability estimates
   - Model disagreement analysis
   - Uncertainty quantification

3. **Content Similarity Analysis** (`models/ContentSimilarity/`)
   - Specialized text pattern analysis
   - Methodology marker detection
   - Vocabulary diversity assessment
   - Technical term density analysis

### Key Features

- **Adaptive Weighting**: Automatically adjusts model weights based on dataset characteristics
- **Continuous Scaling**: No hard thresholds - all parameters use continuous functions
- **Modular Design**: Individual models can be enabled/disabled
- **GPU Acceleration**: Optional cuGraph support for large-scale networks
- **Semantic Integration**: SPECTER2 embeddings for enhanced similarity
- **Comprehensive Evaluation**: Built-in metrics and analysis tools

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)

### Dependencies

```bash
pip install -r requirements.txt
```

### GPU Setup (Optional)

For GPU acceleration, install RAPIDS cuGraph:

```bash
# CUDA 11.x
conda install -c rapidsai -c conda-forge -c nvidia cugraph=23.04 python=3.9 cudatoolkit=11.x

# CUDA 12.x
conda install -c rapidsai -c conda-forge -c nvidia cugraph=23.04 python=3.9 cudatoolkit=12.x
```

## Data Structure

The system expects data in the following structure:

```
data/
├── datasets.json                 # Dataset configuration
├── simulations/                  # Simulation results from ASReview
│   ├── appenzeller.csv
│   ├── hall.csv
│   ├── jeyaraman.csv
│   └── valk.csv
└── embeddings/                   # SPECTER2 embeddings (optional)
    ├── appenzeller.npy
    ├── appenzeller_metadata.json
    ├── hall.npy
    ├── hall_metadata.json
    └── ...
```

### Dataset Configuration

`datasets.json` format:
```json
{
    "dataset_name": {
        "outlier_ids": [414],
        "simulation_file": "dataset_name.csv",
        "synergy_dataset_name": "Dataset_Name_2020",
        "embeddings_filename": "dataset_name.csv",
        "embeddings_metadata_filename": "dataset_name_metadata.csv"
    }
}
```

### Simulation Data

Required columns in simulation CSV files:
- `openalex_id`: Unique document identifier
- `label_included`: Binary label (1 for relevant, 0 for irrelevant)
- `title`: Document title (optional but recommended)
- `abstract`: Document abstract (optional but recommended)

## Usage

### Quick Start

```python
from FINAL_MODEL.hybrid_model import HybridOutlierDetector

# Initialize with all models enabled
detector = HybridOutlierDetector(dataset_name="appenzeller")

# Load and fit on simulation data
detector.fit()

# Get documents to analyze
target_docs = ["W12345", "W67890", ...]

# Extract features
features = detector.extract_features(target_docs)

# Compute relevance scores
scores = detector.predict_relevance_scores(target_docs)

# Identify outliers with dynamic thresholding
outliers = detector.predict_outliers(target_docs)
```

### Command Line Interface

```bash
# Run demo mode
python run_hybrid_model.py --dataset appenzeller --mode demo

# Full evaluation with test/train split
python run_hybrid_model.py --dataset hall --mode evaluate --output results.json

# Interactive mode with guided configuration
python run_hybrid_model.py --mode interactive

# Custom configuration
python run_hybrid_model.py --dataset valk --disable-confidence --citation-weight 0.6 --content-weight 0.4
```

### Model Configuration

```python
from FINAL_MODEL.hybrid_model import ModelConfiguration, ModelWeights

# Custom configuration
config = ModelConfiguration(
    enable_citation_network=True,
    enable_confidence_calibration=False,
    enable_content_similarity=True,
    enable_gpu_acceleration=True,
    enable_semantic_embeddings=True
)

# Custom weights (if not using adaptive weighting)
weights = ModelWeights(
    citation_network=0.5,
    confidence_calibration=0.0,
    content_similarity=0.5
)

detector = HybridOutlierDetector(
    dataset_name="appenzeller",
    model_config=config,
    model_weights=weights,
    use_adaptive_weights=False
)
```

## Individual Model Usage

Each model can be used standalone:

### Citation Network

```python
from FINAL_MODEL.models.CitationNetwork import CitationNetworkModel

model = CitationNetworkModel(
    dataset_name="appenzeller",
    enable_gpu=True,
    enable_semantic=True
)
model.fit()

features = model.extract_features(target_docs)
scores = model.predict_relevance_scores(target_docs)
```

### Confidence Calibration

```python
from FINAL_MODEL.models.ConfidenceCalibration import ConfidenceCalibrationModel

model = ConfidenceCalibrationModel()
model.fit(simulation_df)

scores = model.predict_relevance_scores(target_docs)
```

### Content Similarity

```python
from FINAL_MODEL.models.ContentSimilarity import ContentSimilarityModel

model = ContentSimilarityModel(enable_semantic_embeddings=True)
model.fit(simulation_df, dataset_name="appenzeller")

scores = model.predict_relevance_scores(target_docs)
```

## Evaluation

### Built-in Evaluation

```python
from FINAL_MODEL.utils import create_evaluation_split, calculate_evaluation_metrics

# Split data
train_df, test_df = create_evaluation_split(simulation_df, test_size=0.2)

# Fit on training data
detector.fit(train_df)

# Evaluate on test data
test_docs = test_df['openalex_id'].tolist()
test_labels = test_df['label_included'].tolist()

scores = detector.predict_relevance_scores(test_docs)
score_values = [scores[doc_id] for doc_id in test_docs]

# Calculate metrics
metrics = calculate_evaluation_metrics(test_labels, score_values, threshold=0.5)
print(f"F1 Score: {metrics['f1_score']:.3f}")
print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
```

### Command Line Evaluation

```bash
# Comprehensive evaluation with optimal threshold finding
python run_hybrid_model.py --dataset appenzeller --mode evaluate --test-size 0.3 --output evaluation_results.json

# Results will include:
# - Optimal threshold based on F1 score
# - Performance metrics (Precision, Recall, F1, AUC-ROC)
# - Score distribution analysis
# - Model configuration and weights
```

## Advanced Features

### Adaptive Weighting

The system automatically adjusts model weights based on dataset characteristics:

- **Sparsity Factor**: Higher weight to citation network for sparse datasets
- **Text Richness**: Higher weight to content similarity for text-rich datasets
- **Dataset Size**: Scaling factors based on document count

### Dynamic Thresholding

Instead of fixed thresholds, the system uses adaptive thresholding:

```python
# Dynamic threshold based on score distribution and dataset sparsity
outliers = detector.predict_outliers(target_docs)  # threshold=None

# Or use fixed threshold
outliers = detector.predict_outliers(target_docs, threshold=0.7)
```

### GPU Acceleration

For large networks (>1000 edges), the system automatically uses GPU acceleration if available:

```python
# Check GPU availability
detector = HybridOutlierDetector(enable_gpu_acceleration=True)
status = detector.get_model_status()
print(f"GPU enabled: {status['model_configuration']['gpu_acceleration_enabled']}")
```

## Performance Considerations

### Memory Usage

- **Citation Network**: Memory scales with O(V + E) where V=vertices, E=edges
- **Embeddings**: ~768 floats per document for SPECTER2
- **GPU Memory**: 512MB pool allocated by default for cuGraph

### Runtime Optimization

- Networks <1000 edges: CPU processing
- Networks 1000-2M edges: GPU acceleration
- Networks >2M edges: May require distributed processing

### Batch Processing

For large document sets:

```python
# Process in batches
batch_size = 1000
all_scores = {}

for i in range(0, len(all_docs), batch_size):
    batch_docs = all_docs[i:i+batch_size]
    batch_scores = detector.predict_relevance_scores(batch_docs)
    all_scores.update(batch_scores)
```

## Troubleshooting

### Common Issues

1. **CUDA/cuGraph not found**
   ```
   cuGraph not available - using NetworkX on CPU
   ```
   - Install RAPIDS cuGraph for GPU acceleration
   - Or continue with CPU-only processing

2. **Embeddings not found**
   ```
   Embeddings not found for dataset_name, semantic features disabled
   ```
   - Generate SPECTER2 embeddings using `generate_specter2_embeddings.py`
   - Or disable semantic features

3. **Insufficient memory**
   - Reduce batch size
   - Disable GPU acceleration
   - Process smaller document sets

### Debug Mode

Enable verbose logging:

```bash
python run_hybrid_model.py --dataset appenzeller --verbose
```

Or in code:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hybrid_outlier_detection,
    title={Hybrid Outlier Detection for Systematic Reviews},
    author={Your Name},
    journal={Your Journal},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For questions or issues:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed description 