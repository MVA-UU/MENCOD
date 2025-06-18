# Enriched Citation Network - Supercomputer Implementation

This repository contains a multithreaded implementation of the citation network model for outlier detection in academic literature, optimized for high-performance computing environments.

## Features

### Multiprocessing Optimizations
- **Parallel Semantic Similarity Calculation**: The most computationally intensive part (TF-IDF similarity calculation) is parallelized across multiple CPU cores
- **Parallel Feature Extraction**: Feature extraction for large document sets is distributed across multiple processes
- **Parallel Relevance Scoring**: Relevance score calculation is parallelized for large datasets
- **Adaptive Batch Processing**: Automatically adjusts batch sizes based on available cores and dataset size

### External Data Integration
- **OpenAlex Data Enrichment**: Incorporates external citation data from OpenAlex to create more complete citation networks
- **Configurable External Data**: Can enable/disable external data enrichment based on requirements
- **Safe Outlier Handling**: Ensures external documents cannot be identified as outliers (only used for network enrichment)

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- `networkx` - Graph processing
- `pandas` - Data manipulation  
- `numpy` - Numerical computing
- `scikit-learn` - TF-IDF vectorization and similarity
- `multiprocessing` - Parallel processing (built-in)

## Quick Start

### Command-Line Interface

The recommended way to run the analysis on supercomputers:

```bash
# Basic usage with all available cores
python run_enriched_citation_network.py --dataset appenzeller

# Specify number of cores explicitly 
python run_enriched_citation_network.py --dataset appenzeller --cores 32

# Full analysis on entire search pool (computationally intensive)
python run_enriched_citation_network.py --dataset appenzeller --cores 64 --full-analysis

# Disable external data for faster processing
python run_enriched_citation_network.py --dataset appenzeller --cores 16 --no-external

# Custom sample size
python run_enriched_citation_network.py --dataset appenzeller --cores 8 --sample-size 5000
```

### Available Datasets

- `appenzeller` - Appenzeller-Herzog 2019 dataset
- `hall` - Hall 2012 dataset  
- `jeyaraman` - Jeyaraman 2020 dataset
- `valk` - van der Valk 2021 dataset

### Python API

```python
from CitationNetwork.citation_network import CitationNetworkModel

# Initialize with specific number of cores
model = CitationNetworkModel('appenzeller', n_cores=32)

# Build network with external data enrichment
model.fit(include_external=True)

# Extract features and calculate scores
features = model.get_citation_features(document_list)
scores = model.predict_relevance_scores(document_list)
```

## Supercomputer Usage

### SLURM Job Script Example

```bash
#!/bin/bash
#SBATCH --job-name=citation_network
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=citation_network_%j.out
#SBATCH --error=citation_network_%j.err

# Load Python module (adjust for your system)
module load Python/3.9.6

# Run the analysis
python run_enriched_citation_network.py \
    --dataset appenzeller \
    --cores $SLURM_CPUS_PER_TASK \
    --full-analysis \
    --output results_${SLURM_JOB_ID}.json \
    --features-output features_${SLURM_JOB_ID}.csv \
    --scores-output scores_${SLURM_JOB_ID}.csv \
    --verbose
```

### PBS Job Script Example

```bash
#!/bin/bash
#PBS -N citation_network
#PBS -l nodes=1:ppn=32
#PBS -l mem=64gb
#PBS -l walltime=04:00:00
#PBS -o citation_network.out
#PBS -e citation_network.err

cd $PBS_O_WORKDIR

# Load Python environment
source activate citation_env

# Run with PBS core count
python run_enriched_citation_network.py \
    --dataset appenzeller \
    --cores $PBS_NP \
    --full-analysis \
    --verbose
```

## Performance Guidelines

### Memory Requirements

- **Small datasets** (< 3,000 docs): 8-16 GB RAM
- **Medium datasets** (3,000-10,000 docs): 16-32 GB RAM  
- **Large datasets** (10,000+ docs): 32-64 GB RAM
- **With external data**: Add 50-100% more memory

### CPU Core Scaling

The implementation scales well up to:
- **Semantic similarity**: Scales linearly up to ~64 cores
- **Feature extraction**: Scales linearly up to ~32 cores
- **Network building**: Limited by I/O, optimal at 8-16 cores

### Recommended Configurations

**For fast prototyping:**
```bash
python run_enriched_citation_network.py --dataset appenzeller --cores 8 --sample-size 1000
```

**For full analysis:**
```bash
python run_enriched_citation_network.py --dataset appenzeller --cores 32 --full-analysis
```

**For maximum performance (large systems):**
```bash
python run_enriched_citation_network.py --dataset appenzeller --cores 64 --full-analysis
```

## Output Files

### Results JSON (`citation_network_results.json`)
Contains comprehensive analysis results including:
- Dataset and parameter information
- Network statistics
- Performance timing
- Outlier detection results

### Features CSV (`citation_features.csv`)
All extracted citation-based features for analyzed documents:
- Connectivity features (degree, citations, etc.)
- Coupling features (bibliographic coupling, co-citation)
- Neighborhood features (local network properties)
- Temporal features (citation patterns over time)
- Efficiency features (network position metrics)

### Scores CSV (`relevance_scores.csv`)
Final relevance scores for all analyzed documents, sorted by score.

## Implementation Details

### Parallelization Strategy

1. **Semantic Similarity Calculation**
   - Documents split into batches across cores
   - Each core calculates similarity matrix for its batch
   - Results merged and edges added to graph

2. **Feature Extraction**
   - Documents distributed across cores
   - Each core extracts features for its batch
   - Features combined into final DataFrame

3. **Score Calculation**
   - Large document sets split across cores
   - Each core calculates component scores
   - Final scores weighted and normalized

### Memory Optimization

- **Sparse Matrix Operations**: Uses scipy sparse matrices for TF-IDF
- **Batch Processing**: Processes large datasets in memory-efficient batches
- **Incremental Graph Building**: Builds graph incrementally to manage memory
- **Feature Caching**: Caches computed features to avoid recomputation

### Error Handling

- **Graceful Fallback**: Falls back to single-threaded processing if multiprocessing fails
- **Memory Management**: Automatically adjusts batch sizes if memory issues occur
- **Progress Reporting**: Provides detailed progress information for long-running jobs

## Troubleshooting

### Common Issues

**"Multiprocessing failed" errors:**
- Try reducing `--cores` parameter
- Use `--verbose` flag for detailed error information
- Check available memory

**Out of memory errors:**
- Reduce `--sample-size` for initial testing
- Use `--no-external` to disable external data
- Increase job memory allocation

**Slow performance:**
- Ensure you're using multiple cores (`--cores`)
- Consider using `--sample-size` instead of `--full-analysis`
- Check if external data files are accessible

### Performance Monitoring

Monitor your jobs using:
```bash
# CPU usage
top -u $USER

# Memory usage  
free -h

# Job status (SLURM)
squeue -u $USER
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{enriched_citation_network,
  title={Enriched Citation Network for Academic Outlier Detection},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]}
}
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Run with `--verbose` flag for detailed error information
3. Test with smaller datasets first (`--sample-size 100`)
4. Verify your supercomputer environment supports multiprocessing 