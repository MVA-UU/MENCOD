# SPECTER2 Embedding Generation Script

This script generates SPECTER2 embeddings for scientific documents from CSV files containing titles and abstracts.

## Installation

First, install the required dependencies:

```bash
pip install torch transformers adapters pandas numpy tqdm
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python generate_specter2_embeddings.py --input_csv data/your_papers.csv --output_dir embeddings/
```

### Testing with Limited Documents

```bash
# Test with only 5 documents (great for local testing)
python generate_specter2_embeddings.py --input_csv data/your_papers.csv --output_dir embeddings/ --doc_limit 5
```

### Advanced Usage

```bash
# Use GPU and custom batch size
python generate_specter2_embeddings.py \
    --input_csv data/your_papers.csv \
    --output_dir embeddings/ \
    --batch_size 32 \
    --device cuda \
    --filename_prefix my_dataset
```

### Command Line Arguments

- `--input_csv`: Path to input CSV file containing documents with 'title' and 'abstract' columns (required)
- `--output_dir`: Directory to save the generated embeddings and metadata (required)
- `--batch_size`: Batch size for processing documents (default: 16)
- `--device`: Device to use for computation ('cpu', 'cuda', or 'auto', default: 'auto')
- `--filename_prefix`: Prefix for output filenames (default: 'specter2')
- `--doc_limit`: Maximum number of documents to process (useful for testing, default: no limit)

## Input CSV Format

Your CSV file should contain at least the following columns:
- `title`: The title of the scientific paper
- `abstract`: The abstract of the scientific paper

Example CSV structure:
```csv
title,abstract,other_columns...
"BERT: Pre-training of Deep Bidirectional Transformers","We introduce a new language representation model...",
"Attention Is All You Need","The dominant sequence transduction models are based...",
```

**Note**: Documents with both missing title AND abstract will be skipped. Documents with only one missing field will still be processed.

## Output Files

The script generates three output files:

1. **`{prefix}_embeddings.npy`**: NumPy array containing the embeddings
   - Shape: `(n_documents, 768)` where 768 is the SPECTER2 embedding dimension
   - Documents that were skipped have NaN values

2. **`{prefix}_metadata.json`**: JSON file containing metadata about the embeddings
   - Document-level information about which documents have embeddings
   - Model configuration details
   - Summary statistics

3. **`{prefix}_summary.txt`**: Human-readable summary of the embedding generation process

## Using the Generated Embeddings

### Loading Embeddings in Python

```python
import numpy as np
import json

# Load embeddings
embeddings = np.load('embeddings/specter2_embeddings.npy')

# Load metadata
with open('embeddings/specter2_metadata.json', 'r') as f:
    metadata = json.load(f)

# Filter out documents without embeddings
valid_mask = ~np.isnan(embeddings[:, 0])
valid_embeddings = embeddings[valid_mask]

print(f"Total documents: {len(embeddings)}")
print(f"Valid embeddings: {len(valid_embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")
```

### Computing Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(valid_embeddings)

# Find most similar papers to the first paper
similarities = cosine_similarity(valid_embeddings[0:1], valid_embeddings)[0]
most_similar_indices = np.argsort(similarities)[::-1]  # Sort by descending similarity

print("Most similar papers to the first paper:")
for i, idx in enumerate(most_similar_indices[:5]):  # Top 5 similar papers
    print(f"{i+1}. Document {idx}: similarity = {similarities[idx]:.4f}")
```

## Model Information

This script uses:
- **Model**: `allenai/specter2_base` (base SPECTER2 model)
- **Adapter**: `allenai/specter2` (proximity adapter - general purpose)
- **Input Format**: Title + [SEP] + Abstract (concatenated with SPECTER2's separator token)
- **Output**: 768-dimensional embeddings optimized for document similarity tasks

## Performance Considerations

- **GPU Usage**: The script automatically detects and uses GPU if available
- **Batch Processing**: Processes documents in batches to manage memory usage
- **Memory Requirements**: Approximately 2-4GB GPU memory for batch size 16-32
- **Speed**: ~100-500 documents per minute depending on hardware and batch size

## Troubleshooting

### Common Issues

1. **ImportError for 'adapters'**: Install the adapters library: `pip install adapters`
2. **CUDA out of memory**: Reduce batch size with `--batch_size 8`
3. **Model download fails**: Ensure internet connection for first-time model download
4. **Missing columns**: Script will try to auto-detect similar column names (e.g., 'Title' instead of 'title')

### Column Name Variations

The script automatically handles common variations:
- `Title` → `title`
- `Abstract` → `abstract` 
- Case-insensitive matching

## Example Workflow

```bash
# 1. Generate embeddings for your dataset
python generate_specter2_embeddings.py \
    --input_csv data/hall.csv \
    --output_dir embeddings/hall/ \
    --filename_prefix hall_papers

# 2. The script will create:
#    - embeddings/hall/hall_papers_embeddings.npy
#    - embeddings/hall/hall_papers_metadata.json
#    - embeddings/hall/hall_papers_summary.txt

# 3. Use embeddings in your citation network analysis
python your_analysis_script.py --embeddings embeddings/hall/hall_papers_embeddings.npy
```

This workflow integrates well with citation network analysis where you need document similarity features for ranking, clustering, or recommendation tasks. 