# ASReview Outlier Detection

This project analyzes ASReview simulation data to identify patterns and strategies for detecting outlier documents - relevant documents that are ranked low by conventional ranking algorithms.

## Problem Context

In systematic reviews, relevant documents are sometimes ranked very low by traditional algorithms. The goal of this project is to develop methods to identify these "outlier" documents without requiring a reviewer to scan through the entire dataset.

## Setup

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The main script provides different analysis options:

```bash
# Run all analyses
python src/main.py

# Run only citation network analysis
python src/main.py --citation

# Run only text analysis
python src/main.py --text
```

## Analysis Components

### Citation Analysis (`citation_analysis.py`)

Analyzes the citation network to find patterns related to outlier documents:
- Citation connections between relevant documents
- Bibliographic coupling (shared references)
- Network metrics (centrality, PageRank)
- Visualization of the citation network

### Text Analysis (`text_analysis.py`)

Analyzes textual content to identify distinctive features of outlier documents:
- Text similarity between the outlier and other documents
- Key terms that distinguish the outlier
- Topic modeling to analyze thematic differences
- Visualizations of document similarities and topic distributions

## Output

Analysis results are saved to the `output` directory:
- `output/figures/`: Visualizations and plots
- `output/citation_network.gexf`: Citation network in GEXF format for external analysis

## Results Interpretation

The primary goal is to identify patterns that can help detect outlier documents in future reviews:
1. Look for distinctive network metrics in the citation analysis results
2. Examine text similarity patterns and topic distributions
3. Consider bibliographic coupling as a potential strong signal

These insights can inform the development of a hybrid model for outlier detection in ASReview. 