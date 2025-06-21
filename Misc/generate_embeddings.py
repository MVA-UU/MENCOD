#!/usr/bin/env python3
"""
Generate SPECTER2 Embeddings for Scientific Documents

This script loads a CSV file containing scientific documents with 'title' and 'abstract' fields,
generates embeddings using SPECTER2 (proximity adapter), and saves the embeddings for later use.

Usage:
    # Using a direct CSV file
    python generate_specter2_embeddings.py --input_csv path/to/documents.csv --output_dir embeddings/
    
    # Using a synergy dataset (interactive selection)
    python generate_specter2_embeddings.py --synergy-dataset --output_dir embeddings/

The script will:
1. Load documents from the specified CSV file or synergy dataset
2. Concatenate title and abstract fields (handling missing values)
3. Generate embeddings using SPECTER2 with the proximity adapter
4. Save embeddings to numpy files and metadata to JSON

Requirements:
    pip install transformers adapters pandas numpy torch
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import warnings
import glob

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install the required packages:")
    print("pip install transformers adapters pandas numpy torch")
    exit(1)


class SPECTER2EmbeddingGenerator:
    """Generator for SPECTER2 embeddings."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the SPECTER2 embedding generator.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print("Loading SPECTER2 tokenizer and model...")
        self.tokenizer = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the SPECTER2 model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
            
            # Load base model
            self.model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
            
            # Load the proximity adapter (general purpose)
            self.model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print("✓ SPECTER2 model loaded successfully")
            
        except Exception as e:
            print(f"Error loading SPECTER2 model: {e}")
            print("This might be due to missing dependencies or network issues.")
            print("Make sure you have internet connection for the first run to download the model.")
            raise
    
    def prepare_texts(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Prepare text inputs by concatenating title and abstract.
        
        Args:
            documents: List of documents with 'title' and 'abstract' fields
            
        Returns:
            List of prepared text strings
        """
        prepared_texts = []
        skipped_count = 0
        
        for doc in documents:
            title = doc.get('title', '').strip() if pd.notna(doc.get('title')) else ''
            abstract = doc.get('abstract', '').strip() if pd.notna(doc.get('abstract')) else ''
            
            # Skip if both title and abstract are missing
            if not title and not abstract:
                skipped_count += 1
                prepared_texts.append(None)  # Placeholder for skipped documents
                continue
            
            # Concatenate title and abstract with SEP token
            text = title + self.tokenizer.sep_token + abstract
            prepared_texts.append(text)
        
        if skipped_count > 0:
            print(f"Warning: Skipped {skipped_count} documents with missing title and abstract")
        
        return prepared_texts
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        # Filter out None values (skipped documents)
        valid_indices = [i for i, text in enumerate(texts) if text is not None]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            return np.array([])
        
        embeddings = []
        
        print(f"Generating embeddings for {len(valid_texts)} documents...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(valid_texts), batch_size), desc="Processing batches"):
                batch_texts = valid_texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    max_length=512
                )
                
                # Move inputs to device
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                
                # Get model output
                output = self.model(**inputs)
                
                # Extract embeddings (first token embeddings)
                batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        if embeddings:
            all_embeddings = np.vstack(embeddings)
        else:
            all_embeddings = np.array([])
        
        # Create full embedding array with placeholders for skipped documents
        if len(valid_indices) < len(texts):
            embedding_dim = all_embeddings.shape[1] if all_embeddings.size > 0 else 768
            full_embeddings = np.full((len(texts), embedding_dim), np.nan)
            full_embeddings[valid_indices] = all_embeddings
            return full_embeddings
        
        return all_embeddings


def load_csv_documents(csv_path: str, doc_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load documents from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        doc_limit: Maximum number of documents to load (None for no limit)
        
    Returns:
        List of document dictionaries
    """
    try:
        df = pd.read_csv(csv_path)
        total_docs = len(df)
        
        # Apply document limit if specified
        if doc_limit is not None and doc_limit > 0:
            df = df.head(doc_limit)
            print(f"✓ Loaded {len(df)} documents from {csv_path} (limited from {total_docs} total)")
        else:
            print(f"✓ Loaded {len(df)} documents from {csv_path}")
        
        # Check for required columns
        required_cols = ['title', 'abstract']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            
            # Try alternative column names
            col_mapping = {}
            for col in missing_cols:
                if col == 'title':
                    alternatives = [c for c in df.columns if 'title' in c.lower()]
                elif col == 'abstract':
                    alternatives = [c for c in df.columns if 'abstract' in c.lower()]
                
                if alternatives:
                    col_mapping[col] = alternatives[0]
                    print(f"Using '{alternatives[0]}' for '{col}'")
            
            # Rename columns
            df = df.rename(columns={v: k for k, v in col_mapping.items()})
        
        return df.to_dict('records')
        
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise


def load_synergy_dataset(synergy_name: str, doc_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load a synergy dataset by name using the synergy-dataset package.
    
    Args:
        synergy_name: Name of the synergy dataset (e.g., 'Muthu_2021')
        doc_limit: Maximum number of documents to load (None for no limit)
        
    Returns:
        List of document dictionaries including openalex_id
    """
    try:
        # Import synergy_dataset package
        from synergy_dataset import Dataset
        
        # Load dataset using synergy_dataset package
        print(f"Loading synergy dataset: {synergy_name}")
        synergy_dataset = Dataset(synergy_name)
        df = synergy_dataset.to_frame()
        
        # Get the OpenAlex IDs from the synergy dataset
        synergy_dict = synergy_dataset.to_dict()
        openalex_ids = list(synergy_dict.keys())
        
        # Add record_id and openalex_id columns to the DataFrame
        df['record_id'] = range(len(df))
        df['openalex_id'] = openalex_ids
        
        total_docs = len(df)
        
        # Apply document limit if specified
        if doc_limit is not None and doc_limit > 0:
            df = df.head(doc_limit)
            print(f"✓ Loaded {len(df)} documents from {synergy_name} (limited from {total_docs} total)")
        else:
            print(f"✓ Loaded {len(df)} documents from {synergy_name}")
        
        print(f"Dataset columns: {list(df.columns)}")
        print(f"Sample OpenAlex IDs: {openalex_ids[:3]}")
        
        # Convert to list of dictionaries
        documents = df.to_dict('records')
        
        return documents
        
    except ImportError:
        print("Error: synergy_dataset package not found. Please install it with: pip install synergy-dataset")
        raise
    except Exception as e:
        print(f"Failed to load dataset {synergy_name}: {e}")
        raise


def save_embeddings(embeddings: np.ndarray, documents: List[Dict[str, Any]], 
                   output_dir: str, filename_prefix: str = "specter2"):
    """
    Save embeddings and metadata to files.
    
    Args:
        embeddings: Numpy array of embeddings
        documents: List of document dictionaries
        output_dir: Output directory
        filename_prefix: Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings as numpy file
    embeddings_path = os.path.join(output_dir, f"{filename_prefix}_embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"✓ Saved embeddings to {embeddings_path}")
    
    # Prepare metadata
    metadata = {
        'num_documents': len(documents),
        'embedding_dim': embeddings.shape[1] if embeddings.size > 0 else 0,
        'model_name': 'allenai/specter2',
        'adapter': 'proximity',
        'valid_embeddings': int(np.sum(~np.isnan(embeddings[:, 0]))) if embeddings.size > 0 else 0
    }
    
    # Save document metadata (without embeddings to save space)
    doc_metadata = []
    for i, doc in enumerate(documents):
        doc_meta = {
            'index': i,
            'has_title': bool(doc.get('title', '').strip() if pd.notna(doc.get('title')) else ''),
            'has_abstract': bool(doc.get('abstract', '').strip() if pd.notna(doc.get('abstract')) else ''),
            'has_embedding': not np.isnan(embeddings[i, 0]) if embeddings.size > 0 else False
        }
        
        # Include some identifiers if available
        for field in ['title', 'doi', 'openalex_id', 'id']:
            if field in doc and pd.notna(doc[field]):
                doc_meta[field] = str(doc[field])[:100]  # Truncate long fields
        
        doc_metadata.append(doc_meta)
    
    metadata['documents'] = doc_metadata
    
    # Save metadata as JSON
    metadata_path = os.path.join(output_dir, f"{filename_prefix}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved metadata to {metadata_path}")
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, f"{filename_prefix}_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"SPECTER2 Embeddings Summary\n")
        f.write(f"==========================\n\n")
        f.write(f"Total documents: {metadata['num_documents']}\n")
        f.write(f"Valid embeddings: {metadata['valid_embeddings']}\n")
        f.write(f"Embedding dimension: {metadata['embedding_dim']}\n")
        f.write(f"Model: {metadata['model_name']}\n")
        f.write(f"Adapter: {metadata['adapter']}\n")
        
        if embeddings.size > 0:
            valid_mask = ~np.isnan(embeddings[:, 0])
            valid_embeddings = embeddings[valid_mask]
            if len(valid_embeddings) > 0:
                f.write(f"\nEmbedding Statistics:\n")
                f.write(f"Mean norm: {np.mean(np.linalg.norm(valid_embeddings, axis=1)):.4f}\n")
                f.write(f"Std norm: {np.std(np.linalg.norm(valid_embeddings, axis=1)):.4f}\n")
    
    print(f"✓ Saved summary to {summary_path}")


def get_synergy_datasets() -> List[Tuple[str, str]]:
    """
    Get list of available synergy datasets.
    
    Returns:
        List of tuples (dataset_name, dataset_name) - using dataset name as both identifier and value
    """
    # Try different possible paths for the synergy dataset directory (fallback for CSV files)
    possible_paths = [
        "data/synergy_dataset",  # From project root
        "../data/synergy_dataset",  # From Misc/ directory
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synergy_dataset")  # Relative to script location
    ]
    
    synergy_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            synergy_dir = path
            break
    
    if synergy_dir is None:
        print(f"Error: Synergy dataset directory not found in any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        return []
    
    # Get all CSV files in the synergy dataset directory
    csv_files = glob.glob(os.path.join(synergy_dir, "*.csv"))
    
    # Extract dataset names (remove path and .csv extension)
    datasets = []
    for file_path in sorted(csv_files):
        dataset_name = os.path.basename(file_path).replace('.csv', '')
        datasets.append((dataset_name, dataset_name))  # Use dataset name for both display and loading
    
    return datasets


def select_synergy_dataset() -> Optional[str]:
    """
    Interactive selection of synergy dataset.
    
    Returns:
        Selected dataset name, or None if cancelled
    """
    datasets = get_synergy_datasets()
    
    if not datasets:
        print("No synergy datasets found!")
        return None
    
    print("\n" + "=" * 60)
    print("AVAILABLE SYNERGY DATASETS")
    print("=" * 60)
    
    for i, (dataset_name, _) in enumerate(datasets, 1):
        print(f"{i:2d}. {dataset_name}")
    
    print("=" * 60)
    
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(datasets)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Selection cancelled.")
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(datasets):
                selected_dataset = datasets[choice_num - 1]
                print(f"\n✓ Selected: {selected_dataset[0]}")
                return selected_dataset[0]  # Return dataset name, not file path
            else:
                print(f"Please enter a number between 1 and {len(datasets)}")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nSelection cancelled.")
            return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate SPECTER2 embeddings for scientific documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using a direct CSV file
    python generate_specter2_embeddings.py --input_csv data/papers.csv --output_dir embeddings/

    # Using synergy dataset (interactive selection)
    python generate_specter2_embeddings.py --synergy-dataset --output_dir embeddings/

    # With custom batch size and device
    python generate_specter2_embeddings.py --input_csv data/papers.csv --output_dir embeddings/ --batch_size 32 --device cuda

    # Custom output filename prefix
    python generate_specter2_embeddings.py --input_csv data/papers.csv --output_dir embeddings/ --filename_prefix my_dataset
    
    # Test with limited documents (useful for local testing)
    python generate_specter2_embeddings.py --input_csv data/papers.csv --output_dir embeddings/ --doc_limit 5
        """
    )
    
    # Create mutually exclusive group for input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    input_group.add_argument(
        '--input_csv',
        type=str,
        help='Path to input CSV file containing documents with title and abstract columns'
    )
    
    input_group.add_argument(
        '--synergy-dataset',
        action='store_true',
        help='Select from available synergy datasets interactively'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save the generated embeddings and metadata'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for processing documents (default: 16)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for computation (default: auto)'
    )
    
    parser.add_argument(
        '--filename_prefix',
        type=str,
        default=None,
        help='Prefix for output filenames (default: auto-generated from dataset name)'
    )
    
    parser.add_argument(
        '--doc_limit',
        type=int,
        default=None,
        help='Maximum number of documents to process (useful for testing, default: no limit)'
    )
    
    args = parser.parse_args()
    
    # Determine input source
    input_csv_path = None
    synergy_dataset_name = None
    dataset_name = None
    
    if args.synergy_dataset:
        synergy_dataset_name = select_synergy_dataset()
        if synergy_dataset_name is None:
            return 1
        # Extract dataset name for auto-naming
        dataset_name = synergy_dataset_name.lower().replace('-', '_')
    else:
        input_csv_path = args.input_csv
        dataset_name = os.path.basename(input_csv_path).replace('.csv', '').lower().replace('-', '_')
        
        # Validate input file
        if not os.path.exists(input_csv_path):
            print(f"Error: Input CSV file '{input_csv_path}' does not exist")
            return 1
    
    # Set filename prefix
    filename_prefix = args.filename_prefix if args.filename_prefix else dataset_name
    
    if args.doc_limit is not None and args.doc_limit <= 0:
        print(f"Error: --doc_limit must be a positive integer (got {args.doc_limit})")
        return 1
    
    device = None if args.device == 'auto' else args.device
    
    try:
        print("=" * 60)
        print("SPECTER2 Embedding Generation")
        print("=" * 60)
        if synergy_dataset_name:
            print(f"Synergy dataset: {synergy_dataset_name}")
        else:
            print(f"Input file: {input_csv_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Filename prefix: {filename_prefix}")
        
        # Load documents
        print("\n1. Loading documents...")
        if synergy_dataset_name:
            documents = load_synergy_dataset(synergy_dataset_name, doc_limit=args.doc_limit)
        else:
            documents = load_csv_documents(input_csv_path, doc_limit=args.doc_limit)
        
        # Initialize embedding generator
        print("\n2. Initializing SPECTER2 model...")
        generator = SPECTER2EmbeddingGenerator(device=device)
        
        # Prepare texts
        print("\n3. Preparing text inputs...")
        texts = generator.prepare_texts(documents)
        
        # Generate embeddings
        print("\n4. Generating embeddings...")
        embeddings = generator.generate_embeddings_batch(texts, batch_size=args.batch_size)
        
        # Save results
        print("\n5. Saving results...")
        save_embeddings(embeddings, documents, args.output_dir, filename_prefix)
        
        print("\n" + "=" * 60)
        print("✓ Embedding generation completed successfully!")
        print(f"✓ Results saved to: {args.output_dir}")
        print(f"✓ Files created:")
        print(f"  - {filename_prefix}_embeddings.npy")
        print(f"  - {filename_prefix}_metadata.json")
        print(f"  - {filename_prefix}_summary.txt")
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 