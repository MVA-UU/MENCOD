"""
Utility functions for the Final Hybrid Model

This module provides common utilities for data loading, configuration management,
and evaluation metrics.
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_datasets_config() -> Dict[str, Any]:
    """Load datasets configuration from JSON file."""
    config_path = os.path.join(get_project_root(), 'data', 'datasets.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Datasets configuration not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_available_datasets() -> List[str]:
    """Get list of available dataset names."""
    config = load_datasets_config()
    return list(config.keys())


def load_simulation_data(dataset_name: str) -> pd.DataFrame:
    """
    Load simulation data for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        DataFrame with simulation data
    """
    project_root = get_project_root()
    simulation_path = os.path.join(project_root, 'data', 'simulations', f'{dataset_name}.csv')
    
    if not os.path.exists(simulation_path):
        raise FileNotFoundError(f"Simulation file not found: {simulation_path}")
    
    df = pd.read_csv(simulation_path)
    logger.info(f"Loaded simulation data for {dataset_name}: {len(df)} documents")
    
    return df


def load_embeddings(dataset_name: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Load SPECTER2 embeddings for a dataset.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Tuple of (embeddings array, metadata dict) or (None, None) if not found
    """
    project_root = get_project_root()
    embeddings_path = os.path.join(project_root, 'data', 'embeddings', f'{dataset_name}.npy')
    metadata_path = os.path.join(project_root, 'data', 'embeddings', f'{dataset_name}_metadata.json')
    
    if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
        logger.warning(f"Embeddings not found for {dataset_name}")
        return None, None
    
    try:
        embeddings = np.load(embeddings_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded embeddings for {dataset_name}: {embeddings.shape}")
        return embeddings, metadata
    except Exception as e:
        logger.error(f"Failed to load embeddings for {dataset_name}: {e}")
        return None, None


def load_synergy_dataset(dataset_name: str = None) -> Optional[Dict]:
    """
    Load synergy dataset for citation information.
    
    Args:
        dataset_name: Name of the dataset to load synergy data for
        
    Returns:
        Dictionary with synergy dataset data or None if not available
    """
    try:
        # Try to import synergy_dataset
        import sys
        project_root = get_project_root()
        models_dir = os.path.join(project_root, 'models')
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        
        from synergy_dataset import Dataset
    except ImportError:
        logger.warning("synergy_dataset not available - citation edges will not be added")
        return None
        
    try:
        if not dataset_name:
            logger.warning("No dataset name provided for synergy data loading")
            return None
            
        # Load datasets configuration to get synergy dataset name
        datasets_config = load_datasets_config()
        
        if dataset_name not in datasets_config:
            logger.warning(f"Dataset '{dataset_name}' not found in configuration")
            return None
            
        synergy_name = datasets_config[dataset_name]['synergy_dataset_name']
        logger.info(f"Loading Synergy dataset: {synergy_name}")
        
        # Load the synergy dataset
        dataset = Dataset(synergy_name)
        synergy_data = dataset.to_dict(["id", "title", "referenced_works"])
        
        logger.info(f"Loaded {len(synergy_data)} documents from Synergy dataset")
        return synergy_data
        
    except Exception as e:
        logger.error(f"Error loading synergy dataset: {e}")
        return None


def prompt_dataset_selection() -> str:
    """
    Prompt user to select a dataset from available options.
    
    Returns:
        Selected dataset name
    """
    datasets = get_available_datasets()
    
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")
    
    while True:
        try:
            selection = int(input("\nSelect dataset (enter number): "))
            if 1 <= selection <= len(datasets):
                return datasets[selection-1]
            else:
                print(f"Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("Please enter a valid number")


def create_evaluation_split(simulation_df: pd.DataFrame, 
                          test_size: float = 0.2,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split for evaluation, ensuring relevant documents are in both splits.
    
    Args:
        simulation_df: DataFrame with simulation data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, test_df)
    """
    np.random.seed(random_state)
    
    # Separate relevant and irrelevant documents
    relevant_docs = simulation_df[simulation_df['label_included'] == 1].copy()
    irrelevant_docs = simulation_df[simulation_df['label_included'] == 0].copy()
    
    # Calculate split sizes
    n_relevant_test = max(1, int(len(relevant_docs) * test_size))
    n_irrelevant_test = int(len(irrelevant_docs) * test_size)
    
    # Random split
    relevant_test_idx = np.random.choice(len(relevant_docs), size=n_relevant_test, replace=False)
    irrelevant_test_idx = np.random.choice(len(irrelevant_docs), size=n_irrelevant_test, replace=False)
    
    # Create test set
    relevant_test = relevant_docs.iloc[relevant_test_idx]
    irrelevant_test = irrelevant_docs.iloc[irrelevant_test_idx]
    test_df = pd.concat([relevant_test, irrelevant_test]).reset_index(drop=True)
    
    # Create train set (remaining documents)
    relevant_train = relevant_docs.drop(relevant_docs.index[relevant_test_idx])
    irrelevant_train = irrelevant_docs.drop(irrelevant_docs.index[irrelevant_test_idx])
    train_df = pd.concat([relevant_train, irrelevant_train]).reset_index(drop=True)
    
    logger.info(f"Created evaluation split:")
    logger.info(f"  Train: {len(train_df)} documents ({train_df['label_included'].sum()} relevant)")
    logger.info(f"  Test: {len(test_df)} documents ({test_df['label_included'].sum()} relevant)")
    
    return train_df, test_df


def calculate_evaluation_metrics(y_true: List[int], y_scores: List[float], 
                               threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate evaluation metrics for outlier detection.
    
    Args:
        y_true: True labels (1 for relevant, 0 for irrelevant)
        y_scores: Predicted relevance scores (0-1)
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary with evaluation metrics
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = {}
    
    try:
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC-ROC (if both classes present)
        if len(np.unique(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
        else:
            metrics['auc_roc'] = 0.0
        
        # Additional metrics
        metrics['accuracy'] = np.mean(y_true == y_pred)
        metrics['threshold'] = threshold
        
        # Ranking metrics
        metrics['mean_score_relevant'] = np.mean(y_scores[y_true == 1]) if np.sum(y_true) > 0 else 0.0
        metrics['mean_score_irrelevant'] = np.mean(y_scores[y_true == 0]) if np.sum(y_true == 0) > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        # Return default metrics
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'auc_roc': 0.0,
            'accuracy': 0.0,
            'threshold': threshold,
            'mean_score_relevant': 0.0,
            'mean_score_irrelevant': 0.0
        }
    
    return metrics


def find_optimal_threshold(y_true: List[int], y_scores: List[float], 
                         metric: str = 'f1_score') -> Tuple[float, float]:
    """
    Find optimal threshold for binary classification based on specified metric.
    
    Args:
        y_true: True labels
        y_scores: Predicted scores
        metric: Metric to optimize ('f1_score', 'precision', 'recall')
    
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        metrics = calculate_evaluation_metrics(y_true, y_scores, threshold)
        score = metrics.get(metric, 0.0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def evaluate_outlier_ranking(scores: Dict[str, float], 
                           dataset_name: str, 
                           datasets_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Evaluate how well known outliers are ranked by the model.
    
    Args:
        scores: Dictionary mapping document IDs to outlier scores
        dataset_name: Name of the dataset
        datasets_config: Optional pre-loaded datasets configuration
    
    Returns:
        List of dictionaries with ranking analysis for each known outlier
    """
    if datasets_config is None:
        datasets_config = load_datasets_config()
        
    if dataset_name not in datasets_config:
        logger.warning(f"Dataset {dataset_name} not found in configuration")
        return []
    
    # Get known outlier record IDs
    outlier_record_ids = datasets_config[dataset_name].get('outlier_ids', [])
    
    if not outlier_record_ids:
        logger.info(f"No known outliers defined for dataset {dataset_name}")
        return []
    
    # Load simulation data to map record_id to openalex_id
    simulation_df = load_simulation_data(dataset_name)
    record_to_openalex = dict(zip(simulation_df['record_id'], simulation_df['openalex_id']))
    
    # Convert record IDs to OpenAlex IDs
    outlier_openalex_ids = []
    for record_id in outlier_record_ids:
        if record_id in record_to_openalex:
            outlier_openalex_ids.append(record_to_openalex[record_id])
    
    # Sort all documents by score (highest first)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create ranking lookup
    doc_to_rank = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_docs)}
    
    results = []
    for outlier_id in outlier_openalex_ids:
        if outlier_id in scores:
            rank = doc_to_rank[outlier_id]
            score = scores[outlier_id]
            total_docs = len(scores)
            percentile = ((total_docs - rank + 1) / total_docs) * 100
            
            results.append({
                'outlier_id': outlier_id,
                'rank': rank,
                'score': score,
                'total_documents': total_docs,
                'percentile': percentile
            })
    
    return results


def analyze_score_distribution(scores: Dict[str, float], 
                             labels: Dict[str, int]) -> Dict[str, Any]:
    """
    Analyze the distribution of scores for relevant vs irrelevant documents.
    
    Args:
        scores: Dictionary mapping document IDs to scores
        labels: Dictionary mapping document IDs to labels
    
    Returns:
        Dictionary with distribution analysis
    """
    relevant_scores = []
    irrelevant_scores = []
    
    for doc_id, score in scores.items():
        if doc_id in labels:
            if labels[doc_id] == 1:
                relevant_scores.append(score)
            else:
                irrelevant_scores.append(score)
    
    analysis = {
        'total_documents': len(scores),
        'relevant_documents': len(relevant_scores),
        'irrelevant_documents': len(irrelevant_scores),
    }
    
    if relevant_scores:
        analysis['relevant_stats'] = {
            'mean': np.mean(relevant_scores),
            'std': np.std(relevant_scores),
            'min': np.min(relevant_scores),
            'max': np.max(relevant_scores),
            'median': np.median(relevant_scores),
            'p75': np.percentile(relevant_scores, 75),
            'p90': np.percentile(relevant_scores, 90)
        }
    
    if irrelevant_scores:
        analysis['irrelevant_stats'] = {
            'mean': np.mean(irrelevant_scores),
            'std': np.std(irrelevant_scores),
            'min': np.min(irrelevant_scores),
            'max': np.max(irrelevant_scores),
            'median': np.median(irrelevant_scores),
            'p75': np.percentile(irrelevant_scores, 75),
            'p90': np.percentile(irrelevant_scores, 90)
        }
    
    # Calculate separation metrics
    if relevant_scores and irrelevant_scores:
        analysis['separation'] = {
            'mean_difference': np.mean(relevant_scores) - np.mean(irrelevant_scores),
            'overlap_ratio': len([s for s in irrelevant_scores 
                                if s > np.mean(relevant_scores)]) / len(irrelevant_scores)
        }
    
    return analysis


def calculate_outlier_detection_metrics(y_true: np.ndarray, 
                                      outlier_scores: np.ndarray,
                                      contamination: float = 0.1) -> Dict[str, float]:
    """
    Calculate metrics specifically for outlier detection tasks.
    
    Args:
        y_true: True labels (1 for outliers, 0 for normal)
        outlier_scores: Outlier scores (higher = more anomalous)
        contamination: Expected fraction of outliers
        
    Returns:
        Dictionary with outlier detection metrics
    """
    # Convert to binary predictions using percentile threshold
    threshold = np.percentile(outlier_scores, 100 * (1 - contamination))
    y_pred = (outlier_scores >= threshold).astype(int)
    
    metrics = {}
    
    try:
        # Standard classification metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Ranking-based metrics
        if len(np.unique(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, outlier_scores)
        else:
            metrics['auc_roc'] = 0.0
        
        # Outlier-specific metrics
        true_outliers = np.sum(y_true == 1)
        detected_outliers = np.sum(y_pred == 1)
        
        metrics.update({
            'true_outliers': int(true_outliers),
            'detected_outliers': int(detected_outliers),
            'contamination_used': contamination,
            'threshold_used': float(threshold),
            'mean_outlier_score': float(np.mean(outlier_scores[y_true == 1])) if true_outliers > 0 else 0.0,
            'mean_normal_score': float(np.mean(outlier_scores[y_true == 0])) if np.sum(y_true == 0) > 0 else 0.0
        })
        
    except Exception as e:
        logger.error(f"Error calculating outlier detection metrics: {e}")
        # Return default metrics
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'auc_roc': 0.0,
            'true_outliers': int(np.sum(y_true == 1)),
            'detected_outliers': 0,
            'contamination_used': contamination,
            'threshold_used': 0.0,
            'mean_outlier_score': 0.0,
            'mean_normal_score': 0.0
        }
    
    return metrics


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save results to a JSON file.
    
    Args:
        results: Dictionary with results to save
        output_path: Path where to save the results
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_converted = convert_numpy(results)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def load_results(input_path: str) -> Dict[str, Any]:
    """
    Load results from a JSON file.
    
    Args:
        input_path: Path to the results file
    
    Returns:
        Dictionary with loaded results
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    logger.info(f"Results loaded from {input_path}")
    return results


def print_evaluation_summary(metrics: Dict[str, float]):
    """
    Print a formatted summary of evaluation metrics.
    
    Args:
        metrics: Dictionary with evaluation metrics
    """
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Threshold: {metrics.get('threshold', 0.5):.3f}")
    print(f"Precision: {metrics.get('precision', 0.0):.3f}")
    print(f"Recall: {metrics.get('recall', 0.0):.3f}")
    print(f"F1-Score: {metrics.get('f1_score', 0.0):.3f}")
    print(f"AUC-ROC: {metrics.get('auc_roc', 0.0):.3f}")
    print(f"Accuracy: {metrics.get('accuracy', 0.0):.3f}")
    print(f"Mean Score (Relevant): {metrics.get('mean_score_relevant', 0.0):.3f}")
    print(f"Mean Score (Irrelevant): {metrics.get('mean_score_irrelevant', 0.0):.3f}")
    print("="*50)


def print_outlier_evaluation_summary(metrics: Dict[str, float]):
    """
    Print a formatted summary of outlier detection metrics.
    
    Args:
        metrics: Dictionary with outlier detection metrics
    """
    print("\n" + "="*60)
    print("OUTLIER DETECTION EVALUATION SUMMARY")
    print("="*60)
    print(f"Contamination Used: {metrics.get('contamination_used', 0.1):.3f}")
    print(f"Threshold Used: {metrics.get('threshold_used', 0.0):.4f}")
    print(f"True Outliers: {metrics.get('true_outliers', 0)}")
    print(f"Detected Outliers: {metrics.get('detected_outliers', 0)}")
    print(f"Precision: {metrics.get('precision', 0.0):.3f}")
    print(f"Recall: {metrics.get('recall', 0.0):.3f}")
    print(f"F1-Score: {metrics.get('f1_score', 0.0):.3f}")
    print(f"AUC-ROC: {metrics.get('auc_roc', 0.0):.3f}")
    print(f"Mean Outlier Score: {metrics.get('mean_outlier_score', 0.0):.3f}")
    print(f"Mean Normal Score: {metrics.get('mean_normal_score', 0.0):.3f}")
    print("="*60)


def create_sample_documents(simulation_df: pd.DataFrame, 
                          n_relevant: int = 5, 
                          n_irrelevant: int = 15) -> List[str]:
    """
    Create a sample of documents for testing.
    
    Args:
        simulation_df: DataFrame with simulation data
        n_relevant: Number of relevant documents to include
        n_irrelevant: Number of irrelevant documents to include
    
    Returns:
        List of document IDs
    """
    relevant_docs = simulation_df[simulation_df['label_included'] == 1]['openalex_id'].tolist()
    irrelevant_docs = simulation_df[simulation_df['label_included'] == 0]['openalex_id'].tolist()
    
    # Sample documents
    sample_relevant = relevant_docs[:n_relevant] if len(relevant_docs) >= n_relevant else relevant_docs
    sample_irrelevant = irrelevant_docs[:n_irrelevant] if len(irrelevant_docs) >= n_irrelevant else irrelevant_docs
    
    sample_docs = sample_relevant + sample_irrelevant
    
    logger.info(f"Created sample with {len(sample_relevant)} relevant and {len(sample_irrelevant)} irrelevant documents")
    
    return sample_docs


def check_data_availability(dataset_name: str) -> Dict[str, bool]:
    """
    Check what data is available for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset to check
    
    Returns:
        Dictionary indicating availability of different data types
    """
    project_root = get_project_root()
    
    availability = {
        'simulation_data': False,
        'embeddings': False,
        'embeddings_metadata': False
    }
    
    # Check simulation data
    simulation_path = os.path.join(project_root, 'data', 'simulations', f'{dataset_name}.csv')
    availability['simulation_data'] = os.path.exists(simulation_path)
    
    # Check embeddings
    embeddings_path = os.path.join(project_root, 'data', 'embeddings', f'{dataset_name}.npy')
    metadata_path = os.path.join(project_root, 'data', 'embeddings', f'{dataset_name}_metadata.json')
    availability['embeddings'] = os.path.exists(embeddings_path)
    availability['embeddings_metadata'] = os.path.exists(metadata_path)
    
    return availability


def validate_dataset(dataset_name: str) -> bool:
    """
    Validate that a dataset has the minimum required data.
    
    Args:
        dataset_name: Name of the dataset to validate
    
    Returns:
        True if dataset is valid, False otherwise
    """
    try:
        # Check if dataset exists in configuration
        config = load_datasets_config()
        if dataset_name not in config:
            logger.error(f"Dataset {dataset_name} not found in configuration")
            return False
        
        # Check data availability
        availability = check_data_availability(dataset_name)
        
        if not availability['simulation_data']:
            logger.error(f"Simulation data not found for {dataset_name}")
            return False
        
        # Try to load simulation data
        simulation_df = load_simulation_data(dataset_name)
        
        # Check required columns
        required_columns = ['openalex_id', 'label_included']
        missing_columns = [col for col in required_columns if col not in simulation_df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in {dataset_name}: {missing_columns}")
            return False
        
        # Check for relevant documents
        n_relevant = simulation_df['label_included'].sum()
        if n_relevant == 0:
            logger.error(f"No relevant documents found in {dataset_name}")
            return False
        
        logger.info(f"Dataset {dataset_name} validated successfully")
        logger.info(f"  Documents: {len(simulation_df)}")
        logger.info(f"  Relevant: {n_relevant}")
        logger.info(f"  Data availability: {availability}")
        
        return True
        
    except Exception as e:
        logger.error(f"Validation failed for {dataset_name}: {e}")
        return False


def normalize_scores(scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize scores using specified method.
    
    Args:
        scores: Array of scores to normalize
        method: Normalization method ('minmax', 'zscore', 'rank', 'percentile', 'quantile')
    
    Returns:
        Normalized scores
    """
    scores = np.array(scores)
    
    if method == 'minmax':
        if scores.max() - scores.min() > 1e-8:
            return (scores - scores.min()) / (scores.max() - scores.min())
        else:
            return np.zeros_like(scores)
    
    elif method == 'zscore':
        if scores.std() > 1e-8:
            return (scores - scores.mean()) / scores.std()
        else:
            return np.zeros_like(scores)
    
    elif method == 'rank':
        from scipy.stats import rankdata
        return rankdata(scores) / len(scores)
    
    elif method == 'percentile':
        # Convert scores to percentile ranks (0-100 scale)
        from scipy.stats import rankdata
        percentiles = rankdata(scores, method='average') / len(scores) * 100
        return percentiles / 100  # Normalize to 0-1 range
    
    elif method == 'quantile':
        # Quantile transformation to uniform distribution [0,1]
        try:
            from sklearn.preprocessing import QuantileTransformer
            qt = QuantileTransformer(output_distribution='uniform', random_state=42)
            return qt.fit_transform(scores.reshape(-1, 1)).flatten()
        except ImportError:
            # Fallback to percentile method if sklearn not available
            print("Warning: sklearn not available for quantile transformation, using percentile method")
            from scipy.stats import rankdata
            percentiles = rankdata(scores, method='average') / len(scores)
            return percentiles
    
    else:
        raise ValueError(f"Unknown normalization method: {method}. Available methods: 'minmax', 'zscore', 'rank', 'percentile', 'quantile'")


def compute_ensemble_weights(score_arrays: Dict[str, np.ndarray], 
                           method: str = 'variance') -> Dict[str, float]:
    """
    Compute weights for ensemble methods based on score characteristics.
    
    Args:
        score_arrays: Dictionary mapping method names to score arrays
        method: Weighting method ('variance', 'uniform', 'custom')
    
    Returns:
        Dictionary with weights for each method
    """
    if method == 'uniform':
        n_methods = len(score_arrays)
        return {method_name: 1.0 / n_methods for method_name in score_arrays.keys()}
    
    elif method == 'variance':
        # Higher variance indicates better discrimination
        variances = {method: np.var(scores) for method, scores in score_arrays.items()}
        total_variance = sum(variances.values())
        
        if total_variance > 0:
            return {method: var / total_variance for method, var in variances.items()}
        else:
            # Fallback to uniform if no variance
            n_methods = len(score_arrays)
            return {method: 1.0 / n_methods for method in score_arrays.keys()}
    
    elif method == 'custom':
        # Custom weights for citation network methods
        default_weights = {
            'lof_embedding': 0.40,
            'lof_mixed': 0.30,
            'isolation_forest': 0.15,
            'one_class_svm': 0.10,
            'dbscan': 0.05,
        }
        
        # Only return weights for methods that exist in score_arrays
        weights = {}
        for method_name in score_arrays.keys():
            weights[method_name] = default_weights.get(method_name, 1.0 / len(score_arrays))
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {method: w / total_weight for method, w in weights.items()}
        
        return weights
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")


def get_outlier_ids_for_dataset(dataset_name: str) -> List[str]:
    """
    Get known outlier IDs for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        List of known outlier record IDs
    """
    try:
        datasets_config = load_datasets_config()
        if dataset_name in datasets_config:
            return datasets_config[dataset_name].get('outlier_ids', [])
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting outlier IDs for {dataset_name}: {e}")
        return []


def convert_record_ids_to_openalex(record_ids: List[str], 
                                 simulation_df: pd.DataFrame) -> List[str]:
    """
    Convert record IDs to OpenAlex IDs using simulation data.
    
    Args:
        record_ids: List of record IDs to convert
        simulation_df: DataFrame with simulation data containing the mapping
    
    Returns:
        List of corresponding OpenAlex IDs
    """
    if 'record_id' not in simulation_df.columns or 'openalex_id' not in simulation_df.columns:
        logger.warning("Required columns not found for ID conversion")
        return []
    
    record_to_openalex = dict(zip(simulation_df['record_id'], simulation_df['openalex_id']))
    
    openalex_ids = []
    for record_id in record_ids:
        if record_id in record_to_openalex:
            openalex_ids.append(record_to_openalex[record_id])
        else:
            logger.warning(f"Record ID {record_id} not found in simulation data")
    
    return openalex_ids 