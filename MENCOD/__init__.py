"""
MENCOD - Multi-modal ENsemble Citation Outlier Detector

A streamlined outlier detection system for citation networks using:
- LOF on embeddings for semantic outlier detection
- Isolation Forest for global anomaly detection
- NetworkX-based citation network analysis

Main Components:
- CitationNetworkOutlierDetector: Main detector class
- NetworkBuilder: Citation network construction
- FeatureExtractor: Network and semantic feature extraction
- OutlierDetector: Core outlier detection algorithms
- ResultsAnalyzer: Evaluation and analysis tools
"""

from .core import CitationNetworkOutlierDetector
from .network import NetworkBuilder
from .features import FeatureExtractor
from .detection import OutlierDetector
from .evaluation import ResultsAnalyzer

__version__ = "1.0.0"
__author__ = "M.V.A. van Angeren"

__all__ = [
    'CitationNetworkOutlierDetector',
    'NetworkBuilder',
    'FeatureExtractor', 
    'OutlierDetector',
    'ResultsAnalyzer'
]

# Convenience function for easy usage
def detect_outliers(simulation_df, dataset_name=None, random_state=42):
    """
    Convenience function for outlier detection.
    
    Args:
        simulation_df: DataFrame with paper data
        dataset_name: Optional dataset name for embeddings/synergy data
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with outlier scores and predictions
    """
    detector = CitationNetworkOutlierDetector(random_state=random_state)
    return detector.fit_predict_outliers(simulation_df, dataset_name) 