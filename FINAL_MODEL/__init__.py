"""
Final Hybrid Outlier Detection Model

This package provides a comprehensive hybrid outlier detection system for systematic reviews.
It combines citation network analysis, confidence calibration, and content similarity analysis
with adaptive weighting and continuous scaling for optimal generalizability.

Main Components:
- HybridOutlierDetector: Main hybrid model combining all approaches
- CitationNetworkModel: GPU-accelerated citation network analysis with semantic embeddings
- ConfidenceCalibrationModel: Ensemble-based overconfidence detection
- ContentSimilarityModel: Specialized text pattern analysis

Example Usage:
    from FINAL_MODEL import HybridOutlierDetector
    
    detector = HybridOutlierDetector(dataset_name="appenzeller")
    detector.fit()
    scores = detector.predict_relevance_scores(target_documents)
"""

from .hybrid_model import HybridOutlierDetector, ModelConfiguration, ModelWeights

__version__ = "1.0.0"
__author__ = "ADS Thesis Project"
__email__ = "your.email@example.com"

__all__ = [
    'HybridOutlierDetector',
    'ModelConfiguration', 
    'ModelWeights'
] 