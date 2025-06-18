"""
Confidence Calibration Model for Hybrid Outlier Detection

This module provides confidence calibration features for identifying outlier documents
where models are overconfident in their predictions but potentially wrong.
"""

from .confidence_calibration import ConfidenceCalibrationModel

__all__ = ['ConfidenceCalibrationModel'] 