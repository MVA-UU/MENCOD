#!/usr/bin/env python3
"""
Test Script for Hybrid Outlier Detection Model

This script tests all individual models and the hybrid model to ensure
everything is working correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data() -> pd.DataFrame:
    """Create synthetic test data for testing purposes."""
    np.random.seed(42)
    
    # Create test documents
    test_data = []
    for i in range(100):
        doc_id = f"W{1000000 + i}"
        label = 1 if i < 20 else 0  # 20% relevant documents
        title = f"Test Document {i}: {'Relevant' if label else 'Irrelevant'} Research"
        abstract = f"This is a test abstract for document {i}. " * 10  # Longer abstract
        
        test_data.append({
            'openalex_id': doc_id,
            'label_included': label,
            'title': title,
            'abstract': abstract,
            'year': 2020 + (i % 5),
        })
    
    return pd.DataFrame(test_data)


def test_citation_network_model():
    """Test the citation network model."""
    logger.info("Testing Citation Network Model...")
    
    try:
        from models.CitationNetwork import CitationNetworkModel
        
        # Create test data
        test_df = create_test_data()
        
        # Initialize model
        model = CitationNetworkModel(enable_gpu=False, enable_semantic=False)
        
        # Fit model
        model.fit(test_df, "test_dataset")
        
        # Test feature extraction
        test_docs = test_df['openalex_id'].head(10).tolist()
        features = model.extract_features(test_docs)
        
        # Test score prediction
        scores = model.predict_relevance_scores(test_docs)
        
        logger.info(f"Citation Network Model: ✓ Features shape: {features.shape}, Scores: {len(scores)}")
        return True
        
    except Exception as e:
        logger.error(f"Citation Network Model failed: {e}")
        return False


def test_confidence_calibration_model():
    """Test the confidence calibration model."""
    logger.info("Testing Confidence Calibration Model...")
    
    try:
        from models.ConfidenceCalibration import ConfidenceCalibrationModel
        
        # Create test data
        test_df = create_test_data()
        
        # Initialize model
        model = ConfidenceCalibrationModel(n_estimators=3)  # Fewer estimators for testing
        
        # Fit model
        model.fit(test_df, "test_dataset")
        
        # Test feature extraction
        test_docs = test_df['openalex_id'].head(10).tolist()
        features = model.extract_features(test_docs)
        
        # Test score prediction
        scores = model.predict_relevance_scores(test_docs)
        
        logger.info(f"Confidence Calibration Model: ✓ Features shape: {features.shape}, Scores: {len(scores)}")
        return True
        
    except Exception as e:
        logger.error(f"Confidence Calibration Model failed: {e}")
        return False


def test_content_similarity_model():
    """Test the content similarity model."""
    logger.info("Testing Content Similarity Model...")
    
    try:
        from models.ContentSimilarity import ContentSimilarityModel
        
        # Create test data
        test_df = create_test_data()
        
        # Initialize model
        model = ContentSimilarityModel(enable_semantic_embeddings=False)
        
        # Fit model
        model.fit(test_df, "test_dataset")
        
        # Test feature extraction
        test_docs = test_df['openalex_id'].head(10).tolist()
        features = model.extract_features(test_docs)
        
        # Test score prediction
        scores = model.predict_relevance_scores(test_docs)
        
        logger.info(f"Content Similarity Model: ✓ Features shape: {features.shape}, Scores: {len(scores)}")
        return True
        
    except Exception as e:
        logger.error(f"Content Similarity Model failed: {e}")
        return False


def test_hybrid_model():
    """Test the hybrid model."""
    logger.info("Testing Hybrid Model...")
    
    try:
        from hybrid_model import HybridOutlierDetector, ModelConfiguration
        
        # Create test data
        test_df = create_test_data()
        
        # Test configuration with all models disabled except the ones that work
        config = ModelConfiguration(
            enable_citation_network=True,
            enable_confidence_calibration=True,
            enable_content_similarity=True,
            enable_gpu_acceleration=False,  # Disable GPU for testing
            enable_semantic_embeddings=False  # Disable semantic for testing
        )
        
        # Initialize hybrid model
        hybrid_model = HybridOutlierDetector(
            dataset_name=None,
            model_config=config,
            use_adaptive_weights=True
        )
        
        # Fit model
        hybrid_model.fit(test_df, "test_dataset")
        
        # Test feature extraction
        test_docs = test_df['openalex_id'].head(10).tolist()
        features = hybrid_model.extract_features(test_docs)
        
        # Test score prediction
        scores = hybrid_model.predict_relevance_scores(test_docs)
        
        # Test outlier prediction
        outliers = hybrid_model.predict_outliers(test_docs)
        
        # Test model status
        status = hybrid_model.get_model_status()
        
        logger.info(f"Hybrid Model: ✓ Features shape: {features.shape}, Scores: {len(scores)}, Outliers: {len(outliers)}")
        logger.info(f"Model weights: {status['model_weights']}")
        return True
        
    except Exception as e:
        logger.error(f"Hybrid Model failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions."""
    logger.info("Testing Utility Functions...")
    
    try:
        from utils import (
            create_evaluation_split, calculate_evaluation_metrics,
            analyze_score_distribution, create_sample_documents
        )
        
        # Create test data
        test_df = create_test_data()
        
        # Test evaluation split
        train_df, test_df_split = create_evaluation_split(test_df, test_size=0.3)
        
        # Test metrics calculation
        y_true = [1, 0, 1, 0, 1]
        y_scores = [0.8, 0.3, 0.7, 0.4, 0.9]
        metrics = calculate_evaluation_metrics(y_true, y_scores, threshold=0.5)
        
        # Test score distribution analysis
        scores = {f"doc_{i}": np.random.random() for i in range(20)}
        labels = {f"doc_{i}": i % 2 for i in range(20)}
        distribution = analyze_score_distribution(scores, labels)
        
        # Test sample creation
        sample_docs = create_sample_documents(test_df, n_relevant=5, n_irrelevant=10)
        
        logger.info(f"Utils: ✓ Split: {len(train_df)}/{len(test_df_split)}, Metrics: {len(metrics)}, Sample: {len(sample_docs)}")
        return True
        
    except Exception as e:
        logger.error(f"Utils test failed: {e}")
        return False


def test_individual_model_configurations():
    """Test different model configurations."""
    logger.info("Testing Individual Model Configurations...")
    
    try:
        from hybrid_model import HybridOutlierDetector, ModelConfiguration
        
        test_df = create_test_data()
        test_docs = test_df['openalex_id'].head(5).tolist()
        
        configs_to_test = [
            ("Citation Only", ModelConfiguration(
                enable_citation_network=True,
                enable_confidence_calibration=False,
                enable_content_similarity=False,
                enable_gpu_acceleration=False,
                enable_semantic_embeddings=False
            )),
            ("Confidence Only", ModelConfiguration(
                enable_citation_network=False,
                enable_confidence_calibration=True,
                enable_content_similarity=False,
                enable_gpu_acceleration=False,
                enable_semantic_embeddings=False
            )),
            ("Content Only", ModelConfiguration(
                enable_citation_network=False,
                enable_confidence_calibration=False,
                enable_content_similarity=True,
                enable_gpu_acceleration=False,
                enable_semantic_embeddings=False
            )),
        ]
        
        for config_name, config in configs_to_test:
            logger.info(f"Testing {config_name} configuration...")
            
            model = HybridOutlierDetector(model_config=config, use_adaptive_weights=False)
            model.fit(test_df, "test_dataset")
            scores = model.predict_relevance_scores(test_docs)
            
            logger.info(f"{config_name}: ✓ Scores computed: {len(scores)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Individual model configuration test failed: {e}")
        return False


def run_all_tests() -> Dict[str, bool]:
    """Run all tests and return results."""
    logger.info("="*60)
    logger.info("RUNNING ALL TESTS")
    logger.info("="*60)
    
    test_results = {}
    
    # Test individual models
    test_results['citation_network'] = test_citation_network_model()
    test_results['confidence_calibration'] = test_confidence_calibration_model()
    test_results['content_similarity'] = test_content_similarity_model()
    
    # Test hybrid model
    test_results['hybrid_model'] = test_hybrid_model()
    
    # Test utilities
    test_results['utils'] = test_utils()
    
    # Test configurations
    test_results['configurations'] = test_individual_model_configurations()
    
    return test_results


def main():
    """Main test function."""
    try:
        # Run all tests
        results = run_all_tests()
        
        # Print summary
        logger.info("="*60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*60)
        
        passed = 0
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✓ PASSED" if passed_test else "✗ FAILED"
            logger.info(f"{test_name:20s}: {status}")
            if passed_test:
                passed += 1
        
        logger.info("="*60)
        logger.info(f"OVERALL: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! The hybrid model is ready to use.")
            return 0
        else:
            logger.error(f"❌ {total-passed} tests failed. Please check the errors above.")
            return 1
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 