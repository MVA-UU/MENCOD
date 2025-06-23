"""
Test script to verify MENCOD evaluation setup and dependencies.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError:
        print("✗ pandas - please install: pip install pandas")
        return False
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError:
        print("✗ numpy - please install: pip install numpy")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError:
        print("✗ matplotlib - please install: pip install matplotlib")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn")
    except ImportError:
        print("✗ seaborn - please install: pip install seaborn")
        return False
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        print("✓ scikit-learn")
    except ImportError:
        print("✗ scikit-learn - please install: pip install scikit-learn")
        return False
    
    return True

def test_mencod_imports():
    """Test MENCOD-specific imports."""
    print("\nTesting MENCOD imports...")
    
    try:
        from MENCOD import CitationNetworkOutlierDetector
        print("✓ MENCOD.CitationNetworkOutlierDetector")
    except ImportError as e:
        print(f"✗ MENCOD.CitationNetworkOutlierDetector - {e}")
        return False
    
    try:
        from utils import get_available_datasets, load_simulation_data, load_datasets_config
        print("✓ utils functions")
    except ImportError as e:
        print(f"✗ utils functions - {e}")
        return False
    
    return True

def test_data_availability():
    """Test if datasets are available."""
    print("\nTesting data availability...")
    
    try:
        from utils import get_available_datasets, load_datasets_config
        
        datasets = get_available_datasets()
        print(f"✓ Available datasets: {datasets}")
        
        config = load_datasets_config()
        print(f"✓ Datasets config loaded with {len(config)} entries")
        
        # Test loading one dataset
        if datasets:
            test_dataset = datasets[0]
            from utils import load_simulation_data
            df = load_simulation_data(test_dataset)
            print(f"✓ Successfully loaded {test_dataset} with {len(df)} documents")
        
        return True
        
    except Exception as e:
        print(f"✗ Data availability test failed: {e}")
        return False

def test_evaluation_scripts():
    """Test if evaluation scripts can be imported."""
    print("\nTesting evaluation script imports...")
    
    try:
        from dataset_visualizer import DatasetVisualizer
        print("✓ DatasetVisualizer")
    except ImportError as e:
        print(f"✗ DatasetVisualizer - {e}")
        return False
    
    try:
        from multi_dataset_analyzer import MultiDatasetAnalyzer
        print("✓ MultiDatasetAnalyzer")
    except ImportError as e:
        print(f"✗ MultiDatasetAnalyzer - {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("MENCOD Evaluation Setup Test")
    print("=" * 40)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test MENCOD imports
    if not test_mencod_imports():
        all_passed = False
    
    # Test data availability
    if not test_data_availability():
        all_passed = False
    
    # Test evaluation scripts
    if not test_evaluation_scripts():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Ready to run evaluations.")
        print("\nTo run individual dataset analysis:")
        print("  python Evaluation/dataset_visualizer.py")
        print("\nTo run multi-dataset analysis:")
        print("  python Evaluation/multi_dataset_analyzer.py")
        print("\nTo run complete evaluation suite:")
        print("  python Evaluation/run_complete_evaluation.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main() 