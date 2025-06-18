#!/usr/bin/env python3
"""
Run Hybrid Outlier Detection Model

This script provides a comprehensive interface for running the hybrid outlier detection
model with different configurations and evaluation options.
"""

import argparse
import sys
import os
import time
import json
from typing import Optional, Dict, Any

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_model import HybridOutlierDetector, ModelConfiguration, ModelWeights
from utils import (
    get_available_datasets, load_simulation_data, prompt_dataset_selection,
    create_evaluation_split, calculate_evaluation_metrics, find_optimal_threshold,
    analyze_score_distribution, save_results, print_evaluation_summary,
    create_sample_documents, validate_dataset, check_data_availability
)


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run Hybrid Outlier Detection Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_hybrid_model.py --dataset appenzeller --mode demo
  python run_hybrid_model.py --dataset hall --mode evaluate --output results.json
  python run_hybrid_model.py --mode interactive
  python run_hybrid_model.py --dataset valk --disable-citation --disable-confidence
        """
    )
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, help='Dataset name to use')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets and exit')
    
    # Execution mode
    parser.add_argument('--mode', type=str, choices=['demo', 'evaluate', 'interactive', 'analysis'],
                       default='demo', help='Execution mode (default: demo)')
    
    # Model configuration
    parser.add_argument('--disable-citation', action='store_true', help='Disable citation network model')
    parser.add_argument('--disable-confidence', action='store_true', help='Disable confidence calibration model')
    parser.add_argument('--disable-content', action='store_true', help='Disable content similarity model')
    parser.add_argument('--disable-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--disable-semantic', action='store_true', help='Disable semantic embeddings')
    parser.add_argument('--disable-adaptive', action='store_true', help='Disable adaptive weighting')
    
    # Model weights (manual override)
    parser.add_argument('--citation-weight', type=float, help='Manual weight for citation network')
    parser.add_argument('--confidence-weight', type=float, help='Manual weight for confidence calibration')
    parser.add_argument('--content-weight', type=float, help='Manual weight for content similarity')
    
    # Evaluation options
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size for evaluation (default: 0.2)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--threshold', type=float, help='Fixed threshold for outlier detection')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--quiet', action='store_true', help='Minimize output')
    
    # Sample size for demo/analysis
    parser.add_argument('--sample-size', type=int, default=50, help='Sample size for demo/analysis (default: 50)')
    
    return parser


def setup_logging(verbose: bool, quiet: bool):
    """Setup logging configuration."""
    import logging
    
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_model_config(args) -> ModelConfiguration:
    """Create model configuration from command line arguments."""
    return ModelConfiguration(
        enable_citation_network=not args.disable_citation,
        enable_confidence_calibration=not args.disable_confidence,
        enable_content_similarity=not args.disable_content,
        enable_gpu_acceleration=not args.disable_gpu,
        enable_semantic_embeddings=not args.disable_semantic
    )


def create_model_weights(args) -> Optional[ModelWeights]:
    """Create model weights from command line arguments."""
    if any([args.citation_weight, args.confidence_weight, args.content_weight]):
        citation_weight = args.citation_weight or 0.0
        confidence_weight = args.confidence_weight or 0.0
        content_weight = args.content_weight or 0.0
        
        # Normalize weights to sum to 1
        total_weight = citation_weight + confidence_weight + content_weight
        if total_weight > 0:
            return ModelWeights(
                citation_network=citation_weight / total_weight,
                confidence_calibration=confidence_weight / total_weight,
                content_similarity=content_weight / total_weight
            )
    
    return None


def run_demo_mode(dataset_name: str, args) -> Dict[str, Any]:
    """Run demo mode - basic functionality demonstration."""
    print("="*60)
    print("HYBRID OUTLIER DETECTION - DEMO MODE")
    print("="*60)
    
    # Create model configuration
    model_config = create_model_config(args)
    model_weights = create_model_weights(args)
    
    # Initialize hybrid model
    hybrid_model = HybridOutlierDetector(
        dataset_name=dataset_name,
        model_config=model_config,
        model_weights=model_weights,
        use_adaptive_weights=not args.disable_adaptive
    )
    
    # Load data
    simulation_df = load_simulation_data(dataset_name)
    
    # Fit model
    print(f"\nFitting hybrid model on {dataset_name}...")
    start_time = time.time()
    hybrid_model.fit(simulation_df)
    fit_time = time.time() - start_time
    
    # Create sample documents
    sample_docs = create_sample_documents(simulation_df, n_relevant=5, n_irrelevant=args.sample_size-5)
    
    # Extract features
    print(f"\nExtracting features for {len(sample_docs)} sample documents...")
    features_df = hybrid_model.extract_features(sample_docs)
    
    # Compute scores
    print("\nComputing hybrid relevance scores...")
    scores = hybrid_model.predict_relevance_scores(sample_docs)
    
    # Predict outliers
    threshold = args.threshold
    print(f"\nPredicting outliers (threshold: {'dynamic' if threshold is None else threshold})...")
    outliers = hybrid_model.predict_outliers(sample_docs, threshold)
    
    # Show results
    print(f"\nResults Summary:")
    print(f"  Fitting time: {fit_time:.2f} seconds")
    print(f"  Features extracted: {features_df.shape}")
    print(f"  Potential outliers found: {len(outliers)}")
    
    # Show top scoring documents
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 documents by hybrid score:")
    for i, (doc_id, score) in enumerate(top_docs, 1):
        print(f"  {i}. {doc_id}: {score:.4f}")
    
    # Show model status
    status = hybrid_model.get_model_status()
    print(f"\nModel Configuration:")
    for model, enabled in status['model_configuration'].items():
        print(f"  {model}: {'✓' if enabled else '✗'}")
    
    print(f"\nModel Weights:")
    for model, weight in status['model_weights'].items():
        print(f"  {model}: {weight:.3f}")
    
    return {
        'mode': 'demo',
        'dataset': dataset_name,
        'fit_time': fit_time,
        'sample_size': len(sample_docs),
        'features_shape': features_df.shape,
        'num_outliers': len(outliers),
        'top_scores': dict(top_docs),
        'model_status': status
    }


def run_evaluation_mode(dataset_name: str, args) -> Dict[str, Any]:
    """Run evaluation mode - comprehensive performance evaluation."""
    print("="*60)
    print("HYBRID OUTLIER DETECTION - EVALUATION MODE")
    print("="*60)
    
    # Create model configuration
    model_config = create_model_config(args)
    model_weights = create_model_weights(args)
    
    # Load and split data
    simulation_df = load_simulation_data(dataset_name)
    train_df, test_df = create_evaluation_split(simulation_df, args.test_size, args.random_seed)
    
    # Initialize hybrid model
    hybrid_model = HybridOutlierDetector(
        dataset_name=dataset_name,
        model_config=model_config,
        model_weights=model_weights,
        use_adaptive_weights=not args.disable_adaptive
    )
    
    # Fit model on training data
    print(f"\nFitting hybrid model on training data...")
    start_time = time.time()
    hybrid_model.fit(train_df)
    fit_time = time.time() - start_time
    
    # Evaluate on test data
    test_docs = test_df['openalex_id'].tolist()
    test_labels = test_df['label_included'].tolist()
    
    print(f"\nEvaluating on {len(test_docs)} test documents...")
    eval_start = time.time()
    
    # Compute scores
    scores = hybrid_model.predict_relevance_scores(test_docs)
    score_values = [scores[doc_id] for doc_id in test_docs]
    
    eval_time = time.time() - eval_start
    
    # Find optimal threshold
    optimal_threshold, best_f1 = find_optimal_threshold(test_labels, score_values, 'f1_score')
    
    # Calculate metrics with optimal threshold
    metrics_optimal = calculate_evaluation_metrics(test_labels, score_values, optimal_threshold)
    
    # Calculate metrics with fixed threshold if provided
    fixed_threshold = args.threshold or 0.5
    metrics_fixed = calculate_evaluation_metrics(test_labels, score_values, fixed_threshold)
    
    # Analyze score distribution
    label_dict = dict(zip(test_docs, test_labels))
    distribution_analysis = analyze_score_distribution(scores, label_dict)
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"  Fitting time: {fit_time:.2f} seconds")
    print(f"  Evaluation time: {eval_time:.2f} seconds")
    print(f"  Optimal threshold: {optimal_threshold:.3f} (F1: {best_f1:.3f})")
    
    print_evaluation_summary(metrics_optimal)
    
    if args.threshold:
        print(f"\nFixed Threshold Results (threshold: {fixed_threshold:.3f}):")
        print_evaluation_summary(metrics_fixed)
    
    return {
        'mode': 'evaluation',
        'dataset': dataset_name,
        'fit_time': fit_time,
        'eval_time': eval_time,
        'test_size': len(test_docs),
        'optimal_threshold': optimal_threshold,
        'best_f1': best_f1,
        'metrics_optimal': metrics_optimal,
        'metrics_fixed': metrics_fixed,
        'distribution_analysis': distribution_analysis,
        'model_status': hybrid_model.get_model_status()
    }


def run_interactive_mode():
    """Run interactive mode - guided user experience."""
    print("="*60)
    print("HYBRID OUTLIER DETECTION - INTERACTIVE MODE")
    print("="*60)
    
    # Dataset selection
    print("\nStep 1: Dataset Selection")
    dataset_name = prompt_dataset_selection()
    
    # Validate dataset
    if not validate_dataset(dataset_name):
        print(f"Dataset {dataset_name} validation failed. Exiting.")
        return {}
    
    # Show data availability
    availability = check_data_availability(dataset_name)
    print(f"\nData availability for {dataset_name}:")
    for data_type, available in availability.items():
        print(f"  {data_type}: {'✓' if available else '✗'}")
    
    # Model configuration
    print("\nStep 2: Model Configuration")
    print("Select which models to enable:")
    
    enable_citation = input("Enable Citation Network? [Y/n]: ").lower() != 'n'
    enable_confidence = input("Enable Confidence Calibration? [Y/n]: ").lower() != 'n'
    enable_content = input("Enable Content Similarity? [Y/n]: ").lower() != 'n'
    enable_gpu = input("Enable GPU Acceleration? [Y/n]: ").lower() != 'n'
    enable_semantic = input("Enable Semantic Embeddings? [Y/n]: ").lower() != 'n'
    use_adaptive = input("Use Adaptive Weighting? [Y/n]: ").lower() != 'n'
    
    model_config = ModelConfiguration(
        enable_citation_network=enable_citation,
        enable_confidence_calibration=enable_confidence,
        enable_content_similarity=enable_content,
        enable_gpu_acceleration=enable_gpu,
        enable_semantic_embeddings=enable_semantic
    )
    
    # Execution mode
    print("\nStep 3: Execution Mode")
    print("1. Demo (quick demonstration)")
    print("2. Evaluation (comprehensive performance assessment)")
    print("3. Analysis (detailed document analysis)")
    
    mode_choice = input("Select mode [1-3]: ")
    
    if mode_choice == '2':
        test_size = float(input("Test set size [0.2]: ") or "0.2")
        args_obj = type('Args', (), {
            'test_size': test_size,
            'random_seed': 42,
            'threshold': None,
            'sample_size': 50
        })()
        return run_evaluation_mode(dataset_name, args_obj)
    
    elif mode_choice == '3':
        return run_analysis_mode(dataset_name, model_config, use_adaptive)
    
    else:  # Default to demo
        args_obj = type('Args', (), {
            'threshold': None,
            'sample_size': 50,
            'disable_adaptive': not use_adaptive
        })()
        return run_demo_mode(dataset_name, args_obj)


def run_analysis_mode(dataset_name: str, model_config: ModelConfiguration, use_adaptive: bool) -> Dict[str, Any]:
    """Run analysis mode - detailed document analysis."""
    print("="*60)
    print("HYBRID OUTLIER DETECTION - ANALYSIS MODE")
    print("="*60)
    
    # Initialize and fit model
    hybrid_model = HybridOutlierDetector(
        dataset_name=dataset_name,
        model_config=model_config,
        use_adaptive_weights=use_adaptive
    )
    
    simulation_df = load_simulation_data(dataset_name)
    hybrid_model.fit(simulation_df)
    
    # Get sample documents
    sample_docs = create_sample_documents(simulation_df, n_relevant=10, n_irrelevant=40)
    
    # Compute scores and find outliers
    scores = hybrid_model.predict_relevance_scores(sample_docs)
    outliers = hybrid_model.predict_outliers(sample_docs)
    
    print(f"\nAnalysis Results:")
    print(f"  Analyzed documents: {len(sample_docs)}")
    print(f"  Identified outliers: {len(outliers)}")
    
    # Show detailed analysis for top outliers
    if outliers:
        print(f"\nDetailed Analysis of Top Outliers:")
        sorted_outliers = sorted(outliers.items(), key=lambda x: x[1]['relevance_score'], reverse=True)
        
        for i, (doc_id, outlier_info) in enumerate(sorted_outliers[:3], 1):
            print(f"\n{i}. Document ID: {doc_id}")
            print(f"   Relevance Score: {outlier_info['relevance_score']:.4f}")
            print(f"   Threshold: {outlier_info['threshold']:.4f}")
            
            # Get detailed analysis
            analysis = hybrid_model.analyze_document(doc_id)
            print(f"   Hybrid Score: {analysis['hybrid_score']:.4f}")
            
            # Show individual model contributions
            for model_name, model_analysis in analysis['model_analyses'].items():
                if 'relevance_score' in model_analysis:
                    print(f"   {model_name.title()}: {model_analysis['relevance_score']:.4f}")
    
    return {
        'mode': 'analysis',
        'dataset': dataset_name,
        'analyzed_documents': len(sample_docs),
        'identified_outliers': len(outliers),
        'outliers': outliers,
        'model_status': hybrid_model.get_model_status()
    }


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.quiet)
    
    # List datasets and exit if requested
    if args.list_datasets:
        datasets = get_available_datasets()
        print("Available datasets:")
        for i, dataset in enumerate(datasets, 1):
            availability = check_data_availability(dataset)
            status = "✓" if all(availability.values()) else "⚠"
            print(f"  {i}. {dataset} {status}")
        return
    
    try:
        # Determine execution mode and dataset
        if args.mode == 'interactive':
            results = run_interactive_mode()
        else:
            # Get dataset name
            dataset_name = args.dataset
            if not dataset_name:
                dataset_name = prompt_dataset_selection()
            
            # Validate dataset
            if not validate_dataset(dataset_name):
                print(f"Dataset {dataset_name} validation failed. Exiting.")
                return
            
            # Run appropriate mode
            if args.mode == 'evaluate':
                results = run_evaluation_mode(dataset_name, args)
            elif args.mode == 'analysis':
                model_config = create_model_config(args)
                results = run_analysis_mode(dataset_name, model_config, not args.disable_adaptive)
            else:  # demo
                results = run_demo_mode(dataset_name, args)
        
        # Save results if output file specified
        if args.output and results:
            save_results(results, args.output)
            print(f"\nResults saved to: {args.output}")
        
        print("\nExecution completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 