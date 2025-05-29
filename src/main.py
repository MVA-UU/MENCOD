import argparse
import os
import time
import sys
import traceback

def create_dirs():
    """Create necessary directories"""
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/figures', exist_ok=True)

def run_citation_analysis():
    """Run the citation analysis module"""
    print("Starting citation network analysis...")
    try:
        from citation_analysis import main as citation_main
        citation_main()
    except Exception as e:
        print(f"Error in citation analysis: {e}")
        traceback.print_exc()

def run_text_analysis():
    """Run the text analysis module"""
    print("Starting text analysis...")
    try:
        from text_analysis import main as text_main
        text_main()
    except Exception as e:
        print(f"Error in text analysis: {e}")
        traceback.print_exc()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze ASReview simulation data to detect outliers"
    )
    
    parser.add_argument(
        "--citation", 
        action="store_true",
        help="Run citation network analysis"
    )
    
    parser.add_argument(
        "--text", 
        action="store_true",
        help="Run text analysis"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run all analyses"
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_args()
    
    # Create necessary directories
    create_dirs()
    
    # Determine which analyses to run
    run_all = args.all or (not any([args.citation, args.text]))
    
    # Run selected analyses
    if args.citation or run_all:
        run_citation_analysis()
    
    if args.text or run_all:
        run_text_analysis()
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"\nAnalyses completed in {elapsed_time:.2f} seconds")
    print(f"Results and visualizations saved to the 'output' directory")

if __name__ == "__main__":
    main() 