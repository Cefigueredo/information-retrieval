#!/usr/bin/env python3
"""
Simple runner script for the Information Retrieval project.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    parser = argparse.ArgumentParser(
        description="Information Retrieval - Abstract Fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py main              # Fetch from NCBI PubMed
  python run.py alternative       # Fetch from alternative sources
  python run.py --help            # Show this help message
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['main', 'alternative', 'alt'],
        help='Which script to run: "main" for NCBI PubMed, "alternative"/"alt" for multi-API search'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'main':
        print("Running main_request.py (NCBI PubMed)...")
        print("=" * 80)
        from main_request import fetch_relevant_abstracts
        fetch_relevant_abstracts()
        
    elif args.mode in ['alternative', 'alt']:
        print("Running alternative_request.py (Multi-API)...")
        print("=" * 80)
        from alternative_request import main as alt_main
        alt_main()
    
    print("\n" + "=" * 80)
    print("Done!")

if __name__ == "__main__":
    main()

