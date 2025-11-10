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
  python run.py                   # Unified approach (all sources, recommended)
  python run.py --phases 1        # Only fast APIs
  python run.py --phases 1 2      # Fast + additional (skip Semantic Scholar)
  python run.py --fresh           # Ignore existing abstracts
  
Legacy modes (still supported):
  python run.py main              # Old main_request.py (PubMed only)
  python run.py alternative       # Old alternative_request.py (multi-API)
        """
    )
    
    parser.add_argument(
        'mode',
        nargs='?',
        choices=['unified', 'main', 'alternative', 'alt'],
        default='unified',
        help='Which mode to run (default: unified)'
    )
    
    parser.add_argument(
        '--phases', '-p',
        nargs='+',
        type=int,
        choices=[1, 2, 3],
        help='Which phases to use in unified mode'
    )
    
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh, ignore existing abstracts'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'unified' or args.mode is None:
        print("Running unified abstract fetcher...")
        print("=" * 80)
        from fetch_abstracts import main as fetch_main
        
        # Pass arguments to fetch_abstracts
        sys.argv = ['fetch_abstracts.py']
        if args.phases:
            sys.argv.extend(['--phases'] + [str(p) for p in args.phases])
        if args.fresh:
            sys.argv.append('--fresh')
        
        fetch_main()
        
    elif args.mode == 'main':
        print("Running main_request.py (NCBI PubMed only - legacy)...")
        print("=" * 80)
        from main_request import fetch_relevant_abstracts
        fetch_relevant_abstracts()
        
    elif args.mode in ['alternative', 'alt']:
        print("Running alternative_request.py (Multi-API - legacy)...")
        print("=" * 80)
        from alternative_request import main as alt_main
        alt_main()
        
    from fetch_non_relevant_abstrats import fetch_non_relevant_abstracts
    fetch_non_relevant_abstracts()
    
    print("\n" + "=" * 80)
    print("Done!")

if __name__ == "__main__":
    main()

