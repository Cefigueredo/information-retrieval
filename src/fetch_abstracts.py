"""
Unified script for fetching abstracts from multiple sources.
Replaces main_request.py and alternative_request.py with a unified approach.
"""
import argparse
from pathlib import Path

from config import (
    PUBLICATIONS_FILE,
    RELEVANT_ABSTRACTS_FILE,
    ALTERNATIVE_ABSTRACTS_FILE,
    FAILED_DOCUMENTS_FILE,
    ALTERNATIVE_FAILED_FILE
)
from utils import (
    load_publications,
    load_existing_abstracts,
    save_abstracts,
    save_failed_documents,
    get_missing_publications,
    ProgressTracker
)
from abstract_retriever import AbstractRetriever


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Fetch abstracts from multiple academic databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch all missing abstracts using all sources
  python fetch_abstracts.py
  
  # Only use fast APIs (Phase 1)
  python fetch_abstracts.py --phases 1
  
  # Use Phase 1 and 2 only (skip Semantic Scholar)
  python fetch_abstracts.py --phases 1 2
  
  # Process specific input/output files
  python fetch_abstracts.py --input my_pubs.xlsx --output my_abstracts.json
  
  # Start fresh (ignore existing abstracts)
  python fetch_abstracts.py --fresh
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        default=str(PUBLICATIONS_FILE),
        help='Input Excel file with publications'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=str(ALTERNATIVE_ABSTRACTS_FILE),
        help='Output JSON file for abstracts'
    )
    
    parser.add_argument(
        '--failed', '-f',
        default=str(ALTERNATIVE_FAILED_FILE),
        help='Output JSON file for failed documents'
    )
    
    parser.add_argument(
        '--existing',
        default=str(RELEVANT_ABSTRACTS_FILE),
        help='Existing abstracts file to check against'
    )
    
    parser.add_argument(
        '--phases', '-p',
        nargs='+',
        type=int,
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help='Which phases to use (1=fast, 2=additional, 3=rate-limited)'
    )
    
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh, ignore existing abstracts'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("UNIFIED ABSTRACT RETRIEVAL")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Failed: {args.failed}")
    print(f"  Phases: {args.phases}")
    
    # Load data
    print("\nLoading data...")
    publications_df = load_publications(args.input)
    
    if args.fresh:
        print("Starting fresh (ignoring existing abstracts)")
        missing_df = publications_df
    else:
        existing_abstracts = load_existing_abstracts(args.existing)
        missing_df = get_missing_publications(publications_df, existing_abstracts)
    
    if len(missing_df) == 0:
        print("\n✓ All publications already have abstracts!")
        return
    
    # Initialize retriever
    retriever = AbstractRetriever()
    
    # Retrieve abstracts
    print(f"\nRetrieving abstracts for {len(missing_df)} publications...")
    results, failed = retriever.retrieve_abstracts(missing_df, use_phases=args.phases)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total processed: {len(missing_df)}")
    print(f"Successfully retrieved: {len(results)}")
    print(f"Failed: {len(failed)}")
    
    if results:
        success_rate = len(results) / len(missing_df) * 100
        print(f"Success rate: {success_rate:.2f}%")
        
        # Show source statistics
        sources = {}
        for data in results.values():
            source = data.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        print("\nAbstracts by source:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count}")
    
    # Save results
    save_abstracts(results, args.output)
    save_failed_documents(failed, len(missing_df), args.failed)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

