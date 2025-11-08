"""
Data handling utilities for loading and saving abstracts.
"""
import json
import pandas as pd
from typing import Dict, List
from pathlib import Path


def load_publications(excel_file: str) -> pd.DataFrame:
    """Load publications from Excel file."""
    return pd.read_excel(excel_file)


def load_existing_abstracts(json_file: str) -> Dict:
    """Load existing abstracts from JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {json_file} not found. Starting with empty abstracts.")
        return {}


def save_abstracts(abstracts: Dict, output_file: str, indent: int = 2) -> None:
    """Save abstracts to JSON file."""
    # Ensure directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(abstracts, f, indent=indent)
    
    print(f"\n✓ Saved {len(abstracts)} abstracts to {output_file}")


def save_failed_documents(failed_docs: List[Dict], 
                          total_processed: int,
                          output_file: str,
                          indent: int = 2) -> None:
    """Save failed documents with statistics."""
    if not failed_docs:
        return
    
    # Ensure directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    success_count = total_processed - len(failed_docs)
    success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
    
    output_data = {
        'total_failed': len(failed_docs),
        'total_processed': total_processed,
        'success_count': success_count,
        'success_rate': f"{success_rate:.2f}%",
        'failed_documents': failed_docs
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=indent)
    
    print(f"⚠ {len(failed_docs)} documents failed. Details saved to {output_file}")


def get_missing_publications(df: pd.DataFrame, existing_abstracts: Dict) -> pd.DataFrame:
    """
    Filter out publications that already have abstracts.
    Matches by title (case-insensitive comparison).
    """
    existing_titles = {info['title'].lower().strip() for info in existing_abstracts.values()}
    
    # Filter rows where title is not in existing abstracts
    missing_mask = ~df['title'].str.lower().str.strip().isin(existing_titles)
    missing_df = df[missing_mask].copy()
    
    print(f"\nTotal publications in Excel: {len(df)}")
    print(f"Publications with existing abstracts: {len(df) - len(missing_df)}")
    print(f"Publications missing abstracts: {len(missing_df)}")
    
    return missing_df

