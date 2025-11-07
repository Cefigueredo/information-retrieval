import json
import requests
import time
import pandas as pd
from typing import Dict, List, Optional, Tuple
from config import (
    EUROPE_PMC_URL,
    CROSSREF_URL,
    NCBI_ESEARCH_URL,
    NCBI_EFETCH_URL,
    PUBLICATIONS_FILE,
    RELEVANT_ABSTRACTS_FILE,
    ALTERNATIVE_ABSTRACTS_FILE,
    ALTERNATIVE_FAILED_FILE,
    GENERAL_API_DELAY,
    SEMANTIC_SCHOLAR_DELAY,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    API_TIMEOUT,
    SEMANTIC_SCHOLAR_API_KEY
)

# API URLs
PUBMED_SEARCH_URL = NCBI_ESEARCH_URL
PUBMED_FETCH_URL = NCBI_EFETCH_URL


def load_existing_abstracts(json_file: str = None) -> Dict:
    """Load existing abstracts from JSON file."""
    json_file = json_file or RELEVANT_ABSTRACTS_FILE
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {json_file} not found. Starting with empty abstracts.")
        return {}


def load_publications(excel_file: str = None) -> pd.DataFrame:
    """Load publications from Excel file."""
    excel_file = excel_file or PUBLICATIONS_FILE
    return pd.read_excel(excel_file)


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


def fetch_abstract_from_europe_pmc_by_title(title: str) -> Optional[Tuple[str, str]]:
    """
    Fetch abstract from Europe PMC API by title.
    Returns tuple of (pmid, abstract) if found, None otherwise.
    """
    params = {
        "query": f'TITLE:"{title}"',
        "resultType": "core",
        "format": "json",
        "pageSize": 1
    }
    try:
        response = requests.get(EUROPE_PMC_URL, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        if 'resultList' in data and 'result' in data['resultList'] and data['resultList']['result']:
            result = data['resultList']['result'][0]
            abstract = result.get('abstractText', '')
            pmid = result.get('pmid', '')
            
            if abstract and pmid:
                return (pmid, abstract.strip())
    except Exception as e:
        print(f"error: {e}")
    
    return None


def fetch_abstract_from_semantic_scholar_by_title(title: str, max_retries: int = MAX_RETRIES) -> Optional[Tuple[str, str]]:
    """
    Fetch abstract from Semantic Scholar API by title with retry logic for 429 errors.
    Uses API key if available for higher rate limits.
    Returns tuple of (external_id, abstract) if found, None otherwise.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "fields": "abstract,externalIds",
        "limit": 1
    }
    
    # Add API key to headers if available
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=API_TIMEOUT)
            
            # Check for rate limit error (429)
            if response.status_code == 429:
                retry_wait = (attempt + 1) * RETRY_BACKOFF_BASE  # Progressive delay
                if attempt < max_retries - 1:
                    print(f"429 (retry {attempt + 1}/{max_retries}, waiting {retry_wait}s)...", end=" ", flush=True)
                    time.sleep(retry_wait)
                    continue
                else:
                    print(f"429 (max retries reached)")
                    return None
            
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and data['data']:
                paper = data['data'][0]
                abstract = paper.get('abstract')
                external_ids = paper.get('externalIds', {})
                pmid = external_ids.get('PubMed', '')
                
                if abstract:
                    return (pmid, abstract.strip())
            
            # If no abstract found, no need to retry
            return None
            
        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1 and hasattr(e.response, 'status_code') and e.response.status_code == 429:
                retry_wait = (attempt + 1) * RETRY_BACKOFF_BASE
                print(f"429 (retry {attempt + 1}/{max_retries}, waiting {retry_wait}s)...", end=" ", flush=True)
                time.sleep(retry_wait)
                continue
            else:
                print(f"HTTP error: {e}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    return None


def fetch_abstract_from_crossref_by_title(title: str) -> Optional[Tuple[str, str]]:
    """
    Fetch abstract from CrossRef API by title.
    Returns tuple of (doi, abstract) if found, None otherwise.
    """
    params = {
        "query.title": title,
        "rows": 1
    }
    
    try:
        response = requests.get(CROSSREF_URL, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        if 'message' in data and 'items' in data['message'] and data['message']['items']:
            item = data['message']['items'][0]
            abstract = item.get('abstract', '')
            doi = item.get('DOI', '')
            
            if abstract:
                # CrossRef abstracts may contain XML tags, clean them
                import re
                abstract = re.sub(r'<[^>]+>', '', abstract)
                return (doi, abstract.strip())
    except Exception as e:
        print(f"error: {e}")
    
    return None


def fetch_abstract_from_pubmed_by_title(title: str) -> Optional[Tuple[str, str]]:
    """
    Fetch abstract from PubMed by title using E-utilities.
    Returns tuple of (pmid, abstract) if found, None otherwise.
    """
    # First, search for PMID
    search_params = {
        "db": "pubmed",
        "term": f'"{title}"[Title]',
        "retmax": 1,
        "retmode": "json"
    }
    
    try:
        search_response = requests.get(PUBMED_SEARCH_URL, params=search_params, timeout=API_TIMEOUT)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        if 'esearchresult' in search_data and 'idlist' in search_data['esearchresult']:
            id_list = search_data['esearchresult']['idlist']
            
            if id_list:
                pmid = id_list[0]
                
                # Fetch abstract using PMID
                fetch_params = {
                    "db": "pubmed",
                    "id": pmid,
                    "retmode": "text",
                    "rettype": "abstract"
                }
                
                fetch_response = requests.get(PUBMED_FETCH_URL, params=fetch_params, timeout=API_TIMEOUT)
                fetch_response.raise_for_status()
                abstract = fetch_response.text.strip()
                
                if abstract and len(abstract) > 50:
                    return (pmid, abstract)
    except Exception as e:
        print(f"error: {e}")
    
    return None


def search_alternative_abstracts(missing_df: pd.DataFrame) -> Dict:
    """
    Search for abstracts using alternative methods in two phases:
    Phase 1: Try Europe PMC, CrossRef, and PubMed for all documents
    Phase 2: If documents still missing, try Semantic Scholar (has stricter rate limits)
    """
    results = {}
    still_missing = []
    
    # Rate limiting configuration from config
    # Using imported values: GENERAL_API_DELAY and SEMANTIC_SCHOLAR_DELAY
    
    print("\n" + "="*80)
    print("PHASE 1: Searching with Europe PMC, CrossRef, and PubMed")
    print("="*80)
    
    # Phase 1: Try fast APIs first (Europe PMC, CrossRef, PubMed)
    for counter, (idx, row) in enumerate(missing_df.iterrows(), start=1):
        title = row['title']
        print(f"[{counter}/{len(missing_df)}] Searching: {title[:60]}...")
        
        abstract_found = False
        pmid = None
        abstract = None
        source = None
        
        # Try Europe PMC first
        print("  Trying Europe PMC...", end=" ", flush=True)
        result = fetch_abstract_from_europe_pmc_by_title(title)
        if result:
            pmid, abstract = result
            source = "Europe PMC"
            abstract_found = True
            print("✓")
        else:
            print("✗")
        
        # Try CrossRef if still not found
        if not abstract_found:
            print("  Trying CrossRef...", end=" ", flush=True)
            result = fetch_abstract_from_crossref_by_title(title)
            if result:
                pmid, abstract = result
                source = "CrossRef"
                abstract_found = True
                print("✓")
            else:
                print("✗")
        
        # Try PubMed if still not found
        if not abstract_found:
            print("  Trying PubMed E-utilities...", end=" ", flush=True)
            result = fetch_abstract_from_pubmed_by_title(title)
            if result:
                pmid, abstract = result
                source = "PubMed"
                abstract_found = True
                print("✓")
            else:
                print("✗")
        
        # Store results or mark as still missing
        if abstract_found:
            identifier = pmid if pmid else f"alt_{idx}"
            results[identifier] = {
                'title': title,
                'abstract': abstract,
                'source': source
            }
            print(f"  ✓ Success! Found via {source} (PMID/ID: {identifier})")
        else:
            still_missing.append((idx, title))
            print(f"  ✗ Not found in Phase 1")
        
        print()  # Empty line between entries
        time.sleep(GENERAL_API_DELAY)
    
    # Phase 2: Try Semantic Scholar for documents still missing abstracts
    failed_documents = []
    
    if still_missing:
        print("\n" + "="*80)
        print(f"PHASE 2: Trying Semantic Scholar for {len(still_missing)} remaining documents")
        print("="*80)
        
        for counter, (idx, title) in enumerate(still_missing, start=1):
            print(f"[{counter}/{len(still_missing)}] Searching: {title[:60]}...")
            print("  Trying Semantic Scholar...", end=" ", flush=True)
            
            result = fetch_abstract_from_semantic_scholar_by_title(title)
            if result:
                pmid, abstract = result
                source = "Semantic Scholar"
                identifier = pmid if pmid else f"alt_{idx}"
                results[identifier] = {
                    'title': title,
                    'abstract': abstract,
                    'source': source
                }
                print("✓")
                print(f"  ✓ Success! Found via {source} (PMID/ID: {identifier})")
            else:
                print("✗")
                failed_documents.append({
                    'index': idx,
                    'title': title,
                    'error': 'Abstract not found in any source'
                })
                print(f"  ✗ Failed: No abstract found in any source")
            
            print()  # Empty line between entries
            print(f"  [Waiting {SEMANTIC_SCHOLAR_DELAY}s for Semantic Scholar rate limit...]")
            time.sleep(SEMANTIC_SCHOLAR_DELAY)
    else:
        print("\n✓ All abstracts found in Phase 1! No need for Semantic Scholar.")
    
    return results, failed_documents


def save_results(results: Dict, failed: List, 
                output_file: str = None,
                failed_file: str = None):
    """Save successfully retrieved abstracts and failed attempts."""
    # Use config defaults if not specified
    output_file = output_file or ALTERNATIVE_ABSTRACTS_FILE
    failed_file = failed_file or ALTERNATIVE_FAILED_FILE
    
    # Save successful abstracts
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved {len(results)} abstracts to {output_file}")
    
    # Save failed documents
    if failed:
        with open(failed_file, 'w') as f:
            json.dump({
                'total_failed': len(failed),
                'failed_documents': failed
            }, f, indent=2)
        print(f"⚠ Saved {len(failed)} failed documents to {failed_file}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("ALTERNATIVE ABSTRACT RETRIEVAL")
    print("=" * 80)
    
    # Load existing data
    print("\nLoading existing data...")
    existing_abstracts = load_existing_abstracts()  # Uses config default
    publications_df = load_publications()  # Uses config default
    
    # Filter out publications that already have abstracts
    print("\nFiltering publications...")
    missing_df = get_missing_publications(publications_df, existing_abstracts)
    
    if len(missing_df) == 0:
        print("\n✓ All publications already have abstracts!")
        return
    
    # Search for missing abstracts using alternative methods
    print("\nSearching for missing abstracts using alternative APIs...")
    print("-" * 80)
    results, failed = search_alternative_abstracts(missing_df)
    
    # Save results
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total missing publications: {len(missing_df)}")
    print(f"Successfully retrieved: {len(results)}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {len(results)/len(missing_df)*100:.2f}%")
    
    save_results(results, failed)
    
    # Show sources breakdown
    if results:
        sources = {}
        for data in results.values():
            source = data.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        print("\nAbstracts by source:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()