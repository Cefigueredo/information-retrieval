import requests
import time
import json
import os
from xml.etree import ElementTree as ET
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import (
    NCBI_API_KEY,
    NCBI_ESEARCH_URL,
    NCBI_EFETCH_URL,
    NCBI_DELAY,
)

# API Key
API_KEY = NCBI_API_KEY

# NCBI E-utilities URLs
ESEARCH_URL = NCBI_ESEARCH_URL
EFETCH_URL = NCBI_EFETCH_URL

# Create session with retry strategy
def create_session_with_retries(retries=3, backoff_factor=0.5):
    """Create requests session with automatic retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_abstract_from_pmid(pmid, session=None, max_retries=3):
    """Use EFetch to get abstract from PMID with retry logic."""
    if session is None:
        session = create_session_with_retries()
    
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "text",
        "rettype": "abstract"
    }
    if API_KEY:
        params["api_key"] = API_KEY
    
    for attempt in range(max_retries):
        try:
            response = session.get(EFETCH_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.text.strip()
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2  # exponential backoff: 2, 4, 8 seconds
                print(f"⚠ Connection error for PMID {pmid} (attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"✗ Failed to fetch PMID {pmid} after {max_retries} attempts: {str(e)}")
                return None
    
    return None

def load_existing_abstracts(output_file):
    """Load already fetched abstracts to resume from checkpoint."""
    folder_name = "data/output/"
    filepath = folder_name + output_file
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠ Could not parse existing {output_file}, starting fresh")
            return {}
    return {}

def fetch_non_relevant_abstracts(num_abstracts=1308, query="neurology NOT (polyphenol OR antioxidant OR food OR fruit OR vegetable)", output_file='non_relevant_abstracts.json', error_file='failed_non_relevant_documents.json'):
    """Task 2: Fetch non-relevant abstracts with resumability and retry logic."""
    folder_name = "data/output/"
    os.makedirs(folder_name, exist_ok=True)
    
    # Load existing abstracts to resume from checkpoint
    abstracts = load_existing_abstracts(output_file)
    existing_pmids = set(abstracts.keys())
    
    # First, get list of PMIDs using ESearch with retmax
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": num_abstracts,
        "retmode": "xml"
    }
    if API_KEY:
        params["api_key"] = API_KEY
    
    try:
        response = requests.get(ESEARCH_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"ESearch failed: {str(e)}")
    
    root = ET.fromstring(response.text)
    id_list = [id_elem.text for id_elem in root.findall("IdList/Id")]
    
    if len(id_list) < num_abstracts:
        print(f"⚠ Warning: Only {len(id_list)} PMIDs found; requested {num_abstracts}")
    
    failed_documents = []
    session = create_session_with_retries()
    
    for idx, pmid in enumerate(id_list[:num_abstracts]):
        # Skip if already fetched
        if pmid in existing_pmids:
            print(f"⊘ Skipped {idx+1}/{len(id_list[:num_abstracts])}: PMID {pmid} (already fetched)")
            continue
        
        abstract = get_abstract_from_pmid(pmid, session=session)
        if abstract:
            abstracts[pmid] = {'abstract': abstract}
            print(f"✓ Processed {idx+1}/{len(id_list[:num_abstracts])}: PMID {pmid}")
        else:
            # Abstract not found for this PMID
            failed_documents.append({
                'pmid': pmid,
                'error': 'Abstract not found or network error'
            })
            print(f"✗ Failed {idx+1}/{len(id_list[:num_abstracts])}: PMID {pmid}")
        
        time.sleep(NCBI_DELAY)
    
    # Save successful abstracts
    with open(folder_name + output_file, 'w') as f:
        json.dump(abstracts, f, indent=2)
    
    # Save failed documents
    if failed_documents:
        with open(folder_name + error_file, 'w') as f:
            json.dump({
                'total_failed': len(failed_documents),
                'total_processed': len(id_list[:num_abstracts]),
                'success_rate': f"{((len(id_list[:num_abstracts]) - len(failed_documents)) / len(id_list[:num_abstracts]) * 100):.2f}%",
                'failed_documents': failed_documents
            }, f, indent=2)
        print(f"\n⚠ {len(failed_documents)} documents failed. Details saved to {folder_name + error_file}")
    
    print(f"\n✓ Successfully retrieved {len(abstracts)} abstracts")
    return abstracts

if __name__ == "__main__":
    fetch_non_relevant_abstracts()