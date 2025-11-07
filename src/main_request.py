import requests
import pandas as pd
import time
import json
from xml.etree import ElementTree as ET
from config import (
    NCBI_API_KEY,
    NCBI_ESEARCH_URL,
    NCBI_EFETCH_URL,
    NCBI_DELAY,
    PUBLICATIONS_FILE,
    RELEVANT_ABSTRACTS_FILE,
    FAILED_DOCUMENTS_FILE,
    API_TIMEOUT
)

# API Key
API_KEY = NCBI_API_KEY

# NCBI E-utilities URLs
ESEARCH_URL = NCBI_ESEARCH_URL
EFETCH_URL = NCBI_EFETCH_URL

def get_pmid_from_title(title):
    """Use ESearch to get PMID from title."""
    params = {
        "db": "pubmed",
        "term": f"{title}[Title]",
        "retmax": 1,  # Get top 1 result
        "retmode": "xml"
    }
    if API_KEY:
        params["api_key"] = API_KEY
    response = requests.get(ESEARCH_URL, params=params)
    if response.status_code != 200:
        return None
    root = ET.fromstring(response.text)
    id_list = root.find("IdList")
    if id_list is not None and len(id_list) > 0:
        return id_list[0].text
    return None

def get_abstract_from_pmid(pmid):
    """Use EFetch to get abstract from PMID."""
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "text",
        "rettype": "abstract"
    }
    if API_KEY:
        params["api_key"] = API_KEY
    response = requests.get(EFETCH_URL, params=params)
    if response.status_code == 200:
        return response.text.strip()
    return None

def fetch_relevant_abstracts(excel_file=None, output_file=None, error_file=None):
    """Task 1: Fetch abstracts for relevant articles from Excel."""
    # Use config defaults if not specified
    excel_file = excel_file or PUBLICATIONS_FILE
    output_file = output_file or RELEVANT_ABSTRACTS_FILE
    error_file = error_file or FAILED_DOCUMENTS_FILE
    
    df = pd.read_excel(excel_file)
    abstracts = {}
    failed_documents = []
    
    for idx, row in df.iterrows():
        title = row['title']  # Assume 'Title' column; adjust if different
        pmid = get_pmid_from_title(title)
        
        if pmid:
            abstract = get_abstract_from_pmid(pmid)
            if abstract:
                abstracts[pmid] = {'title': title, 'abstract': abstract}
                print(f"✓ Processed {idx+1}/{len(df)}: {title[:50]}...")
            else:
                # Abstract not found for this PMID
                failed_documents.append({
                    'row_number': idx + 1,
                    'title': title,
                    'pmid': pmid,
                    'error': 'Abstract not found'
                })
                print(f"✗ Failed {idx+1}/{len(df)}: Abstract not found for PMID {pmid}")
        else:
            # PMID not found for this title
            failed_documents.append({
                'row_number': idx + 1,
                'title': title,
                'pmid': None,
                'error': 'PMID not found'
            })
            print(f"✗ Failed {idx+1}/{len(df)}: PMID not found for title")
        
        time.sleep(NCBI_DELAY)  # Delay to respect rate limits
    
    # Save successful abstracts
    with open(output_file, 'w') as f:
        json.dump(abstracts, f, indent=2)
    
    # Save failed documents
    if failed_documents:
        with open(error_file, 'w') as f:
            json.dump({
                'total_failed': len(failed_documents),
                'total_processed': len(df),
                'success_rate': f"{((len(df) - len(failed_documents)) / len(df) * 100):.2f}%",
                'failed_documents': failed_documents
            }, f, indent=2)
        print(f"\n⚠ {len(failed_documents)} documents failed. Details saved to {error_file}")
    
    print(f"\n✓ Successfully retrieved {len(abstracts)} abstracts")
    return abstracts

def fetch_non_relevant_abstracts(num_abstracts=1308, query="neurology NOT (polyphenol OR antioxidant OR food OR fruit OR vegetable)", output_file='non_relevant_abstracts.json', error_file='failed_non_relevant_documents.json'):
    """Task 2: Fetch non-relevant abstracts."""
    # First, get list of PMIDs using ESearch with retmax
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": num_abstracts,
        "retmode": "xml"
    }
    if API_KEY:
        params["api_key"] = API_KEY
    
    response = requests.get(ESEARCH_URL, params=params)
    if response.status_code != 200:
        raise ValueError(f"ESearch failed with status code {response.status_code}")
    
    root = ET.fromstring(response.text)
    id_list = [id_elem.text for id_elem in root.findall("IdList/Id")]
    
    if len(id_list) < num_abstracts:
        print(f"⚠ Warning: Only {len(id_list)} PMIDs found; requested {num_abstracts}")
    
    abstracts = {}
    failed_documents = []
    
    for idx, pmid in enumerate(id_list[:num_abstracts]):
        abstract = get_abstract_from_pmid(pmid)
        if abstract:
            abstracts[pmid] = {'abstract': abstract}
            print(f"✓ Processed {idx+1}/{len(id_list[:num_abstracts])}: PMID {pmid}")
        else:
            # Abstract not found for this PMID
            failed_documents.append({
                'pmid': pmid,
                'error': 'Abstract not found'
            })
            print(f"✗ Failed {idx+1}/{len(id_list[:num_abstracts])}: Abstract not found for PMID {pmid}")
        
        time.sleep(NCBI_DELAY)
    
    # Save successful abstracts
    with open(output_file, 'w') as f:
        json.dump(abstracts, f, indent=2)
    
    # Save failed documents
    if failed_documents:
        with open(error_file, 'w') as f:
            json.dump({
                'total_failed': len(failed_documents),
                'total_processed': len(id_list[:num_abstracts]),
                'success_rate': f"{((len(id_list[:num_abstracts]) - len(failed_documents)) / len(id_list[:num_abstracts]) * 100):.2f}%",
                'failed_documents': failed_documents
            }, f, indent=2)
        print(f"\n⚠ {len(failed_documents)} documents failed. Details saved to {error_file}")
    
    print(f"\n✓ Successfully retrieved {len(abstracts)} abstracts")
    return abstracts

# Usage
fetch_relevant_abstracts()  # Uncomment to run Task 1
# fetch_non_relevant_abstracts()  # Uncomment to run Task 2