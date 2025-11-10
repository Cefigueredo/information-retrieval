import requests
import time
import json
from xml.etree import ElementTree as ET
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
    
    folder_name = "data/output/"
    
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