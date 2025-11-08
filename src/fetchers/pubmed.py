"""
PubMed fetcher using NCBI E-utilities.
"""
from typing import Optional, Tuple
from xml.etree import ElementTree as ET
import requests
from .base import AbstractFetcher
from config import NCBI_ESEARCH_URL, NCBI_EFETCH_URL, NCBI_API_KEY, NCBI_DELAY


class PubMedFetcher(AbstractFetcher):
    """Fetcher for PubMed/NCBI database."""
    
    def __init__(self):
        super().__init__(name="PubMed", delay=NCBI_DELAY)
        self.api_key = NCBI_API_KEY
        self.search_url = NCBI_ESEARCH_URL
        self.fetch_url = NCBI_EFETCH_URL
    
    def _get_pmid_from_title(self, title: str) -> Optional[str]:
        """Get PMID from title using ESearch."""
        params = {
            "db": "pubmed",
            "term": f"{title}[Title]",
            "retmax": 1,
            "retmode": "xml"
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = self._make_request(self.search_url, params=params)
        if not response:
            return None
        
        try:
            root = ET.fromstring(response.text)
            id_list = root.find("IdList")
            if id_list is not None and len(id_list) > 0:
                return id_list[0].text
        except ET.ParseError:
            pass
        
        return None
    
    def _get_abstract_from_pmid(self, pmid: str) -> Optional[str]:
        """Get abstract from PMID using EFetch."""
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "text",
            "rettype": "abstract"
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = self._make_request(self.fetch_url, params=params)
        if response:
            abstract = response.text.strip()
            if abstract and len(abstract) > 50:
                return abstract
        
        return None
    
    def fetch_by_title(self, title: str) -> Optional[Tuple[str, str]]:
        """Fetch abstract by title."""
        pmid = self._get_pmid_from_title(title)
        if pmid:
            abstract = self._get_abstract_from_pmid(pmid)
            if abstract:
                return (pmid, abstract)
        return None
    
    def fetch_by_id(self, pmid: str) -> Optional[Tuple[str, str]]:
        """Fetch abstract by PMID."""
        abstract = self._get_abstract_from_pmid(pmid)
        if abstract:
            return (pmid, abstract)
        return None

