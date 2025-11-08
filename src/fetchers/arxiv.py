"""arXiv fetcher."""
from typing import Optional, Tuple
from xml.etree import ElementTree as ET
from .base import AbstractFetcher
from config import ARXIV_URL, GENERAL_API_DELAY


class ArXivFetcher(AbstractFetcher):
    """Fetcher for arXiv preprints."""
    
    def __init__(self):
        super().__init__(name="arXiv", delay=GENERAL_API_DELAY)
        self.url = ARXIV_URL
    
    def fetch_by_title(self, title: str) -> Optional[Tuple[str, str]]:
        """Fetch abstract by title."""
        params = {"search_query": f"ti:{title}", "start": 0, "max_results": 1}
        response = self._make_request(self.url, params=params)
        if not response:
            return None
        
        try:
            root = ET.fromstring(response.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            
            if entries:
                entry = entries[0]
                summary = entry.find('atom:summary', ns)
                arxiv_id_elem = entry.find('atom:id', ns)
                
                if summary is not None and summary.text and arxiv_id_elem is not None:
                    abstract = summary.text.strip()
                    arxiv_id = arxiv_id_elem.text.split('/')[-1]
                    if abstract:
                        return (arxiv_id, abstract)
        except Exception:
            pass
        
        return None

