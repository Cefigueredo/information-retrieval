"""CrossRef fetcher."""
from typing import Optional, Tuple
import re
from .base import AbstractFetcher
from config import CROSSREF_URL, GENERAL_API_DELAY


class CrossRefFetcher(AbstractFetcher):
    """Fetcher for CrossRef database."""
    
    def __init__(self):
        super().__init__(name="CrossRef", delay=GENERAL_API_DELAY)
        self.url = CROSSREF_URL
    
    def fetch_by_title(self, title: str) -> Optional[Tuple[str, str]]:
        """Fetch abstract by title."""
        params = {"query.title": title, "rows": 1}
        response = self._make_request(self.url, params=params)
        if not response:
            return None
        
        try:
            data = response.json()
            if 'message' in data and 'items' in data['message'] and data['message']['items']:
                item = data['message']['items'][0]
                abstract = item.get('abstract', '')
                doi = item.get('DOI', '')
                
                if abstract:
                    # Clean XML tags
                    abstract = re.sub(r'<[^>]+>', '', abstract)
                    return (doi, abstract.strip())
        except Exception:
            pass
        
        return None

