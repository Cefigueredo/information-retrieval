"""
Europe PMC fetcher.
"""
from typing import Optional, Tuple
from .base import AbstractFetcher
from config import EUROPE_PMC_URL, GENERAL_API_DELAY


class EuropePMCFetcher(AbstractFetcher):
    """Fetcher for Europe PMC database."""
    
    def __init__(self):
        super().__init__(name="Europe PMC", delay=GENERAL_API_DELAY)
        self.url = EUROPE_PMC_URL
    
    def fetch_by_title(self, title: str) -> Optional[Tuple[str, str]]:
        """Fetch abstract by title."""
        params = {
            "query": f'TITLE:"{title}"',
            "resultType": "core",
            "format": "json",
            "pageSize": 1
        }
        
        response = self._make_request(self.url, params=params)
        if not response:
            return None
        
        try:
            data = response.json()
            if 'resultList' in data and 'result' in data['resultList'] and data['resultList']['result']:
                result = data['resultList']['result'][0]
                abstract = result.get('abstractText', '')
                pmid = result.get('pmid', '')
                
                if abstract and pmid:
                    return (pmid, abstract.strip())
        except Exception:
            pass
        
        return None

