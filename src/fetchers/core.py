"""CORE fetcher."""
from typing import Optional, Tuple
from .base import AbstractFetcher
from config import CORE_URL, CORE_API_KEY, GENERAL_API_DELAY


class COREFetcher(AbstractFetcher):
    """Fetcher for CORE open access database."""
    
    def __init__(self):
        super().__init__(name="CORE", delay=GENERAL_API_DELAY)
        self.url = CORE_URL
        self.api_key = CORE_API_KEY
    
    def fetch_by_title(self, title: str) -> Optional[Tuple[str, str]]:
        """Fetch abstract by title."""
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        params = {'title': title, 'limit': 1}
        response = self._make_request(self.url, params=params, headers=headers)
        if not response:
            return None
        
        try:
            data = response.json()
            if 'results' in data and data['results']:
                work = data['results'][0]
                abstract = work.get('abstract')
                core_id = work.get('id', '')
                
                if abstract:
                    return (str(core_id), abstract.strip())
        except Exception:
            pass
        
        return None

