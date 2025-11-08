"""
OpenAlex fetcher.
"""
from typing import Optional, Tuple
from .base import AbstractFetcher
from config import OPENALEX_URL, GENERAL_API_DELAY


class OpenAlexFetcher(AbstractFetcher):
    """Fetcher for OpenAlex database."""
    
    def __init__(self):
        super().__init__(name="OpenAlex", delay=GENERAL_API_DELAY)
        self.url = OPENALEX_URL
    
    def fetch_by_title(self, title: str) -> Optional[Tuple[str, str]]:
        """Fetch abstract by title."""
        params = {
            "search": title,
            "filter": "type:article",
            "per-page": 1
        }
        
        response = self._make_request(self.url, params=params)
        if not response:
            return None
        
        try:
            data = response.json()
            if 'results' in data and data['results']:
                work = data['results'][0]
                abstract_inv = work.get('abstract_inverted_index')
                
                if abstract_inv:
                    # Reconstruct abstract from inverted index
                    words = {}
                    for word, positions in abstract_inv.items():
                        for pos in positions:
                            words[pos] = word
                    abstract_text = ' '.join([words[i] for i in sorted(words.keys())])
                    
                    # Get identifier
                    identifier = work.get('doi', work.get('id', '')).replace('https://openalex.org/', '')
                    
                    if abstract_text:
                        return (identifier, abstract_text.strip())
        except Exception:
            pass
        
        return None

