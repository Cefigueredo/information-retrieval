"""bioRxiv/medRxiv fetcher."""
from typing import Optional, Tuple
from datetime import datetime, timedelta
from .base import AbstractFetcher
from config import BIORXIV_URL, MEDRXIV_URL, GENERAL_API_DELAY


class BioRxivFetcher(AbstractFetcher):
    """Fetcher for bioRxiv and medRxiv preprints."""
    
    def __init__(self, server: str = "biorxiv"):
        """
        Initialize fetcher.
        
        Args:
            server: Either "biorxiv" or "medrxiv"
        """
        name = "bioRxiv" if server == "biorxiv" else "medRxiv"
        super().__init__(name=name, delay=GENERAL_API_DELAY, timeout=20)
        self.base_url = BIORXIV_URL if server == "biorxiv" else MEDRXIV_URL
    
    def fetch_by_title(self, title: str) -> Optional[Tuple[str, str]]:
        """Fetch abstract by title."""
        try:
            # API requires date range - search last 2 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            url = f"{self.base_url}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            response = self._make_request(url)
            if not response:
                return None
            
            data = response.json()
            if 'collection' in data:
                # Search for matching title
                for item in data['collection']:
                    if item.get('title', '').lower() == title.lower():
                        abstract = item.get('abstract')
                        doi = item.get('doi')
                        if abstract and doi:
                            return (doi, abstract.strip())
        except Exception:
            pass
        
        return None

