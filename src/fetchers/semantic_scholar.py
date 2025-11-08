"""Semantic Scholar fetcher with retry logic."""
from typing import Optional, Tuple
import time
import requests
from .base import AbstractFetcher
from config import (
    SEMANTIC_SCHOLAR_API_KEY, 
    SEMANTIC_SCHOLAR_DELAY,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE
)


class SemanticScholarFetcher(AbstractFetcher):
    """Fetcher for Semantic Scholar with retry logic for rate limits."""
    
    def __init__(self):
        super().__init__(name="Semantic Scholar", delay=SEMANTIC_SCHOLAR_DELAY)
        self.api_key = SEMANTIC_SCHOLAR_API_KEY
        self.url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.max_retries = MAX_RETRIES
        self.backoff_base = RETRY_BACKOFF_BASE
    
    def fetch_by_title(self, title: str) -> Optional[Tuple[str, str]]:
        """Fetch abstract by title with retry logic."""
        params = {
            "query": title,
            "fields": "abstract,externalIds",
            "limit": 1
        }
        
        headers = {}
        if self.api_key:
            headers['x-api-key'] = self.api_key
        
        for attempt in range(self.max_retries):
            self._rate_limit()
            
            try:
                response = requests.get(self.url, params=params, headers=headers, timeout=self.timeout)
                
                # Handle 429 with retry
                if response.status_code == 429:
                    retry_wait = (attempt + 1) * self.backoff_base
                    if attempt < self.max_retries - 1:
                        print(f"429 (retry {attempt + 1}/{self.max_retries}, waiting {retry_wait}s)...", end=" ", flush=True)
                        time.sleep(retry_wait)
                        continue
                    else:
                        print(f"429 (max retries)")
                        return None
                
                response.raise_for_status()
                data = response.json()
                
                if 'data' in data and data['data']:
                    paper = data['data'][0]
                    abstract = paper.get('abstract')
                    external_ids = paper.get('externalIds', {})
                    pmid = external_ids.get('PubMed', '')
                    
                    if abstract:
                        return (pmid, abstract.strip())
                
                return None
                
            except requests.exceptions.HTTPError as e:
                if attempt < self.max_retries - 1 and hasattr(e.response, 'status_code') and e.response.status_code == 429:
                    retry_wait = (attempt + 1) * self.backoff_base
                    print(f"429 (retry {attempt + 1}/{self.max_retries}, waiting {retry_wait}s)...", end=" ", flush=True)
                    time.sleep(retry_wait)
                    continue
                else:
                    return None
            except Exception:
                return None
        
        return None

