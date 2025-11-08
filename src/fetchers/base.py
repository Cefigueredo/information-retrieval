"""
Base class for abstract fetchers.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import requests
import time


class AbstractFetcher(ABC):
    """Base class for all abstract fetchers."""
    
    def __init__(self, name: str, timeout: int = 10, delay: float = 0.5):
        """
        Initialize fetcher.
        
        Args:
            name: Name of the data source
            timeout: Request timeout in seconds
            delay: Delay between requests in seconds
        """
        self.name = name
        self.timeout = timeout
        self.delay = delay
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        if self.delay > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()
    
    @abstractmethod
    def fetch_by_title(self, title: str) -> Optional[Tuple[str, str]]:
        """
        Fetch abstract by publication title.
        
        Args:
            title: Publication title
            
        Returns:
            Tuple of (identifier, abstract) if found, None otherwise
        """
        pass
    
    def fetch_by_id(self, identifier: str) -> Optional[Tuple[str, str]]:
        """
        Fetch abstract by identifier (PMID, DOI, etc.).
        
        Args:
            identifier: Publication identifier
            
        Returns:
            Tuple of (identifier, abstract) if found, None otherwise
        """
        # Default implementation - can be overridden
        return None
    
    def _make_request(self, url: str, params: dict = None, headers: dict = None) -> Optional[requests.Response]:
        """
        Make HTTP request with error handling.
        
        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers
            
        Returns:
            Response object if successful, None otherwise
        """
        self._rate_limit()
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"error: {e}")
            return None
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"

