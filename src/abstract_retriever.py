"""
Unified abstract retrieval engine.
"""
import time
from typing import List, Dict, Tuple
import pandas as pd

from fetchers import (
    PubMedFetcher,
    EuropePMCFetcher,
    OpenAlexFetcher,
    CrossRefFetcher,
    ArXivFetcher,
    COREFetcher,
    BioRxivFetcher,
    SemanticScholarFetcher
)
from utils import ProgressTracker
from config import GENERAL_API_DELAY


class AbstractRetriever:
    """
    Unified engine for retrieving abstracts from multiple sources.
    
    Uses a three-phase approach:
    - Phase 1: Fast & reliable APIs
    - Phase 2: Additional specialized sources
    - Phase 3: Rate-limited APIs (only if needed)
    """
    
    def __init__(self):
        """Initialize retriever with all fetchers."""
        # Phase 1: Fast & reliable
        self.phase1_fetchers = [
            EuropePMCFetcher(),
            OpenAlexFetcher(),
            CrossRefFetcher(),
            PubMedFetcher()
        ]
        
        # Phase 2: Additional sources
        self.phase2_fetchers = [
            ArXivFetcher(),
            COREFetcher(),
            BioRxivFetcher("biorxiv"),
            BioRxivFetcher("medrxiv")
        ]
        
        # Phase 3: Rate-limited
        self.phase3_fetchers = [
            SemanticScholarFetcher()
        ]
    
    def retrieve_abstracts(self, 
                          df: pd.DataFrame,
                          use_phases: List[int] = [1, 2, 3]) -> Tuple[Dict, List[Dict]]:
        """
        Retrieve abstracts for all publications in dataframe.
        
        Args:
            df: DataFrame with 'title' column
            use_phases: Which phases to use (1, 2, 3)
            
        Returns:
            Tuple of (results dict, failed documents list)
        """
        results = {}
        still_missing = []
        
        # Phase 1
        if 1 in use_phases:
            tracker = ProgressTracker(len(df))
            tracker.start_phase("PHASE 1: Fast & Reliable APIs")
            
            for idx, row in df.iterrows():
                title = row['title']
                tracker.update(title)
                
                found = self._try_fetchers(self.phase1_fetchers, title, idx, results, tracker)
                if not found:
                    still_missing.append((idx, title))
                
                print()  # Empty line
                time.sleep(GENERAL_API_DELAY)
        else:
            still_missing = [(idx, row['title']) for idx, row in df.iterrows()]
        
        # Phase 2
        still_missing_phase2 = []
        if 2 in use_phases and still_missing:
            tracker = ProgressTracker(len(still_missing))
            tracker.start_phase(f"PHASE 2: Additional Sources ({len(still_missing)} remaining)")
            
            for idx, title in still_missing:
                tracker.update(title)
                
                found = self._try_fetchers(self.phase2_fetchers, title, idx, results, tracker)
                if not found:
                    still_missing_phase2.append((idx, title))
                
                print()  # Empty line
                time.sleep(GENERAL_API_DELAY)
        elif still_missing:
            still_missing_phase2 = still_missing
        
        # Phase 3
        failed_documents = []
        if 3 in use_phases and still_missing_phase2:
            tracker = ProgressTracker(len(still_missing_phase2))
            tracker.start_phase(f"PHASE 3: Rate-Limited APIs ({len(still_missing_phase2)} remaining)")
            
            for idx, title in still_missing_phase2:
                tracker.update(title)
                
                found = self._try_fetchers(self.phase3_fetchers, title, idx, results, tracker)
                if not found:
                    failed_documents.append({
                        'index': idx,
                        'title': title,
                        'error': 'Abstract not found in any source'
                    })
                
                print()  # Empty line
        elif still_missing_phase2:
            failed_documents = [
                {'index': idx, 'title': title, 'error': 'Phases not enabled'}
                for idx, title in still_missing_phase2
            ]
        
        return results, failed_documents
    
    def _try_fetchers(self, 
                     fetchers: List,
                     title: str,
                     idx: int,
                     results: Dict,
                     tracker: ProgressTracker) -> bool:
        """
        Try multiple fetchers for a single publication.
        
        Returns:
            True if abstract found, False otherwise
        """
        for fetcher in fetchers:
            print(f"  Trying {fetcher.name}...", end=" ", flush=True)
            
            result = fetcher.fetch_by_title(title)
            if result:
                identifier, abstract = result
                print("✓")
                
                results[identifier] = {
                    'title': title,
                    'abstract': abstract,
                    'source': fetcher.name
                }
                
                tracker.log_success(fetcher.name, identifier)
                return True
            else:
                print("✗")
        
        tracker.log_failure()
        return False
    
    def retrieve_by_id(self, identifier: str, fetcher_name: str = None) -> Tuple[str, str]:
        """
        Retrieve abstract by identifier (PMID, DOI, etc.).
        
        Args:
            identifier: Publication identifier
            fetcher_name: Specific fetcher to use (optional)
            
        Returns:
            Tuple of (identifier, abstract) if found
        """
        # Map fetcher names to instances
        fetcher_map = {
            'PubMed': PubMedFetcher(),
            'Europe PMC': EuropePMCFetcher(),
            # Add more as needed
        }
        
        if fetcher_name and fetcher_name in fetcher_map:
            fetcher = fetcher_map[fetcher_name]
            return fetcher.fetch_by_id(identifier)
        
        # Try all fetchers
        all_fetchers = self.phase1_fetchers + self.phase2_fetchers + self.phase3_fetchers
        for fetcher in all_fetchers:
            result = fetcher.fetch_by_id(identifier)
            if result:
                return result
        
        return None

