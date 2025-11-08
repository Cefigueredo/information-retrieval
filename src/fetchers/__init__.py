"""
Abstract fetchers for different data sources.
"""

from .base import AbstractFetcher
from .pubmed import PubMedFetcher
from .europe_pmc import EuropePMCFetcher
from .openalex import OpenAlexFetcher
from .crossref import CrossRefFetcher
from .arxiv import ArXivFetcher
from .core import COREFetcher
from .biorxiv import BioRxivFetcher
from .semantic_scholar import SemanticScholarFetcher

__all__ = [
    'AbstractFetcher',
    'PubMedFetcher',
    'EuropePMCFetcher',
    'OpenAlexFetcher',
    'CrossRefFetcher',
    'ArXivFetcher',
    'COREFetcher',
    'BioRxivFetcher',
    'SemanticScholarFetcher'
]

