"""
Utility functions for the Information Retrieval project.
"""

from .data_handler import (
    load_publications,
    load_existing_abstracts,
    save_abstracts,
    save_failed_documents,
    get_missing_publications
)

from .progress import ProgressTracker

__all__ = [
    'load_publications',
    'load_existing_abstracts',
    'save_abstracts',
    'save_failed_documents',
    'get_missing_publications',
    'ProgressTracker'
]

