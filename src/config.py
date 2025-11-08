"""
Configuration settings for the Information Retrieval project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# Input files
PUBLICATIONS_FILE = INPUT_DIR / "publications.xlsx"

# Output files
RELEVANT_ABSTRACTS_FILE = OUTPUT_DIR / "relevant_abstracts.json"
ALTERNATIVE_ABSTRACTS_FILE = OUTPUT_DIR / "alternative_abstracts.json"
FAILED_DOCUMENTS_FILE = OUTPUT_DIR / "failed_documents.json"
ALTERNATIVE_FAILED_FILE = OUTPUT_DIR / "alternative_failed.json"

# API Configuration
NCBI_API_KEY = os.getenv('NCBI_API_KEY')
SEMANTIC_SCHOLAR_API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
CORE_API_KEY = os.getenv('CORE_API_KEY')  # Optional
UNPAYWALL_EMAIL = os.getenv('UNPAYWALL_EMAIL', 'your@email.com')  # Required for Unpaywall

# API URLs
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
CROSSREF_URL = "https://api.crossref.org/works"
OPENALEX_URL = "https://api.openalex.org/works"
ARXIV_URL = "http://export.arxiv.org/api/query"
BIORXIV_URL = "https://api.biorxiv.org/details/biorxiv"
MEDRXIV_URL = "https://api.biorxiv.org/details/medrxiv"
CORE_URL = "https://api.core.ac.uk/v3/search/works"
UNPAYWALL_URL = "https://api.unpaywall.org/v2"

# Rate Limiting (seconds)
GENERAL_API_DELAY = 0.5  # For Europe PMC, CrossRef, PubMed
SEMANTIC_SCHOLAR_DELAY = 1.0  # Semantic Scholar has strict limits
NCBI_DELAY = 0.4  # NCBI rate limit delay

# Retry Configuration
MAX_RETRIES = 5  # Maximum retry attempts for 429 errors
RETRY_BACKOFF_BASE = 1  # Base multiplier for progressive delay

# API Timeouts (seconds)
API_TIMEOUT = 10

# Request Configuration
REQUEST_TIMEOUT = 10  # seconds
USER_AGENT = "InformationRetrieval/1.0 (Academic Research)"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create directories if they don't exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

