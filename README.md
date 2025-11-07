# Information Retrieval - Abstract Fetcher

A Python-based tool for retrieving academic abstracts from multiple APIs including PubMed, Europe PMC, Semantic Scholar, and CrossRef.

## 📁 Project Structure

```
information-retrieval/
├── src/                          # Source code
│   ├── main_request.py          # Main script for fetching abstracts from NCBI
│   ├── alternative_request.py   # Alternative sources with retry logic
│   └── config.py                # Configuration settings
├── data/                         # Data directory
│   ├── input/                   # Input files
│   │   └── publications.xlsx    # Source publications list
│   └── output/                  # Generated output files
│       ├── relevant_abstracts.json
│       ├── alternative_abstracts.json
│       └── failed_documents.json
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
├── .env                         # Environment variables (API keys)
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd information-retrieval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```bash
NCBI_API_KEY=your_ncbi_api_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
```

**Note:** 
- NCBI API key is required for PubMed requests (get one at https://www.ncbi.nlm.nih.gov/account/settings/)
- Semantic Scholar API key is optional but highly recommended to avoid rate limits (get one at https://www.semanticscholar.org/product/api)

## 📖 Usage

### Main Request (NCBI PubMed)

Fetch abstracts using NCBI's E-utilities API:

```bash
python src/main_request.py
```

This script:
- Reads publications from `data/input/publications.xlsx`
- Searches for PMIDs by title
- Fetches abstracts from PubMed
- Saves results to `data/output/relevant_abstracts.json`
- Logs failures to `data/output/failed_documents.json`

### Alternative Request (Multi-API)

Fetch missing abstracts using alternative APIs:

```bash
python src/alternative_request.py
```

This script uses a **two-phase approach**:

**Phase 1**: Fast APIs (Europe PMC, CrossRef, PubMed)
- Tries multiple sources quickly
- 0.5s delay between requests

**Phase 2**: Semantic Scholar (only if needed)
- Used for remaining documents
- 3s delay with automatic retry on 429 errors (max 5 retries)
- Progressive backoff: 3s, 6s, 9s, 12s, 15s

## 🔑 Features

- **Multi-source retrieval**: Europe PMC, Semantic Scholar, CrossRef, PubMed
- **Authenticated API access**: Supports API keys for NCBI and Semantic Scholar (higher rate limits)
- **Smart retry logic**: Automatic retry for rate-limited requests (429 errors)
- **Two-phase strategy**: Fast APIs first, then slower APIs only when needed
- **Progress tracking**: Real-time progress and source statistics
- **Error handling**: Comprehensive error logging and retry mechanisms
- **Rate limiting**: Respects API rate limits automatically

## 📊 Output Files

### `relevant_abstracts.json`
Main abstracts from PubMed:
```json
{
  "12345678": {
    "title": "Publication Title",
    "abstract": "Abstract text..."
  }
}
```

### `alternative_abstracts.json`
Abstracts from alternative sources:
```json
{
  "12345678": {
    "title": "Publication Title",
    "abstract": "Abstract text...",
    "source": "Europe PMC"
  }
}
```

### `failed_documents.json`
Documents that couldn't be retrieved:
```json
{
  "total_failed": 5,
  "failed_documents": [
    {
      "index": 10,
      "title": "Publication Title",
      "error": "Abstract not found in any source"
    }
  ]
}
```

## 🔧 Configuration

Edit `src/config.py` to customize:
- API URLs
- Rate limiting delays
- Retry attempts
- File paths

## 📝 API Rate Limits

- **PubMed**: 3 requests/second (without API key), 10 requests/second (with API key)
- **Europe PMC**: No strict limit, but recommended 1-2 requests/second
- **Semantic Scholar**: 
  - Without API key: 100 requests per 5 minutes (~1 request per 3 seconds)
  - With API key: 5,000 requests per 5 minutes (~16 requests/second)
- **CrossRef**: No strict limit (polite pool)

**💡 Tip:** Using API keys significantly improves performance and reliability!

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- NCBI for PubMed E-utilities API
- Europe PMC for their comprehensive literature database
- Semantic Scholar for their AI-powered search
- CrossRef for DOI resolution services

