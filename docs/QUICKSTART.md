# Quick Start Guide

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up your API keys:**
Create a `.env` file in the project root:
```bash
NCBI_API_KEY=your_ncbi_api_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
```

**Where to get API keys:**
- **NCBI API Key** (required for PubMed): https://www.ncbi.nlm.nih.gov/account/settings/
- **Semantic Scholar API Key** (optional, but recommended): https://www.semanticscholar.org/product/api

**Note:** The Semantic Scholar API key is optional, but highly recommended. With an API key, you get:
- 5,000 requests per 5 minutes (vs 100 without key)
- Much faster processing
- Fewer rate limit errors

3. **Prepare your data:**
Place your publications Excel file in `data/input/publications.xlsx`

## Running the Scripts

### Option 1: Main Request (NCBI PubMed)

This fetches abstracts directly from NCBI PubMed:

```bash
cd information-retrieval
python src/main_request.py
```

**Output:**
- `data/output/relevant_abstracts.json` - Successfully retrieved abstracts
- `data/output/failed_documents.json` - Failed retrievals

### Option 2: Alternative Request (Multi-API)

This searches multiple sources for missing abstracts:

```bash
python src/alternative_request.py
```

**Output:**
- `data/output/alternative_abstracts.json` - Abstracts from alternative sources
- `data/output/alternative_failed.json` - Documents still not found

## Workflow

**Recommended workflow for best results:**

1. Run `main_request.py` first to get abstracts from PubMed
2. Check `failed_documents.json` to see what's missing
3. Run `alternative_request.py` to find missing abstracts from other sources
4. Combine results as needed

## Configuration

Edit `src/config.py` to customize:
- API rate limits
- File paths
- Retry behavior
- Timeout values

## Troubleshooting

### Error: 429 Too Many Requests
- The script automatically retries with backoff
- **Best solution**: Add a Semantic Scholar API key to your `.env` file (increases limit from 100 to 5,000 requests per 5 minutes)
- If persistent without API key, increase delays in `config.py`:
  - `GENERAL_API_DELAY` for Europe PMC, CrossRef, PubMed
  - `SEMANTIC_SCHOLAR_DELAY` for Semantic Scholar

### Error: File not found
- Ensure `data/input/publications.xlsx` exists
- Check that your Excel file has a `title` column

### Error: No NCBI API key
- Create a `.env` file with your API key
- Get a free key at: https://www.ncbi.nlm.nih.gov/account/settings/

## Output Format

### `relevant_abstracts.json` and `alternative_abstracts.json`
```json
{
  "PMID": {
    "title": "Paper Title",
    "abstract": "Full abstract text...",
    "source": "Europe PMC"  // Only in alternative_abstracts.json
  }
}
```

### `failed_documents.json` and `alternative_failed.json`
```json
{
  "total_failed": 10,
  "total_processed": 100,
  "success_rate": "90.00%",
  "failed_documents": [
    {
      "row_number": 5,
      "title": "Paper Title",
      "pmid": "12345678",
      "error": "Abstract not found"
    }
  ]
}
```

## Tips

- **NCBI API Key**: Significantly increases rate limits (10 req/s vs 3 req/s)
- **Large datasets**: The script automatically handles rate limiting
- **Resume capability**: Failed documents are logged so you can retry specific ones
- **Progress tracking**: Real-time progress updates in console

## Support

For issues or questions, check:
- README.md for detailed documentation
- config.py for configuration options
- Console output for detailed error messages

