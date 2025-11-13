# Information Retrieval - Abstract Fetcher

A Python tool for retrieving academic abstracts from **9 different APIs** including PubMed, OpenAlex, arXiv, Semantic Scholar, and more.

## 🚀 Quick Start

### Installation
```bash
git clone <repository-url>
cd information-retrieval
pip install -r requirements.txt
```

### Setup API Keys
Copy `env_template.txt` to `.env` and add your keys:
```bash
cp env_template.txt .env
```

**Required:** NCBI API key ([get one here](https://www.ncbi.nlm.nih.gov/account/settings/))  
**Highly Recommended:** Semantic Scholar key ([get one here](https://www.semanticscholar.org/product/api)) - 50x faster!

### Run
```bash
python run.py
```

## 📖 Usage

### Basic Commands
```bash
# All sources (recommended)
python run.py

# Only fast APIs (skip Semantic Scholar)
python run.py --phases 1 2

# Start fresh (ignore existing abstracts)
python run.py --fresh

# Custom input/output
python src/fetch_abstracts.py --input my_pubs.xlsx --output results.json
```

### Input/Output
- **Input:** `data/input/publications.xlsx` (must have 'title' column)
- **Output:** `data/output/alternative_abstracts.json`
- **Failed:** `data/output/alternative_failed.json`

## 🎯 Features

- **9 Data Sources** - Comprehensive coverage across disciplines
- **Smart 3-Phase Search** - Fast APIs first, specialized sources second, rate-limited last
- **Auto Retry** - Handles rate limits automatically (up to 5 retries with progressive backoff)
- **Progress Tracking** - Real-time updates and source statistics
- **High Success Rate** - 85-95% coverage across all sources
- **Modular Architecture** - Easy to extend with new sources

## 🔍 Data Sources

### Phase 1: Fast & Reliable APIs
1. **Europe PMC** - 40M+ biomedical papers, fast and reliable, no key needed
2. **OpenAlex** - 200M+ open access works, all disciplines, unlimited rate
3. **CrossRef** - 130M+ DOIs, publisher metadata, polite pool
4. **PubMed** - 35M+ biomedical citations, requires NCBI API key

### Phase 2: Additional Sources
5. **arXiv** - 2M+ preprints (physics, CS, math, biology), always has abstracts
6. **CORE** - 200M+ open access papers, optional API key recommended
7. **bioRxiv** - Biology preprints, free API
8. **medRxiv** - Medical preprints, free API

### Phase 3: Rate-Limited APIs
9. **Semantic Scholar** - 200M+ papers, AI-powered search, API key highly recommended (50x faster)

## 🔑 API Keys

| Key | Required? | Rate Limit | Get It |
|-----|-----------|------------|--------|
| **NCBI** | ✅ Required | 3→10 req/s | [Link](https://www.ncbi.nlm.nih.gov/account/settings/) |
| **Semantic Scholar** | 🌟 Highly Recommended | 100→5,000 req/5min | [Link](https://www.semanticscholar.org/product/api) |
| **CORE** | Optional | Limited→10 req/s | [Link](https://core.ac.uk/services/api) |
| **Unpaywall Email** | Optional | Any valid email | [Link](https://unpaywall.org/products/api) |

### Why Get API Keys?

**Without Semantic Scholar API Key:**
- 100 requests per 5 minutes (~1 every 3 seconds)
- Processing 1,000 documents: ~58 minutes

**With Semantic Scholar API Key:**
- 5,000 requests per 5 minutes (~16/second)
- Processing 1,000 documents: ~16 minutes
- **Time saved: ~72% faster!**

### Setup Steps

1. **NCBI API Key** (Required):
   - Create free account at https://www.ncbi.nlm.nih.gov/account/settings/
   - Generate API key in Settings → API Key Management
   - Add to `.env`: `NCBI_API_KEY=your_key_here`

2. **Semantic Scholar API Key** (Recommended):
   - Go to https://www.semanticscholar.org/product/api
   - Request API access (usually instant for academic emails)
   - Add to `.env`: `SEMANTIC_SCHOLAR_API_KEY=your_key_here`

3. **CORE API Key** (Optional):
   - Register at https://core.ac.uk/services/api
   - Add to `.env`: `CORE_API_KEY=your_key_here`

4. **Unpaywall Email** (Optional):
   - Just needs any valid email
   - Add to `.env`: `UNPAYWALL_EMAIL=your@email.com`

## 📊 Output Format

### Successful Abstracts
```json
{
  "12345678": {
    "title": "Publication Title",
    "abstract": "Abstract text...",
    "source": "Europe PMC"
  }
}
```

### Failed Documents
```json
{
  "total_failed": 5,
  "total_processed": 100,
  "success_rate": "95.00%",
  "failed_documents": [
    {
      "index": 10,
      "title": "Publication Title",
      "error": "Abstract not found in any source"
    }
  ]
}
```

## ⚙️ Configuration

Edit `src/config.py` to customize:
- API URLs and timeouts
- Rate limiting delays (default: 0.5s general, 1.0s Semantic Scholar)
- Retry attempts (default: 5 retries)
- File paths
- Backoff multipliers

### Adding New Sources

1. Create a new fetcher class in `src/fetchers/`:
```python
from .base import AbstractFetcher

class MyNewSourceFetcher(AbstractFetcher):
    def __init__(self):
        super().__init__(name="MyNewSource", delay=0.5)
    
    def fetch_by_title(self, title: str):
        # Implementation
        return (identifier, abstract)
```

2. Add to `abstract_retriever.py` in appropriate phase
3. Done!

### Verifying Your Data

Before training, use the inspection utility to verify your data:

```bash
python inspect_data.py
```

This will show you:
- Number of documents in each file
- Document structure and fields
- Text length statistics
- Sample documents
- Dataset balance
- Most common words

---

## Usage

### Basic Training and Evaluation

```bash
python transformer_encoder.py
```

This will:
1. Load data from `relevant_documents.json` and `non_relevant_documents.json`
2. Split into train (70%), validation (15%), test (15%)
3. Build vocabulary from training data
4. Check if there is any model already trained or train the Transformer Encoder for 15 epochs
5. Evaluate on the test set
6. Save the trained model to `transformer_ir_model.pth`

### Inference utilities

If you want to check the predictions with any custom text that you want to input to the model, you can do:

```bash
python inference.py
```

This way you can check the confidence on any text input that you use and check if the predictions are okay!

### Evaluation Visualized

If you want to check the evaluation metrics in a visual way, just run:

```bash
python visualizations.py
```
All the images containing the different evaluations will be stored in a folder called figures.

