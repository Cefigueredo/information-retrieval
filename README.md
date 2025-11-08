# Information Retrieval - Abstract Fetcher

A Python-based tool for retrieving academic abstracts from multiple APIs.

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
CORE_API_KEY=your_core_api_key_here
UNPAYWALL_EMAIL=your_email@example.com
```

**Note:** 
- NCBI API key is required for PubMed requests (get one at https://www.ncbi.nlm.nih.gov/account/settings/)
- Semantic Scholar API key is optional but highly recommended to avoid rate limits (get one at https://www.semanticscholar.org/product/api)
- CORE API key is optional but highly recommended to avoid rate limits (get one at https://core.ac.uk/services/api)
- Unpaywall email is required for Unpaywall API (get one at https://unpaywall.org/products/api)

## 📖 Usage

### ⭐ Unified Approach (Recommended)

The new unified script provides the best of both worlds:

```bash
# Use all sources (recommended)
python run.py

# Or directly:
python src/fetch_abstracts.py

# Only fast APIs (skip Semantic Scholar)
python run.py --phases 1 2

# Start fresh (ignore existing abstracts)
python run.py --fresh
```

This script uses a **three-phase approach** with **9 different APIs**:

**Phase 1**: Fast & Reliable APIs
- Europe PMC - Comprehensive biomedical database
- OpenAlex - 200M+ open access works
- CrossRef - DOI metadata
- PubMed - NCBI database
- 0.5s delay between requests

**Phase 2**: Additional Sources  
- arXiv - Physics, math, CS, biology preprints
- CORE - Open access papers
- bioRxiv - Biology preprints
- medRxiv - Medicine preprints
- 0.5s delay between requests

**Phase 3**: Rate-Limited APIs (only if needed)
- Semantic Scholar - AI-powered search
- 1s delay with automatic retry on 429 errors (max 5 retries)
- Progressive backoff: 1s, 2s, 3s, 4s, 5s

## 🔑 Features

- **9 Data Sources**: Europe PMC, OpenAlex, CrossRef, PubMed, arXiv, CORE, bioRxiv, medRxiv, Semantic Scholar
- **Authenticated API access**: Supports API keys for NCBI, Semantic Scholar, and CORE (higher rate limits)
- **Smart retry logic**: Automatic retry for rate-limited requests (429 errors)
- **Three-phase strategy**: Fast APIs first, additional sources second, rate-limited APIs last
- **Progress tracking**: Real-time progress and source statistics
- **Error handling**: Comprehensive error logging and retry mechanisms
- **Rate limiting**: Respects API rate limits automatically
- **Maximum coverage**: Searches preprints, open access, and traditional databases

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

**💡 Tip:** Using API keys (especially Semantic Scholar and CORE) significantly improves performance and reliability!


# Data Sources Guide

## Overview

This document describes all 9 data sources used by the alternative abstract retrieval system, their strengths, coverage, and when they're most useful.

## Phase 1: Fast & Reliable APIs

### 1. Europe PMC
**Website:** https://europepmc.org/  
**API Docs:** https://europepmc.org/RestfulWebService

**Coverage:**
- 40+ million life science publications
- Full-text articles and abstracts
- PubMed, PubMed Central, and more

**Strengths:**
- Fast and reliable
- Comprehensive biomedical coverage
- No API key required
- Good abstract availability

**Best for:** Biomedical, life sciences, clinical research

**Rate Limit:** ~2 requests/second (polite usage)

---

### 2. OpenAlex
**Website:** https://openalex.org/  
**API Docs:** https://docs.openalex.org/

**Coverage:**
- 200+ million scholarly works
- All disciplines
- Open access focus

**Strengths:**
- Massive coverage across all fields
- Free and open
- Fast API
- Good metadata quality
- Abstracts stored as inverted index (unique format)

**Best for:** Cross-disciplinary searches, recent publications, open access works

**Rate Limit:** Unlimited (polite usage recommended)

**Note:** Abstracts are in inverted index format and reconstructed by the script

---

### 3. CrossRef
**Website:** https://www.crossref.org/  
**API Docs:** https://api.crossref.org/swagger-ui/

**Coverage:**
- 130+ million DOIs
- All disciplines
- Publisher metadata

**Strengths:**
- Authoritative DOI metadata
- Fast API
- No key required
- Good for published articles

**Best for:** Published articles with DOIs, recent publications

**Rate Limit:** Unlimited (polite pool)

**Limitations:** Abstract availability varies by publisher

---

### 4. PubMed (E-utilities)
**Website:** https://pubmed.ncbi.nlm.nih.gov/  
**API Docs:** https://www.ncbi.nlm.nih.gov/books/NBK25501/

**Coverage:**
- 35+ million biomedical citations
- MEDLINE and life science journals
- Comprehensive abstracts

**Strengths:**
- Authoritative source
- High-quality abstracts
- Reliable
- Direct PMID access

**Best for:** Biomedical research, clinical studies, established publications

**Rate Limit:** 
- Without key: 3 requests/second
- With key: 10 requests/second

**Note:** NCBI API key required (free)

---

## Phase 2: Additional Sources

### 5. arXiv
**Website:** https://arxiv.org/  
**API Docs:** http://arxiv.org/help/api/

**Coverage:**
- 2+ million preprints
- Physics, mathematics, computer science, biology

**Strengths:**
- High-quality preprints
- Always has full abstracts
- Fast API
- No key required
- Great for cutting-edge research

**Best for:** Physics, CS, math, quantitative biology, recent/unpublished work

**Rate Limit:** ~1 request per 3 seconds

**Format:** Returns XML which the script parses

---

### 6. CORE
**Website:** https://core.ac.uk/  
**API Docs:** https://core.ac.uk/services/api

**Coverage:**
- 200+ million open access papers
- Aggregates from 10,000+ repositories
- All disciplines

**Strengths:**
- Massive open access coverage
- Good abstract availability
- Multiple format support
- API key gives higher limits

**Best for:** Open access papers, institutional repositories, grey literature

**Rate Limit:**
- Without key: Limited
- With key: 10 requests/second

**Note:** Optional API key (free) - highly recommended

---

### 7. bioRxiv
**Website:** https://www.biorxiv.org/  
**API Docs:** https://api.biorxiv.org/

**Coverage:**
- Biology preprints
- Growing rapidly (started 2013)
- Unrefereed manuscripts

**Strengths:**
- High-quality biology preprints
- Full abstracts always available
- Free API

**Best for:** Recent biology research, preprints, unpublished work

**Rate Limit:** ~1 request/second

**Limitations:** 
- API requires date ranges (searches last 2 years)
- Title matching may be imperfect
- Can be slower due to large result sets

---

### 8. medRxiv
**Website:** https://www.medrxiv.org/  
**API Docs:** https://api.biorxiv.org/ (same as bioRxiv)

**Coverage:**
- Medical and health sciences preprints
- Clinical research
- Unrefereed manuscripts

**Strengths:**
- Medical/clinical preprints
- Full abstracts
- Same API as bioRxiv

**Best for:** Clinical research, medical studies, COVID-19 research, recent work

**Rate Limit:** ~1 request/second

**Limitations:** Same as bioRxiv

---

## Phase 3: Rate-Limited APIs

### 9. Semantic Scholar
**Website:** https://www.semanticscholar.org/  
**API Docs:** https://api.semanticscholar.org/

**Coverage:**
- 200+ million papers
- All disciplines
- AI-enhanced metadata

**Strengths:**
- Intelligent search
- Cross-disciplinary
- Good abstract coverage
- API key gives massive rate increase

**Best for:** When other sources fail, cross-disciplinary work, AI/NLP papers

**Rate Limit:**
- Without key: 100 requests per 5 minutes (~1 every 3 seconds)
- With key: 5,000 requests per 5 minutes (~16/second)

**Note:** API key (free) makes this 50x faster!

---

## API Keys Priority

### Required:
1. **NCBI API Key** - PubMed won't work without it

### Highly Recommended:
1. **Semantic Scholar** - 50x rate increase (100 → 5,000 req/5min)
2. **CORE** - Better rate limits and reliability

### Optional:
1. **Unpaywall Email** - Just needs any valid email

# API Keys Guide

## Overview

This project supports API keys for enhanced performance and reliability when fetching abstracts from academic databases.

## Supported API Keys

### 1. NCBI API Key (Required)

**Purpose:** Access to PubMed database via E-utilities

**Get your key:** https://www.ncbi.nlm.nih.gov/account/settings/

**Benefits:**
- ✅ Increase rate limit from 3 to 10 requests per second
- ✅ More reliable access during peak times
- ✅ Required for production use

**How to get it:**
1. Create a free NCBI account
2. Go to Settings → API Key Management
3. Generate a new API key
4. Copy the key to your `.env` file

### 2. Semantic Scholar API Key (Optional but Recommended)

**Purpose:** Access to Semantic Scholar's academic search API

**Get your key:** https://www.semanticscholar.org/product/api

**Benefits:**
- ✅ **Massive rate limit increase:** 5,000 requests per 5 minutes (vs 100 without key)
- ✅ **~50x faster processing** for large datasets
- ✅ Virtually eliminates 429 rate limit errors
- ✅ Priority access during high-traffic periods

**How to get it:**
1. Go to https://www.semanticscholar.org/product/api
2. Click "Get API Key" or "Request API Access"
3. Fill out the form (usually instant approval for academic use)
4. Copy the key to your `.env` file

## Setup

### Step 1: Create .env File

Copy the template file:
```bash
cp env_template.txt .env
```

Or create a new `.env` file in the project root:
```bash
touch .env
```

### Step 2: Add Your Keys

Edit `.env` and add your API keys:

```bash
# NCBI API Key (Required)
NCBI_API_KEY=your_actual_ncbi_api_key_here

# Semantic Scholar API Key (Recommended)
SEMANTIC_SCHOLAR_API_KEY=your_actual_semantic_scholar_key_here
```

### Step 3: Verify

The scripts will automatically use your API keys. You'll see:
- Faster processing
- Fewer rate limit errors
- Higher success rates

## Rate Limits Comparison

### Without API Keys
| Service | Limit | Delay Needed |
|---------|-------|--------------|
| PubMed | 3 req/s | 0.4s between requests |
| Semantic Scholar | 100 req/5min | 3s between requests |

**Result:** Slow processing, frequent 429 errors

### With API Keys
| Service | Limit | Delay Needed |
|---------|-------|--------------|
| PubMed | 10 req/s | 0.4s (conservative) |
| Semantic Scholar | 5,000 req/5min | 0.5s (conservative) |

**Result:** 🚀 Much faster processing, minimal errors

## Performance Impact

### Example: Processing 1,000 Documents

**Without Semantic Scholar API Key:**
- Phase 1 (other APIs): ~8 minutes
- Phase 2 (Semantic Scholar): ~50 minutes (if 1,000 documents need it)
- **Total: ~58 minutes**

**With Semantic Scholar API Key:**
- Phase 1 (other APIs): ~8 minutes
- Phase 2 (Semantic Scholar): ~8 minutes (much shorter delay)
- **Total: ~16 minutes**

**Time saved: ~42 minutes (72% faster)** 

## Troubleshooting

### API Key Not Working

**NCBI:**
1. Check that the key is correctly copied (no extra spaces)
2. Verify the key is active in your NCBI account settings
3. Test with a simple request first

**Semantic Scholar:**
1. Ensure the key is in the correct format
2. Check if the key is activated (may take a few minutes after generation)
3. Verify you haven't exceeded the daily limit

### Still Getting 429 Errors

**With Semantic Scholar API key:**
- Very rare, but if it happens:
  - Check if your key is valid and active
  - Make sure the key is correctly set in `.env`
  - The script will automatically retry with backoff

**Without Semantic Scholar API key:**
- Expected with large datasets
- The script automatically retries
- Consider getting an API key for better performance

### Environment Variable Not Loading

1. Make sure `.env` file is in the project root directory
2. Restart your Python script/terminal
3. Check file permissions: `chmod 600 .env`
4. Verify the file format (no quotes needed around values)

## Security Best Practices

### Do's ✅
- Keep `.env` file in your local project only
- Add `.env` to `.gitignore` (already done)
- Use different keys for different projects if possible
- Regenerate keys if they're accidentally exposed

### Don'ts ❌
- Never commit `.env` to version control
- Don't share your API keys publicly
- Don't hardcode keys in source files
- Don't include keys in screenshots or documentation

## Additional Resources

### NCBI E-utilities
- Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- Rate limits: https://www.ncbi.nlm.nih.gov/books/NBK25497/
- Support: https://support.nlm.nih.gov/

### Semantic Scholar API
- Documentation: https://api.semanticscholar.org/
- API Dashboard: https://www.semanticscholar.org/product/api/dashboard
- Support: https://www.semanticscholar.org/product/api#contact

## FAQ

**Q: Are the API keys free?**
A: Yes, both NCBI and Semantic Scholar offer free API access for academic and research purposes.

**Q: How long does it take to get a Semantic Scholar API key?**
A: Usually instant for academic email addresses. May take 1-2 business days for review otherwise.

**Q: Can I use the script without any API keys?**
A: You need an NCBI API key for PubMed access. Semantic Scholar key is optional but highly recommended.

**Q: Will my API key work if I share my code?**
A: The `.env` file is git-ignored, so your keys won't be shared. Each user needs their own keys.

**Q: What happens if I hit the rate limit?**
A: The script automatically retries with progressive backoff (3s, 6s, 9s, etc.). With API keys, this is very rare.

**Q: Can I increase the rate limits further?**
A: NCBI limits are fixed. For Semantic Scholar, contact them for enterprise/higher limits if needed.

## Summary

Getting API keys takes just a few minutes and provides:
- ✅ **Faster processing** (up to 50x for Semantic Scholar)
- ✅ **Higher reliability** (fewer errors)
- ✅ **Better results** (fewer timeouts)
- ✅ **Free for academic use**

**Recommendation:** Set up both API keys before processing large datasets!

