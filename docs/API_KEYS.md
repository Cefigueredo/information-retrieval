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

