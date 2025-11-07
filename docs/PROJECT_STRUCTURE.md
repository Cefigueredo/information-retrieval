# Project Structure Documentation

## Overview

This document describes the organized structure of the Information Retrieval project.

## Directory Structure

```
information-retrieval/
├── README.md                    # Main documentation
├── .gitignore                   # Git ignore rules
├── .env                         # Environment variables (not in git)
├── env_template.txt             # Template for .env file
├── requirements.txt             # Python dependencies
├── run.py                       # Simple runner script
│
├── src/                         # Source code
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Centralized configuration
│   ├── main_request.py         # NCBI PubMed fetcher
│   └── alternative_request.py  # Multi-API fetcher
│
├── data/                        # Data directory
│   ├── input/                  # Input files
│   │   ├── .gitkeep           # Keep directory in git
│   │   └── publications.xlsx   # Source publications
│   │
│   └── output/                 # Generated output files
│       ├── .gitkeep           # Keep directory in git
│       ├── relevant_abstracts.json
│       ├── alternative_abstracts.json
│       ├── failed_documents.json
│       └── alternative_failed.json
│
└── docs/                        # Documentation
    ├── .gitkeep                # Keep directory in git
    ├── QUICKSTART.md           # Quick start guide
    └── PROJECT_STRUCTURE.md    # This file
```

## Key Improvements

### 1. **Centralized Configuration** (`src/config.py`)
- All settings in one place
- Environment variable management
- Path management with pathlib
- API URLs and timeouts
- Rate limiting configuration
- Easy to modify without touching core code

### 2. **Organized File Structure**
- **src/**: All Python code
- **data/input/**: Source files
- **data/output/**: Generated files (git-ignored)
- **docs/**: Documentation

### 3. **Better Dependency Management**
- Imports use config module
- Centralized API configuration
- Consistent timeout and retry settings

### 4. **Git Integration**
- `.gitignore` prevents committing:
  - Output JSON files
  - Environment variables
  - Python cache files
  - IDE settings
- `.gitkeep` preserves empty directories

### 5. **Easy Execution**
```bash
# Simple way to run scripts
python run.py main              # Run main_request.py
python run.py alternative       # Run alternative_request.py

# Or run directly
python src/main_request.py
python src/alternative_request.py
```

## Configuration System

### Environment Variables (`.env`)
```bash
NCBI_API_KEY=your_ncbi_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key_here
```

**API Keys:**
- **NCBI**: Required for PubMed access (10 req/s vs 3 req/s without key)
- **Semantic Scholar**: Optional but recommended (5,000 vs 100 requests per 5 minutes)

### Config Module (`src/config.py`)
All configuration in one place:
- File paths (automatically resolved)
- API URLs
- Rate limits
- Retry settings
- Timeouts

**Benefits:**
- Change settings without modifying scripts
- Easy to adapt for different environments
- Type-safe path handling with pathlib

## File Purposes

### Core Scripts

| File | Purpose |
|------|---------|
| `src/main_request.py` | Fetch abstracts from NCBI PubMed |
| `src/alternative_request.py` | Fetch from Europe PMC, Semantic Scholar, CrossRef |
| `src/config.py` | Centralized configuration |
| `run.py` | Simple runner interface |

### Data Files

| File | Purpose |
|------|---------|
| `data/input/publications.xlsx` | Source publications list |
| `data/output/relevant_abstracts.json` | PubMed abstracts |
| `data/output/alternative_abstracts.json` | Alternative source abstracts |
| `data/output/failed_documents.json` | Failed PubMed retrievals |
| `data/output/alternative_failed.json` | Failed alternative retrievals |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation |
| `docs/QUICKSTART.md` | Quick start guide |
| `docs/PROJECT_STRUCTURE.md` | This file - structure documentation |

## Migration from Old Structure

### Before
```
information-retrieval/
├── alternative_request.py       # Mixed with data
├── main_request.py             # Mixed with data
├── publications.xlsx           # Mixed with code
├── relevant_abstracts.json     # Mixed with code
├── failed_documents.json       # Mixed with code
└── requirements.txt
```

### After
```
information-retrieval/
├── src/                        # Clean separation
│   ├── alternative_request.py
│   ├── main_request.py
│   └── config.py
├── data/
│   ├── input/
│   │   └── publications.xlsx
│   └── output/
│       ├── relevant_abstracts.json
│       └── failed_documents.json
├── docs/
└── requirements.txt
```

## Best Practices Implemented

1. **Separation of Concerns**
   - Code in `src/`
   - Data in `data/`
   - Docs in `docs/`

2. **Configuration Management**
   - Centralized in `config.py`
   - Environment variables for secrets
   - Path management with pathlib

3. **Documentation**
   - README for overview
   - QUICKSTART for getting started
   - PROJECT_STRUCTURE for understanding

4. **Git Hygiene**
   - `.gitignore` for generated files
   - `.gitkeep` for directory structure
   - `.env` for secrets (not committed)

5. **Easy Execution**
   - `run.py` for simple interface
   - Direct script execution still works
   - Clear command examples

## Development Workflow

1. **Setup:**
   ```bash
   pip install -r requirements.txt
   cp .env.example .env  # Create .env with your keys
   ```

2. **Place your data:**
   ```bash
   # Put your Excel file here:
   data/input/publications.xlsx
   ```

3. **Run scripts:**
   ```bash
   python run.py main
   python run.py alternative
   ```

4. **Get results:**
   ```bash
   # Check output:
   data/output/relevant_abstracts.json
   data/output/alternative_abstracts.json
   ```

## Extending the Project

### Adding New API Sources
1. Add API URL to `config.py`
2. Create fetch function in `alternative_request.py`
3. Add to search pipeline

### Adding New Configuration
1. Add to `config.py`
2. Import in scripts
3. Use throughout codebase

### Adding Documentation
1. Create `.md` file in `docs/`
2. Reference in README.md

## Version Control

### What's Tracked
- Source code (`src/`)
- Documentation (`docs/`, `README.md`)
- Configuration template
- Requirements (`requirements.txt`)
- Directory structure (`.gitkeep` files)

### What's Ignored
- Output data files (`data/output/*.json`)
- Environment variables (`.env`)
- Python cache (`__pycache__/`)
- IDE settings (`.vscode/`, `.idea/`)

## Summary

The reorganized structure provides:
- ✅ Clear separation of code, data, and docs
- ✅ Centralized configuration
- ✅ Easy to navigate and understand
- ✅ Professional project layout
- ✅ Git-friendly structure
- ✅ Easy to extend and maintain
- ✅ Simple execution with `run.py`

