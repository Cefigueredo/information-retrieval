"""
Data Inspection Utility
Verify and inspect the JSON data files before training
"""

import json
import re
from collections import Counter


def extract_text_from_pubmed_abstract(abstract: str) -> str:
    """Extract relevant text from PubMed formatted abstract."""
    if not abstract:
        return ""
    
    lines = abstract.split('\n')
    
    # Find title
    title = ""
    main_text = []
    
    start_idx = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if i < 10 and line and not line.startswith('Author') and len(line) > 20:
            if '.' in line and not line.startswith('DOI:') and not line.startswith('PMID:'):
                title = line
                start_idx = i + 1
                break
    
    # Extract main abstract text
    in_abstract = False
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        
        if line.startswith('Author information:'):
            continue
        
        if any(line.startswith(keyword) for keyword in 
               ['BACKGROUND:', 'INTRODUCTION:', 'OBJECTIVE:', 'METHODS:', 'METHODOLOGY:',
                'RESULTS:', 'CONCLUSION:', 'PURPOSE:', 'AIM:']):
            in_abstract = True
        
        if any(line.startswith(meta) for meta in 
               ['Copyright', 'DOI:', 'PMCID:', 'PMID:', '©']):
            break
        
        if in_abstract and line:
            main_text.append(line)
    
    combined = f"{title} {' '.join(main_text)}"
    return combined if combined.strip() else abstract


def clean_text(text: str) -> str:
    """Clean text for word counting."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def analyze_json_file(filepath: str, file_type: str):
    """Analyze a JSON data file."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {filepath} ({file_type})")
    print(f"{'='*70}\n")
    
    # Load data
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total documents: {len(data)}")
    
    # Analyze structure
    has_title = 0
    has_abstract = 0
    empty_abstract = 0
    word_counts = []
    all_words = []
    
    # Sample documents
    sample_ids = list(data.keys())[:3]
    
    for doc_id, doc in data.items():
        # Check fields
        if 'title' in doc and doc['title']:
            has_title += 1
        
        if 'abstract' in doc:
            has_abstract += 1
            abstract = doc['abstract']
            
            if not abstract or len(abstract.strip()) == 0:
                empty_abstract += 1
            else:
                # Extract text based on format
                if 'title' in doc and doc['title']:
                    text = f"{doc['title']} {abstract}"
                else:
                    text = extract_text_from_pubmed_abstract(abstract)
                
                # Count words
                cleaned = clean_text(text)
                words = cleaned.split()
                word_counts.append(len(words))
                all_words.extend(words)
    
    print(f"\nField Statistics:")
    print(f"  Documents with 'title' field: {has_title} ({has_title/len(data)*100:.1f}%)")
    print(f"  Documents with 'abstract' field: {has_abstract} ({has_abstract/len(data)*100:.1f}%)")
    print(f"  Empty abstracts: {empty_abstract}")
    
    if word_counts:
        print(f"\nText Length Statistics:")
        print(f"  Average words per document: {sum(word_counts)/len(word_counts):.1f}")
        print(f"  Min words: {min(word_counts)}")
        print(f"  Max words: {max(word_counts)}")
        print(f"  Median words: {sorted(word_counts)[len(word_counts)//2]}")
    
    # Most common words
    if all_words:
        word_freq = Counter(all_words)
        print(f"\nTop 15 Most Common Words:")
        for word, count in word_freq.most_common(15):
            print(f"  {word}: {count}")
    
    # Show sample documents
    print(f"\n{'='*70}")
    print("SAMPLE DOCUMENTS:")
    print(f"{'='*70}")
    
    for i, doc_id in enumerate(sample_ids, 1):
        doc = data[doc_id]
        print(f"\nSample {i} (ID: {doc_id}):")
        print(f"-" * 70)
        
        if 'title' in doc and doc['title']:
            print(f"Title: {doc['title'][:100]}...")
            print(f"Abstract: {doc['abstract'][:200]}...")
        else:
            abstract = doc.get('abstract', '')
            print(f"Abstract (PubMed format):")
            print(f"{abstract[:500]}...")
            
            # Show extracted text
            extracted = extract_text_from_pubmed_abstract(abstract)
            print(f"\nExtracted text preview:")
            print(f"{extracted[:300]}...")


def compare_datasets(relevant_path: str, non_relevant_path: str):
    """Compare relevant and non-relevant datasets."""
    print(f"\n{'='*70}")
    print("DATASET COMPARISON")
    print(f"{'='*70}\n")
    
    # Load both datasets
    with open(relevant_path, 'r', encoding='utf-8') as f:
        relevant_data = json.load(f)
    
    with open(non_relevant_path, 'r', encoding='utf-8') as f:
        non_relevant_data = json.load(f)
    
    print(f"Relevant documents: {len(relevant_data)}")
    print(f"Non-relevant documents: {len(non_relevant_data)}")
    print(f"Total documents: {len(relevant_data) + len(non_relevant_data)}")
    
    # Check balance
    ratio = len(relevant_data) / len(non_relevant_data) if len(non_relevant_data) > 0 else 0
    print(f"\nDataset balance ratio: {ratio:.2f}")
    
    if abs(ratio - 1.0) < 0.1:
        print("✓ Datasets are well balanced!")
    else:
        print("⚠ Warning: Datasets are imbalanced. Consider balancing them.")
    
    # Check for potential overlap (by ID)
    relevant_ids = set(relevant_data.keys())
    non_relevant_ids = set(non_relevant_data.keys())
    overlap = relevant_ids.intersection(non_relevant_ids)
    
    if overlap:
        print(f"\n⚠ Warning: {len(overlap)} document IDs appear in both datasets!")
        print(f"Overlapping IDs: {list(overlap)[:5]}...")
    else:
        print("\n✓ No overlapping document IDs found.")


def main():
    """Main inspection function."""
    print("\n" + "="*70)
    print("DATA INSPECTION UTILITY")
    print("="*70)
    
    # Analyze relevant documents
    try:
        analyze_json_file('data/output/alternative_abstracts.json', 'Relevant')
    except FileNotFoundError:
        print("\n⚠ Error: 'data/output/alternative_abstracts.json' not found!")
    except Exception as e:
        print(f"\n⚠ Error analyzing relevant documents: {e}")
    
    # Analyze non-relevant documents
    try:
        analyze_json_file('data/output/non_relevant_abstracts.json', 'Non-Relevant')
    except FileNotFoundError:
        print("\n⚠ Error: 'data/output/non_relevant_abstracts.json' not found!")
    except Exception as e:
        print(f"\n⚠ Error analyzing non-relevant documents: {e}")
    
    # Compare datasets
    try:
        compare_datasets('data/output/alternative_abstracts.json', 'data/output/non_relevant_abstracts.json')
    except FileNotFoundError:
        print("\n⚠ Error: One or both data files not found!")
    except Exception as e:
        print(f"\n⚠ Error comparing datasets: {e}")
    
    print(f"\n{'='*70}")
    print("INSPECTION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()