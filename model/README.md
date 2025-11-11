# Biomedical Information Retrieval System - Transformer Encoder Implementation

## Task 4 - Assignment 1: Information Retrieval System

This implementation uses a **Transformer Encoder** architecture for binary classification of scientific abstracts related to polyphenol composition research.

---

## System Architecture

### Overview
The system consists of:
1. **Text Preprocessor**: Cleans, tokenizes, and builds vocabulary from abstracts
2. **Transformer Encoder**: Multi-head self-attention mechanism for sequence encoding
3. **Binary Classifier**: Determines relevance to polyphenol research

### Model Components

#### 1. Embedding Layer
- Converts word indices to dense vectors (d_model=256)
- Includes padding token handling

#### 2. Positional Encoding
- Adds position information using sine/cosine functions
- Enables the model to understand word order

#### 3. Transformer Encoder
- **Number of layers**: 3
- **Attention heads**: 8
- **Feed-forward dimension**: 512
- **Dropout**: 0.1

#### 4. Classification Head
- Global average pooling over sequence
- Two-layer neural network (256 → 128 → 2)
- Outputs binary classification (relevant/non-relevant)

---

## Installation

### Requirements
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)

### Setup Steps

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Data Format

The system expects two JSON files with different structures:

### Relevant Documents (relevant_documents.json)
Documents with separate title and abstract fields:

```json
{
  "10552788": {
    "title": "HPLC method for the quantification of procyanidins...",
    "abstract": "Monomeric and oligomeric procyanidins present in cocoa...",
    "source": "Europe PMC"
  }
}
```

### Non-Relevant Documents (non_relevant_documents.json)
Documents in PubMed format (title embedded in abstract):

```json
{
  "41169831": {
    "abstract": "1. Niger Med J. 2025 Sep 19;66(3):1046-1054...\n\nMagnetic Resonance Neuroimaging Findings in High-altitude Cerebral Edema...\n\nWani AH(1), Wani M(1)...\n\nBACKGROUND: High-altitude illness..."
  }
}
```

**Important**: 
- Relevant documents have separate `title` and `abstract` fields
- Non-relevant documents only have the `abstract` field with PubMed-formatted text
- The system automatically detects and handles both formats
- For PubMed format, the system extracts the title and main content automatically

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
python transformer_ir_system.py
```

This will:
1. Load data from `relevant_documents.json` and `non_relevant_documents.json`
2. Split into train (70%), validation (15%), test (15%)
3. Build vocabulary from training data
4. Train the Transformer Encoder for 15 epochs
5. Evaluate on the test set
6. Save the trained model to `transformer_ir_model.pth`

### Custom Usage

You can modify the main script to adjust hyperparameters:

```python
ir_system = TransformerIRSystem(
    max_vocab_size=10000,    # Maximum vocabulary size
    d_model=256,             # Embedding dimension
    nhead=8,                 # Number of attention heads
    num_layers=3,            # Number of transformer layers
    max_seq_length=256       # Maximum sequence length
)

# Adjust training parameters
ir_system.train(
    train_dataset,
    val_dataset,
    batch_size=32,
    epochs=15,
    lr=0.001
)
```

### Prediction Example

```python
from transformer_ir_system import TransformerIRSystem

# Load trained model
ir_system = TransformerIRSystem()
ir_system.load_model('transformer_ir_model.pth')

# Predict on new text
text = "Polyphenol composition and antioxidant activity in strawberries"
prediction, confidence = ir_system.predict(text)

if prediction == 1:
    print(f"RELEVANT (confidence: {confidence:.2%})")
else:
    print(f"NON-RELEVANT (confidence: {confidence:.2%})")
```

---

## Technical Details

### Data Preprocessing
1. **Text cleaning**: Lowercase, remove special characters, normalize whitespace
2. **Tokenization**: Simple whitespace-based splitting
3. **Vocabulary**: Built from training data (top 10,000 most frequent words)
4. **Encoding**: Words → indices, with padding to fixed length

### Training Strategy
- **Optimizer**: Adam with learning rate 0.001
- **Learning rate scheduler**: ReduceLROnPlateau (patience=2, factor=0.5)
- **Loss function**: Cross-entropy loss
- **Early stopping**: Model with best validation F1 score is saved
- **Data split**: 70% train, 15% validation, 15% test (stratified)

### Evaluation Metrics
The system reports:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

---

## Output Files

After training, the following files are generated:

1. **transformer_ir_model.pth**: Saved model checkpoint containing:
   - Model weights
   - Vocabulary dictionary
   - Model configuration

---

## Performance Expectations

With proper dataset balance (1,308 relevant + 1,308 non-relevant documents):
- Expected accuracy: 85-95%
- Training time: ~5-15 minutes on GPU, ~20-60 minutes on CPU
- Memory usage: ~2-4 GB RAM

---

## Troubleshooting

### Common Issues

**1. Out of Memory Error**
```python
# Reduce batch size
ir_system.train(train_dataset, val_dataset, batch_size=16)

# Or reduce model size
ir_system = TransformerIRSystem(d_model=128, num_layers=2)
```

**2. Poor Performance**
- Ensure balanced dataset (equal relevant/non-relevant)
- Check data quality (abstracts should be complete)
- Increase training epochs
- Try different hyperparameters

**3. Slow Training**
- Install CUDA-enabled PyTorch for GPU acceleration
- Increase batch size if memory allows
- Reduce vocabulary size or sequence length

---

## Model Architecture Diagram

```
Input Text → Preprocessing → Tokenization → Embedding (256d)
                                               ↓
                                    Positional Encoding
                                               ↓
                    ┌──────────────────────────────────────┐
                    │   Transformer Encoder (3 layers)     │
                    │   - Multi-Head Attention (8 heads)   │
                    │   - Feed-Forward Network              │
                    │   - Layer Normalization               │
                    └──────────────────────────────────────┘
                                               ↓
                                    Global Average Pooling
                                               ↓
                                    Classification Head
                                         (256→128→2)
                                               ↓
                                    Output: [Relevant, Non-Relevant]
```

---

## References

### Transformer Architecture
- Vaswani et al. (2017). "Attention Is All You Need"
- Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"

### Biomedical Text Classification
- Lee et al. (2020). "BioBERT: a pre-trained biomedical language representation model"
- Beltagy et al. (2019). "SciBERT: A Pretrained Language Model for Scientific Text"

---

## Contact & Support

For questions or issues with this implementation, please contact the course instructor:
- Prof. Miguel García-Remesal: mgremesal@fi.upm.es

---

## License

This code is provided for educational purposes as part of the BMI course assignment at Universidad Politécnica de Madrid (UPM).