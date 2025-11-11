# Quick Start Guide - Transformer IR System

## 📋 Prerequisites

1. Python 3.8 or higher
2. Two JSON files:
   - `relevant_documents.json` - 1,308 polyphenol articles
   - `non_relevant_documents.json` - 1,308 non-relevant articles

---

## 🚀 Step-by-Step Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (deep learning framework)
- NumPy (numerical operations)
- scikit-learn (evaluation metrics)
- requests (API calls, if needed)

---

### Step 2: Verify Your Data (Recommended)

Before training, check that your data is properly formatted:

```bash
python inspect_data.py
```

**What to look for:**
- ✓ Total documents match (should be 1,308 each)
- ✓ No empty abstracts
- ✓ Datasets are balanced
- ✓ No overlapping document IDs

**Example Output:**
```
ANALYZING: relevant_documents.json (Relevant)
Total documents: 1308
Field Statistics:
  Documents with 'title' field: 1308 (100.0%)
  Documents with 'abstract' field: 1308 (100.0%)
  
ANALYZING: non_relevant_documents.json (Non-Relevant)
Total documents: 1308
Field Statistics:
  Documents with 'title' field: 0 (0.0%)
  Documents with 'abstract' field: 1308 (100.0%)

DATASET COMPARISON
Dataset balance ratio: 1.00
✓ Datasets are well balanced!
✓ No overlapping document IDs found.
```

---

### Step 3: Train the Model

```bash
python transformer_ir_system.py
```

**What happens:**
1. Loads 2,616 total documents (1,308 + 1,308)
2. Splits: 70% train, 15% validation, 15% test
3. Builds vocabulary from training data
4. Trains for 15 epochs (~10-20 minutes)
5. Saves best model to `transformer_ir_model.pth`

**Expected Output:**
```
Using device: cuda
Loading data...
Preparing datasets...
Dataset sizes - Train: 1831, Val: 392, Test: 393
Vocabulary size: 10002

Training model...
Epoch 1/15
  Train Loss: 0.4523
  Val Accuracy: 0.8520, F1: 0.8495
  New best model saved!
...
Epoch 15/15
  Train Loss: 0.0856
  Val Accuracy: 0.9235, F1: 0.9228

Training completed. Best validation F1: 0.9228

Evaluating on test set...
=== Test Set Results ===
Accuracy:  0.9211
Precision: 0.9156
Recall:    0.9287
F1-Score:  0.9221

Model saved to transformer_ir_model.pth
```

---

### Step 4: Test Predictions

#### Interactive Mode (Recommended for testing)
```bash
python inference.py --mode interactive
```

Then type or paste abstracts:
```
Enter text: Polyphenol composition and antioxidant activity in strawberries

============================================================
Text: Polyphenol composition and antioxidant activity in strawberries...
============================================================
Prediction: RELEVANT
Confidence: 94.23%
============================================================
```

#### Single Prediction
```bash
python inference.py --mode single --text "Your abstract here"
```

#### Batch Prediction on File
```bash
python inference.py --mode batch --input new_documents.json --output predictions.json
```

#### Evaluate on Labeled Data
```bash
# Evaluate on relevant documents
python inference.py --mode eval --input relevant_documents.json --label 1

# Evaluate on non-relevant documents
python inference.py --mode eval --input non_relevant_documents.json --label 0
```

---

## 📊 Understanding the Results

### Evaluation Metrics Explained

**Accuracy**: Overall percentage of correct predictions
- Example: 92.11% means 92.11% of documents were classified correctly

**Precision**: Of documents predicted as relevant, how many actually are?
- High precision = Few false positives (non-relevant marked as relevant)

**Recall**: Of all relevant documents, how many did we find?
- High recall = Few false negatives (relevant marked as non-relevant)

**F1-Score**: Balance between precision and recall
- Best for imbalanced datasets
- Range: 0.0 (worst) to 1.0 (perfect)

### Confusion Matrix Example
```
                Predicted
              Non-rel  Relevant
Actual Non-rel    180       16     ← 16 false positives
       Relevant    15      182     ← 15 false negatives
```

---

## 🎯 Tips for Best Results

### 1. Data Quality
- Ensure abstracts are complete (not truncated)
- Remove duplicates between datasets
- Verify balance (equal relevant/non-relevant)

### 2. Hyperparameter Tuning

Edit `transformer_ir_system.py` main() function:

```python
# Increase model capacity for better performance
ir_system = TransformerIRSystem(
    max_vocab_size=15000,  # More words (default: 10000)
    d_model=512,           # Larger embeddings (default: 256)
    nhead=8,               # Attention heads (default: 8)
    num_layers=4,          # More layers (default: 3)
    max_seq_length=512     # Longer sequences (default: 256)
)

# Adjust training parameters
ir_system.train(
    train_dataset,
    val_dataset,
    batch_size=16,         # Smaller for GPU memory (default: 32)
    epochs=20,             # More epochs (default: 15)
    lr=0.0005             # Lower learning rate (default: 0.001)
)
```

### 3. GPU vs CPU
- **GPU**: 5-15 minutes training time
- **CPU**: 20-60 minutes training time
- Model automatically detects and uses GPU if available

---

## 🐛 Troubleshooting

### Problem: "Out of Memory" Error
**Solution:**
```python
# Reduce batch size
ir_system.train(train_dataset, val_dataset, batch_size=8)

# OR reduce model size
ir_system = TransformerIRSystem(d_model=128, num_layers=2)
```

### Problem: Poor Performance (<80% accuracy)
**Possible Causes:**
1. Imbalanced dataset → Check with `inspect_data.py`
2. Poor data quality → Verify abstracts are complete
3. Too few epochs → Increase to 20-25 epochs
4. Model too small → Increase d_model and num_layers

### Problem: Training Too Slow
**Solutions:**
1. Install GPU-enabled PyTorch
2. Increase batch size (if memory allows)
3. Reduce vocabulary size or sequence length

### Problem: "File not found" Error
**Check:**
- Files named exactly `relevant_documents.json` and `non_relevant_documents.json`
- Files in same directory as scripts
- Files are valid JSON format

---

## 📁 File Structure

After setup, your directory should look like:

```
your_project/
├── transformer_ir_system.py      # Main training script
├── inference.py                   # Prediction script
├── inspect_data.py                # Data verification utility
├── requirements.txt               # Dependencies
├── README.md                      # Full documentation
├── QUICK_START.md                 # This file
├── relevant_documents.json        # Your data (1,308 docs)
├── non_relevant_documents.json    # Your data (1,308 docs)
└── transformer_ir_model.pth       # Saved model (after training)
```

---

## 🎓 For Your Report

### Things to Include:

1. **System Architecture**
   - Transformer Encoder with 3 layers
   - 8-head multi-attention mechanism
   - Positional encoding
   - Binary classification head

2. **Implementation Details**
   - Vocabulary size: 10,000 words
   - Sequence length: 256 tokens
   - Embedding dimension: 256
   - Training: Adam optimizer, LR=0.001

3. **Results**
   - Report test set metrics (accuracy, precision, recall, F1)
   - Include confusion matrix
   - Discuss any errors or misclassifications

4. **Code Availability**
   - All code provided in submission
   - Requirements.txt for reproducibility
   - Model weights saved for evaluation

---

## ❓ Need Help?

1. **Data Format Issues**: Run `inspect_data.py` to diagnose
2. **Training Problems**: Check console output for errors
3. **Prediction Issues**: Try interactive mode to test manually

---

## 🎉 Success Indicators

You'll know everything is working when:

✅ Data inspection shows balanced datasets
✅ Training completes without errors
✅ Validation F1 score > 0.85
✅ Test accuracy > 0.85
✅ Model file created (transformer_ir_model.pth)
✅ Predictions make sense on sample texts

---

## Next Steps

Once training is complete:

1. ✅ Generate predictions on test set
2. ✅ Create visualizations for report (accuracy curves, confusion matrix)
3. ✅ Write report sections on methodology and results
4. ✅ Prepare presentation slides
5. ✅ Test presentation (12 minutes max)

**Good luck with your assignment! 🚀**