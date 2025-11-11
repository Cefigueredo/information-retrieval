"""
Biomedical Information Retrieval System using Transformer Encoder
Task 4 - Assignment 1
Author: Student Implementation
Date: November 2025

This system uses a Transformer Encoder architecture to classify scientific abstracts
as relevant or non-relevant to polyphenol composition research.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import Dict, List, Tuple
import math
import re


class TextPreprocessor:
    """Handles text preprocessing and tokenization."""
    
    def __init__(self, max_vocab_size: int = 10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def build_vocabulary(self, texts: List[str]):
        """Build vocabulary from training texts."""
        word_freq = {}
        for text in texts:
            cleaned = self.clean_text(text)
            for word in cleaned.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for word, _ in sorted_words[:self.max_vocab_size - 2]:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text: str, max_length: int = 256) -> List[int]:
        """Convert text to sequence of indices."""
        cleaned = self.clean_text(text)
        words = cleaned.split()[:max_length]
        indices = [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>
        
        # Pad to max_length
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        
        return indices


class PolyphenolDataset(Dataset):
    """Dataset class for polyphenol abstracts."""
    
    def __init__(self, data: List[Tuple[str, int]], preprocessor: TextPreprocessor, max_length: int = 256):
        self.data = data
        self.preprocessor = preprocessor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoded = self.preprocessor.encode(text, self.max_length)
        return torch.LongTensor(encoded), torch.LongTensor([label])


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:x.size(1)]


class TransformerEncoder(nn.Module):
    """Transformer Encoder for binary text classification."""
    
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 3, dim_feedforward: int = 512, 
                 dropout: float = 0.1, max_seq_length: int = 256):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Binary classification
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """Forward pass."""
        # Create padding mask (True for padding tokens)
        padding_mask = (x == 0)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Global average pooling (excluding padding)
        mask = (~padding_mask).unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        
        # Classification
        return self.classifier(x)


class TransformerIRSystem:
    """Complete Information Retrieval System."""
    
    def __init__(self, max_vocab_size: int = 10000, d_model: int = 256, 
                 nhead: int = 8, num_layers: int = 3, max_seq_length: int = 256):
        self.preprocessor = TextPreprocessor(max_vocab_size)
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_data(self, relevant_path: str, non_relevant_path: str) -> Tuple[List, List]:
        """Load data from JSON files."""
        with open(relevant_path, 'r', encoding='utf-8') as f:
            relevant_data = json.load(f)
        
        with open(non_relevant_path, 'r', encoding='utf-8') as f:
            non_relevant_data = json.load(f)
        
        # Process relevant documents (have separate title and abstract)
        relevant_texts = []
        for doc_id, doc in relevant_data.items():
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            text = f"{title} {abstract}"
            relevant_texts.append((text, 1))  # Label 1 for relevant
        
        # Process non-relevant documents (title embedded in abstract)
        non_relevant_texts = []
        for doc_id, doc in non_relevant_data.items():
            abstract = doc.get('abstract', '')
            # Extract title from abstract (usually before first newline or period)
            # For PubMed format, title is typically after journal info and before author info
            text = self._extract_text_from_pubmed_abstract(abstract)
            non_relevant_texts.append((text, 0))  # Label 0 for non-relevant
        
        return relevant_texts, non_relevant_texts
    
    def _extract_text_from_pubmed_abstract(self, abstract: str) -> str:
        """Extract relevant text from PubMed formatted abstract."""
        if not abstract:
            return ""
        
        # Split by lines
        lines = abstract.split('\n')
        
        # Find title (usually line 3-4, ends with period)
        title = ""
        main_text = []
        
        # Skip the first few lines (journal info, DOI, etc.)
        start_idx = 0
        for i, line in enumerate(lines):
            line = line.strip()
            # Title typically appears after journal line and before author info
            if i < 10 and line and not line.startswith('Author') and len(line) > 20:
                if '.' in line and not line.startswith('DOI:') and not line.startswith('PMID:'):
                    title = line
                    start_idx = i + 1
                    break
        
        # Extract main abstract text (after author info)
        in_abstract = False
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            
            # Skip author information section
            if line.startswith('Author information:'):
                continue
            
            # Main abstract sections typically start with keywords like BACKGROUND, METHODS, etc.
            if any(line.startswith(keyword) for keyword in 
                   ['BACKGROUND:', 'INTRODUCTION:', 'OBJECTIVE:', 'METHODS:', 'METHODOLOGY:',
                    'RESULTS:', 'CONCLUSION:', 'PURPOSE:', 'AIM:']):
                in_abstract = True
            
            # Skip metadata at the end
            if any(line.startswith(meta) for meta in 
                   ['Copyright', 'DOI:', 'PMCID:', 'PMID:', '©']):
                break
            
            if in_abstract and line:
                main_text.append(line)
        
        # Combine title and abstract text
        combined = f"{title} {' '.join(main_text)}"
        return combined if combined.strip() else abstract
    
    def prepare_datasets(self, relevant_texts: List, non_relevant_texts: List, 
                        train_size: float = 0.7, val_size: float = 0.15):
        """Split data into train, validation, and test sets."""
        all_data = relevant_texts + non_relevant_texts
        
        # First split: train and temp (val + test)
        train_data, temp_data = train_test_split(
            all_data, train_size=train_size, random_state=42, stratify=[x[1] for x in all_data]
        )
        
        # Second split: validation and test
        val_ratio = val_size / (1 - train_size)
        val_data, test_data = train_test_split(
            temp_data, train_size=val_ratio, random_state=42, stratify=[x[1] for x in temp_data]
        )
        
        print(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Build vocabulary on training data
        train_texts = [text for text, _ in train_data]
        self.preprocessor.build_vocabulary(train_texts)
        print(f"Vocabulary size: {self.preprocessor.vocab_size}")
        
        # Create datasets
        train_dataset = PolyphenolDataset(train_data, self.preprocessor, self.max_seq_length)
        val_dataset = PolyphenolDataset(val_data, self.preprocessor, self.max_seq_length)
        test_dataset = PolyphenolDataset(test_data, self.preprocessor, self.max_seq_length)
        
        return train_dataset, val_dataset, test_dataset
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
              batch_size: int = 32, epochs: int = 10, lr: float = 0.001):
        """Train the model."""
        # Initialize model
        self.model = TransformerEncoder(
            vocab_size=self.preprocessor.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            max_seq_length=self.max_seq_length
        ).to(self.device)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
        
        best_val_f1 = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for texts, labels in train_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                labels = labels.squeeze()
                
                optimizer.zero_grad()
                outputs = self.model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_metrics = self.evaluate(val_loader)
            scheduler.step(val_metrics['f1'])
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model_state = self.model.state_dict().copy()
                print(f"  New best model saved!")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        print(f"\nTraining completed. Best validation F1: {best_val_f1:.4f}")
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for texts, labels in data_loader:
                texts = texts.to(self.device)
                labels = labels.squeeze()
                
                outputs = self.model(texts)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0)
        }
        
        return metrics
    
    def predict(self, text: str) -> Tuple[int, float]:
        """Predict if a text is relevant."""
        self.model.eval()
        encoded = self.preprocessor.encode(text, self.max_seq_length)
        input_tensor = torch.LongTensor([encoded]).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        return pred, confidence
    
    def save_model(self, path: str):
        """Save model and preprocessor."""
        torch.save({
            'model_state': self.model.state_dict(),
            'vocab': self.preprocessor.word2idx,
            'config': {
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'max_seq_length': self.max_seq_length,
                'vocab_size': self.preprocessor.vocab_size
            }
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and preprocessor."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.preprocessor.word2idx = checkpoint['vocab']
        self.preprocessor.idx2word = {v: k for k, v in checkpoint['vocab'].items()}
        self.preprocessor.vocab_size = len(checkpoint['vocab'])
        
        config = checkpoint['config']
        self.model = TransformerEncoder(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_seq_length=config['max_seq_length']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        print(f"Model loaded from {path}")


def main():
    """Main execution function."""
    # Initialize system
    ir_system = TransformerIRSystem(
        max_vocab_size=10000,
        d_model=256,
        nhead=8,
        num_layers=3,
        max_seq_length=256
    )
    
    # Load data
    print("Loading data...")
    relevant_texts, non_relevant_texts = ir_system.load_data(
        'data/output/alternative_abstracts.json',
        'data/output/non_relevant_abstracts.json'
    )
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset, val_dataset, test_dataset = ir_system.prepare_datasets(
        relevant_texts, non_relevant_texts
    )
    
    # Train model
    print("\nTraining model...")
    ir_system.train(
        train_dataset,
        val_dataset,
        batch_size=32,
        epochs=15,
        lr=0.001
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loader = DataLoader(test_dataset, batch_size=32)
    test_metrics = ir_system.evaluate(test_loader)
    
    print("\n=== Test Set Results ===")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-Score:  {test_metrics['f1']:.4f}")
    
    # Save model
    ir_system.save_model('transformer_ir_model.pth')
    
    # Example prediction
    print("\n=== Example Prediction ===")
    example_text = "Polyphenol composition and antioxidant activity in strawberry purees"
    pred, conf = ir_system.predict(example_text)
    print(f"Text: {example_text}")
    print(f"Prediction: {'Relevant' if pred == 1 else 'Non-relevant'}")
    print(f"Confidence: {conf:.4f}")


if __name__ == "__main__":
    main()