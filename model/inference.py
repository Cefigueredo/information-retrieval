"""
Inference Script for Transformer IR System
Allows prediction on new documents and batch evaluation
"""

import json
import argparse
from typing import List, Dict
import torch
from torch.utils.data import DataLoader
from transformer_encoder import TransformerIRSystem, PolyphenolDataset


def predict_single(ir_system: TransformerIRSystem, text: str):
    """Predict on a single text."""
    prediction, confidence = ir_system.predict(text)
    label = "RELEVANT" if prediction == 1 else "NON-RELEVANT"
    
    print(f"\n{'='*60}")
    print(f"Text: {text[:100]}...")
    print(f"{'='*60}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"{'='*60}\n")
    
    return prediction, confidence


def predict_from_json(ir_system: TransformerIRSystem, json_path: str, output_path: str = None):
    """Predict on documents from JSON file."""
    # Load documents
    with open(json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    results = {}
    relevant_count = 0
    
    print(f"\nProcessing {len(documents)} documents...")
    
    for doc_id, doc in documents.items():
        title = doc.get('title', '')
        abstract = doc.get('abstract', '')
        text = f"{title} {abstract}"
        
        prediction, confidence = ir_system.predict(text)
        
        results[doc_id] = {
            'title': title,
            'abstract': abstract,
            'prediction': 'relevant' if prediction == 1 else 'non_relevant',
            'confidence': float(confidence)
        }
        
        if prediction == 1:
            relevant_count += 1
    
    print(f"\nResults:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Predicted relevant: {relevant_count}")
    print(f"  Predicted non-relevant: {len(documents) - relevant_count}")
    
    # Save results if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results


def evaluate_with_labels(ir_system: TransformerIRSystem, json_path: str, 
                        true_label: int, batch_size: int = 32):
    """Evaluate model on labeled data."""
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Load documents
    with open(json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Prepare data
    data = []
    for doc_id, doc in documents.items():
        title = doc.get('title', '')
        abstract = doc.get('abstract', '')
        text = f"{title} {abstract}"
        data.append((text, true_label))
    
    # Create dataset
    dataset = PolyphenolDataset(data, ir_system.preprocessor, ir_system.max_seq_length)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    
    # Evaluate
    metrics = ir_system.evaluate(data_loader)
    
    # Get predictions for detailed analysis
    all_preds = []
    all_labels = []
    
    ir_system.model.eval()
    with torch.no_grad():
        for texts, labels in data_loader:
            texts = texts.to(ir_system.device)
            labels = labels.squeeze()
            
            outputs = ir_system.model(texts)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nDataset: {json_path}")
    print(f"True label: {'Relevant' if true_label == 1 else 'Non-relevant'}")
    print(f"Number of samples: {len(data)}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"                Predicted")
    print(f"              Non-rel  Relevant")
    print(f"Actual Non-rel  {cm[0][0]:6d}   {cm[0][1]:6d}")
    print(f"       Relevant {cm[1][0]:6d}   {cm[1][1]:6d}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=['Non-relevant', 'Relevant']))
    
    return metrics


def interactive_mode(ir_system: TransformerIRSystem):
    """Interactive prediction mode."""
    print("\n" + "="*60)
    print("INTERACTIVE PREDICTION MODE")
    print("="*60)
    print("Enter text to classify (or 'quit' to exit)")
    print("="*60 + "\n")
    
    while True:
        text = input("\nEnter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        predict_single(ir_system, text)


def main():
    parser = argparse.ArgumentParser(description='Inference script for Transformer IR System')
    parser.add_argument('--model', type=str, default='transformer_ir_model.pth',
                       help='Path to trained model')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'eval', 'interactive'],
                       default='interactive', help='Inference mode')
    parser.add_argument('--text', type=str, help='Text to classify (for single mode)')
    parser.add_argument('--input', type=str, help='Input JSON file (for batch/eval mode)')
    parser.add_argument('--output', type=str, help='Output JSON file (for batch mode)')
    parser.add_argument('--label', type=int, choices=[0, 1],
                       help='True label for evaluation (0=non-relevant, 1=relevant)')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    ir_system = TransformerIRSystem()
    ir_system.load_model(args.model)
    print("Model loaded successfully!\n")
    
    # Execute based on mode
    if args.mode == 'single':
        if not args.text:
            print("Error: --text required for single mode")
            return
        predict_single(ir_system, args.text)
    
    elif args.mode == 'batch':
        if not args.input:
            print("Error: --input required for batch mode")
            return
        predict_from_json(ir_system, args.input, args.output)
    
    elif args.mode == 'eval':
        if not args.input or args.label is None:
            print("Error: --input and --label required for eval mode")
            return
        evaluate_with_labels(ir_system, args.input, args.label)
    
    elif args.mode == 'interactive':
        interactive_mode(ir_system)


if __name__ == "__main__":
    main()