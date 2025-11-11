"""
Visualization Script for Report Generation
Creates plots and figures for the assignment report
Requires: matplotlib, seaborn
Install: pip install matplotlib seaborn
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from torch.utils.data import DataLoader
from transformer_encoder import TransformerIRSystem, PolyphenolDataset


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Relevant', 'Relevant'],
                yticklabels=['Non-Relevant', 'Relevant'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation metrics over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['val_accuracy'], 'g-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(epochs, history['val_precision'], 'r-', label='Validation Precision', linewidth=2)
    axes[1, 0].set_title('Validation Precision', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 1].plot(epochs, history['val_f1'], 'm-', label='Validation F1-Score', linewidth=2)
    axes[1, 1].set_title('Validation F1-Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_scores, save_path='roc_curve.png'):
    """Generate and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path='metrics_comparison.png'):
    """Compare different metrics in a bar chart."""
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.title('Test Set Performance Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.ylim([0, 1.1])
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison saved to {save_path}")
    plt.close()


def plot_prediction_distribution(predictions, confidences, save_path='prediction_distribution.png'):
    """Plot distribution of predictions and confidence scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prediction distribution
    unique, counts = np.unique(predictions, return_counts=True)
    labels = ['Non-Relevant', 'Relevant']
    colors = ['#ff6b6b', '#51cf66']
    
    axes[0].bar(labels, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].set_title('Prediction Distribution', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=11)
    for i, count in enumerate(counts):
        axes[0].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # Confidence distribution
    axes[1].hist(confidences, bins=20, color='skyblue', alpha=0.7, edgecolor='black', linewidth=1.2)
    axes[1].set_title('Confidence Score Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Confidence', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].axvline(np.mean(confidences), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(confidences):.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction distribution saved to {save_path}")
    plt.close()


def generate_all_visualizations(ir_system, test_dataset, output_dir='figures/'):
    """Generate all visualizations for the report."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating visualizations for report...")
    print("="*60)
    
    # Get predictions and labels
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    ir_system.model.eval()
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(ir_system.device)
            labels = labels.squeeze()
            
            outputs = ir_system.model(texts)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            # Get probability of positive class (relevant)
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Confidence is the max probability for each prediction
    confidences = np.array([all_probs[i] if all_preds[i] == 1 
                           else 1 - all_probs[i] 
                           for i in range(len(all_preds))])
    
    # Generate plots
    plot_confusion_matrix(all_labels, all_preds, 
                         save_path=f'{output_dir}confusion_matrix.png')
    
    plot_roc_curve(all_labels, all_probs, 
                  save_path=f'{output_dir}roc_curve.png')
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = {
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Precision': precision_score(all_labels, all_preds),
        'Recall': recall_score(all_labels, all_preds),
        'F1-Score': f1_score(all_labels, all_preds)
    }
    
    plot_metrics_comparison(metrics, 
                           save_path=f'{output_dir}metrics_comparison.png')
    
    plot_prediction_distribution(all_preds, confidences,
                                save_path=f'{output_dir}prediction_distribution.png')
    
    print("="*60)
    print(f"All visualizations saved to '{output_dir}' directory")
    print("\nGenerated files:")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    print(f"  - metrics_comparison.png")
    print(f"  - prediction_distribution.png")
    print("\nInclude these figures in your report!")


def main():
    """Main function to generate visualizations."""
    print("Loading trained model...")
    ir_system = TransformerIRSystem()
    
    try:
        ir_system.load_model('transformer_ir_model.pth')
    except FileNotFoundError:
        print("Error: Model file 'transformer_ir_model.pth' not found!")
        print("Please train the model first by running: python transformer_ir_system.py")
        return
    
    print("Loading test data...")
    relevant_texts, non_relevant_texts = ir_system.load_data(
        'data/output/alternative_abstracts.json',
        'data/output/non_relevant_abstracts.json'
    )
    
    # Recreate the same split as in training
    from sklearn.model_selection import train_test_split
    all_data = relevant_texts + non_relevant_texts
    
    train_data, temp_data = train_test_split(
        all_data, train_size=0.7, random_state=42, 
        stratify=[x[1] for x in all_data]
    )
    
    val_data, test_data = train_test_split(
        temp_data, train_size=0.5, random_state=42,
        stratify=[x[1] for x in temp_data]
    )
    
    test_dataset = PolyphenolDataset(test_data, ir_system.preprocessor, 
                                     ir_system.max_seq_length)
    
    print(f"Test set size: {len(test_dataset)}")
    
    # Generate all visualizations
    generate_all_visualizations(ir_system, test_dataset)
    
    print("\n✅ Visualization generation complete!")


if __name__ == "__main__":
    main()