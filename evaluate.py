"""
Evaluation Script for Medical Image Severity Classification
Evaluates the trained hybrid model on test dataset
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import load_dataset
from hybrid_model import HybridVGG16Fuzzy


def main():
    """
    Complete evaluation pipeline with 4 main steps:
    1. Load test dataset
    2. Load saved hybrid model
    3. Run evaluation and compute accuracy
    4. Generate classification report, confusion matrix, and accuracy charts
    """
    
    # Create output directories
    os.makedirs('results/', exist_ok=True)
    os.makedirs('saved_model/', exist_ok=True)
    
    # Print header banner
    print("=" * 70)
    print("  Medical MRI Severity Classification - Evaluation")
    print("  VGG16 Transfer Learning + Fuzzy Logic Hybrid Model")
    print("=" * 70)
    
    # ===== STEP 1: Load test dataset =====
    print("\n[1/4] Loading and preprocessing test dataset...")
    print("-" * 70)
    
    # Load test dataset WITHOUT augmentation (original images only)
    X_test, y_test = load_dataset('dataset/Testing/', augment=False)
    
    print(f"✓ Test samples loaded: {len(X_test)}")
    
    # ===== STEP 2: Load saved hybrid model =====
    print("\n[2/4] Loading saved hybrid model...")
    print("-" * 70)
    
    model = HybridVGG16Fuzzy()
    model.load('saved_model/')
    
    # ===== STEP 3: Evaluate model =====
    print("\n[3/4] Evaluating model on test set...")
    print("-" * 70)
    
    hybrid_accuracy, y_pred, y_true = model.evaluate(X_test, y_test)
    cnn_accuracy, cnn_preds, _ = model.evaluate(X_test, y_test, use_fuzzy=False)
    
    print(f"✓ Evaluation completed!")
    print(f"  - Hybrid accuracy: {hybrid_accuracy*100:.2f}%")
    print(f"  - CNN-only accuracy: {cnn_accuracy*100:.2f}%")
    
    # ===== STEP 4: Generate evaluation reports and visualizations =====
    print("\n[4/4] Generating evaluation reports and visualizations...")
    print("-" * 70)
    
    # Define label names
    labels = ['Normal', 'Mild', 'Moderate', 'Severe']
    
    # ===== Classification Report =====
    print("\nDetailed Classification Report:")
    print("-" * 70)
    
    class_report = classification_report(y_true, y_pred, target_names=labels)
    print(class_report)
    
    # Save classification report to file
    with open('results/classification_report.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Overall Accuracy: {hybrid_accuracy*100:.2f}%\n\n")
        f.write(class_report)
    
    # ===== Confusion Matrix =====
    print("\nGenerating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - VGG16 + Fuzzy Logic\n' + 
                 f'Overall Accuracy: {hybrid_accuracy*100:.2f}%', 
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to results/confusion_matrix.png")
    plt.close()
    
    # ===== Per-Class Accuracy =====
    print("Generating per-class accuracy chart...")
    
    # Calculate per-class accuracy
    class_accuracies = []
    for class_idx in range(4):
        # Indices of samples from this class
        class_mask = y_true == class_idx
        
        if np.sum(class_mask) == 0:
            # No samples of this class in test set
            class_accuracies.append(0.0)
        else:
            # Accuracy for this class
            class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
            class_accuracies.append(class_acc)
    
    # Plot per-class accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']  # Green, Orange, Red, Purple
    bars = ax.bar(labels, class_accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add percentage labels on top of bars
    for bar, accuracy_val in zip(bars, class_accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{accuracy_val*100:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Severity Class', fontsize=12)
    ax.set_title('Per-Class Accuracy - VGG16 + Fuzzy Logic', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/class_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"✓ Per-class accuracy chart saved to results/class_accuracy.png")
    plt.close()
    
    # ===== Summary Statistics =====
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {hybrid_accuracy*100:.2f}%")
    
    print("\nPer-Class Accuracy:")
    for label, acc in zip(labels, class_accuracies):
        class_idx = labels.index(label)
        class_samples = np.sum(y_true == class_idx)
        correct_samples = np.sum((y_pred == class_idx) & (y_true == class_idx))
        print(f"  {label:10} - {acc*100:6.2f}% ({correct_samples}/{class_samples} correct)")
    
    print(f"\nTest Samples: {len(y_test)}")
    
    print("\nGenerated Files:")
    print(f"  ✓ Classification Report: results/classification_report.txt")
    print(f"  ✓ Confusion Matrix: results/confusion_matrix.png")
    print(f"  ✓ Class Accuracy Chart: results/class_accuracy.png")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
