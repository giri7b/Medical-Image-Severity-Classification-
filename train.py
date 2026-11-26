"""
Training Script for Medical Image Severity Classification
Complete pipeline for training the hybrid VGG16 + Fuzzy Logic model
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import load_dataset
from hybrid_model import HybridVGG16Fuzzy


def main():
    """
    Complete training pipeline with 5 main steps:
    1. Load and preprocess training dataset
    2. Split into train/validation
    3. Build hybrid model
    4. Train model (2 phases)
    5. Save trained model and plot training curves
    """
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create output directories
    os.makedirs('results/', exist_ok=True)
    os.makedirs('saved_model/', exist_ok=True)
    
    # Print header banner
    print("=" * 70)
    print("  Medical MRI Severity Classification - Training")
    print("  VGG16 Transfer Learning + Fuzzy Logic Hybrid Model")
    print("=" * 70)
    
    # ===== STEP 1: Load and preprocess training dataset =====
    print("\n[1/5] Loading and preprocessing training dataset...")
    print("-" * 70)
    
# Load dataset with optional augmentation (default: enabled for better accuracy)
    # Use MAX_PER_CLASS environment variable to limit samples per class (useful for low-memory testing)
    max_per_class_env = os.getenv('MAX_PER_CLASS', '')
    max_per_class = int(max_per_class_env) if max_per_class_env.isdigit() and int(max_per_class_env) > 0 else 100

    # Set `AUGMENT_TRAINING` to '0' to disable augmentation, otherwise enable it by default.
    augment_training = os.getenv('AUGMENT_TRAINING', '1') != '0'
    print(f"Data augmentation enabled: {augment_training}")

    X, y = load_dataset('dataset/Training/', augment=augment_training, max_per_class=max_per_class)
    
    print(f"✓ Total samples loaded: {len(X)}")
    
    # ===== STEP 2: Split into train and validation =====
    print("\n[2/5] Splitting dataset into training and validation...")
    print("-" * 70)
    
    # Use 80-20 split with stratification to preserve class distribution
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    training_labels, training_counts = np.unique(y_train, return_counts=True)
    print(f"\nTraining set class distribution:")
    label_map = {0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
    for label, count in zip(training_labels, training_counts):
        percentage = (count / len(y_train)) * 100
        print(f"  {label_map[label]}: {count} ({percentage:.1f}%)")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight = {int(cls): float(weight) for cls, weight in zip(np.unique(y_train), class_weights)}
    print(f"\nClass weights: {class_weight}")
    
    # ===== STEP 3: Build hybrid model =====
    print("\n[3/5] Building Hybrid VGG16 + Fuzzy Logic model...")
    print("-" * 70)
    
    model = HybridVGG16Fuzzy()
    model.build()
    
    # ===== STEP 4: Train model =====
    print("\n[4/5] Training model (Phase 1 + Phase 2)...")
    print("-" * 70)
    
    history = model.train(X_train, y_train, X_val, y_val, class_weight=class_weight)
    
    # ===== STEP 5: Save model and plot training curves =====
    print("\n[5/5] Saving trained model and plotting results...")
    print("-" * 70)
    
    # Save trained models
    model.save('saved_model/')
    
    # Plot training curves
    print("Generating training curves...")
    
    # Extract history from both phases
    hist1 = history['phase1']
    hist2 = history['phase2']
    
    # Combine phase 1 and phase 2 histories
    phase1_epochs = len(hist1.history['accuracy'])
    phase2_epochs = len(hist2.history['accuracy'])
    total_epochs = phase1_epochs + phase2_epochs
    
    # Combine training and validation curves
    combined_accuracy = hist1.history['accuracy'] + hist2.history['accuracy']
    combined_val_accuracy = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    combined_loss = hist1.history['loss'] + hist2.history['loss']
    combined_val_loss = hist1.history['val_loss'] + hist2.history['val_loss']
    
    epochs_range = range(1, len(combined_accuracy) + 1)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training vs Validation Accuracy
    axes[0].plot(epochs_range, combined_accuracy, 'b-', label='Training Accuracy', linewidth=2)
    axes[0].plot(epochs_range, combined_val_accuracy, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0].axvline(x=phase1_epochs, color='gray', linestyle='--', linewidth=1, 
                    label=f'Phase 1→2 (epoch {phase1_epochs})')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy: Training vs Validation', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Training vs Validation Loss
    axes[1].plot(epochs_range, combined_loss, 'b-', label='Training Loss', linewidth=2)
    axes[1].plot(epochs_range, combined_val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[1].axvline(x=phase1_epochs, color='gray', linestyle='--', linewidth=1,
                    label=f'Phase 1→2 (epoch {phase1_epochs})')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss: Training vs Validation', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to results/training_history.png")
    plt.close()
    
    # Print final training summary
    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    
    final_train_accuracy = combined_accuracy[-1]
    final_val_accuracy = combined_val_accuracy[-1]
    
    print(f"Final Training Accuracy:   {final_train_accuracy*100:.2f}%")
    print(f"Final Validation Accuracy: {final_val_accuracy*100:.2f}%")
    print(f"Total Epochs Trained:      {total_epochs}")
    print(f"  - Phase 1 (Head Training): {phase1_epochs} epochs")
    print(f"  - Phase 2 (Fine-tuning):   {phase2_epochs} epochs")
    print(f"\n✓ Model saved to: saved_model/")
    print(f"✓ Training curves saved to: results/training_history.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
