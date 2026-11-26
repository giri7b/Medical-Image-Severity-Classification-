"""
Prediction Script for Medical Image Severity Classification
Predicts severity for a single MRI image with detailed analysis
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from preprocessing import full_preprocessing_pipeline, resize_and_convert
from hybrid_model import HybridVGG16Fuzzy


def visualize_prediction(result):
    """
    Create detailed visualization of prediction with preprocessing stages.
    
    Shows 3 subplots:
    1. Original MRI image
    2. CLAHE contrast-enhanced version
    3. Final preprocessed image ready for VGG16
    
    Includes prediction result in main title.
    
    Args:
        result: Prediction result dict from model.predict_single()
    """
    image_path = result['image_path']
    
    # Load original image
    original = cv2.imread(image_path)
    if original is not None:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Load and preprocess with intermediate steps for visualization
    img_rgb = resize_and_convert(image_path)
    
    if img_rgb is None:
        print("Could not load image for visualization")
        return
    
    # Apply preprocessing steps to show intermediate results
    from preprocessing import remove_noise, skull_strip, apply_clahe, sharpen_edges
    
    img_denoised = remove_noise(img_rgb)
    img_skull = skull_strip(img_denoised)
    img_enhanced = apply_clahe(img_skull)
    img_sharp = sharpen_edges(img_enhanced)
    
    # Create 3-subplot visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Subplot 1: Original image
    if original is not None:
        axes[0].imshow(cv2.resize(original, (224, 224)))
        axes[0].set_title('Original MRI Image', fontsize=12, fontweight='bold')
    else:
        axes[0].imshow(img_rgb)
        axes[0].set_title('Resized to 224x224', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Subplot 2: Skull-stripped and denoised
    axes[1].imshow(img_skull)
    axes[1].set_title('Skull-Stripped & Denoised', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Subplot 3: Final preprocessed
    axes[2].imshow(img_sharp)
    axes[2].set_title('Enhanced & Sharpened', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Main title with prediction
    severity_label = result['final_label'].upper()
    severity_score = result['fuzzy_score']
    fig.suptitle(
        f"Predicted Severity: {severity_label} (Score: {severity_score:.2f}/3.0)",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results/', exist_ok=True)
    output_path = 'results/prediction_output.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Prediction visualization saved to {output_path}")
    
    plt.show()


def print_prediction_result(result):
    """
    Print detailed prediction result in a formatted box.
    
    Shows:
    - Final prediction label and fuzzy score
    - CNN confidence for each class (with progress bars)
    - Fuzzy logic input values
    
    Args:
        result: Prediction result dict from model.predict_single()
    """
    print("\n" + "="*50)
    print("     MRI SEVERITY CLASSIFICATION RESULT")
    print("="*50)
    
    print(f"  Image       : {result['image_path']}")
    print(f"  Final Label : {result['final_label'].upper()}")
    print(f"  Fuzzy Score : {result['fuzzy_score']:.2f} / 3.0")
    print(f"  Decision   : {result.get('decision_method', 'Hybrid')}")

    # Add tumor likelihood interpretation
    severity_order = ['NORMAL', 'MILD', 'MODERATE', 'SEVERE']
    severity_idx = severity_order.index(result['final_label'].upper()) if result['final_label'].upper() in severity_order else 1
    if severity_idx <= 1:
        tumor_risk = 'LOW'
        risk_details = 'Low likelihood of severe tumor progression'
    elif severity_idx == 2:
        tumor_risk = 'MEDIUM'
        risk_details = 'Moderate likelihood of tumor severity; follow-up recommended'
    else:
        tumor_risk = 'HIGH'
        risk_details = 'High likelihood of severe tumor pathology; urgent review recommended'

    print(f"  Tumor Risk : {tumor_risk}")
    print(f"  Notes      : {risk_details}")
    
    print("-"*50)
    print("  CNN Confidence:")
    
    # Get CNN probabilities
    cnn_probs = result['cnn_probabilities']
    
    for label in ['Normal', 'Mild', 'Moderate', 'Severe']:
        prob = cnn_probs[label]
        # Create visual bar (20 characters max)
        bar_length = int(prob * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"    {label:<10}: {bar} {prob*100:6.1f}%")
    
    print("-"*50)
    print("  Fuzzy Logic Inputs:")
    
    fuzzy_inputs = result['fuzzy_inputs']
    print(f"    Tumor Intensity      : {fuzzy_inputs['tumor_intensity']:.3f}")
    print(f"    Texture Irregularity : {fuzzy_inputs['texture_irregularity']:.3f}")
    print(f"    Boundary Clarity     : {fuzzy_inputs['boundary_clarity']:.3f}")
    
    print("="*50 + "\n")


def main():
    """
    Main prediction function.
    
    Command line usage:
        python predict.py --image path/to/image.jpg
    
    Accepts MRI image file and outputs:
    1. Formatted text result with CNN confidence and fuzzy inputs
    2. Visualization showing preprocessing stages
    3. Result saved to results/prediction_output.png
    """
    
    parser = argparse.ArgumentParser(
        description='Predict MRI severity using VGG16 + Fuzzy Logic model'
    )
    parser.add_argument('--image', required=True, 
                       help='Path to MRI image file')
    
    args = parser.parse_args()
    
    image_path = args.image
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        return
    
    print("=" * 70)
    print("  Medical MRI Severity Classification - Single Image Prediction")
    print("  VGG16 Transfer Learning + Fuzzy Logic Hybrid Model")
    print("=" * 70)
    
    # Load saved model
    print("\nLoading trained model...")
    
    model = HybridVGG16Fuzzy()
    model.load('saved_model/')
    
    # Run prediction
    print(f"Predicting for image: {image_path}")
    
    result = model.predict_single(image_path)
    
    if result is None:
        print("ERROR: Prediction failed (image preprocessing error)")
        return
    
    # Print formatted result
    print_prediction_result(result)
    
    # Create visualization
    print("Generating visualization...")
    visualize_prediction(result)
    
    print("=" * 70)


if __name__ == "__main__":
    main()
