"""
Utility Functions for Medical Image Severity Classification
Helper functions for common operations
"""

import os
import sys


def print_banner(title):
    """
    Print a formatted banner for terminal output.
    
    Args:
        title: Title text to display in banner
    """
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def check_dataset_structure():
    """
    Verify that the dataset directory structure is correct.
    
    Returns:
        bool: True if structure is valid, False otherwise
    """
    required_dirs = [
        'dataset/Training/glioma',
        'dataset/Training/meningioma',
        'dataset/Training/notumor',
        'dataset/Training/pituitary',
        'dataset/Testing/glioma',
        'dataset/Testing/meningioma',
        'dataset/Testing/notumor',
        'dataset/Testing/pituitary'
    ]
    
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            print(f"ERROR: Required directory not found: {dir_path}")
            return False
    
    return True


def check_model_exists():
    """
    Check if trained model files exist.
    
    Returns:
        bool: True if model files exist, False otherwise
    """
    model_files = [
        'saved_model/vgg16_model.keras',
        'saved_model/vgg16_model.h5'
    ]
    extractor_files = [
        'saved_model/feature_extractor.keras',
        'saved_model/feature_extractor.h5'
    ]

    if not any(os.path.exists(path) for path in model_files):
        print("ERROR: Trained model not found. Please run training first.")
        return False

    if not any(os.path.exists(path) for path in extractor_files):
        print("ERROR: Feature extractor model not found. Please run training first.")
        return False

    return True


def validate_image_path(image_path):
    """
    Validate that an image file exists and has a valid extension.
    
    Args:
        image_path: Path to image file
        
    Returns:
        bool: True if valid, False otherwise
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        return False
    
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in valid_extensions:
        print(f"ERROR: Invalid image format. Supported formats: {valid_extensions}")
        return False
    
    return True


def create_output_directories():
    """
    Create necessary output directories if they don't exist.
    """
    os.makedirs('results/', exist_ok=True)
    os.makedirs('saved_model/', exist_ok=True)
