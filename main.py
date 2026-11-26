"""
Main Script for Medical Image Severity Classification
Interactive menu for training, evaluation, and prediction
"""

import os
import sys
from utils import (print_banner, check_dataset_structure, 
                   check_model_exists, validate_image_path,
                   create_output_directories)


def print_menu():
    """Print the main menu options."""
    print("\n" + "=" * 60)
    print("  Medical Image Severity Classification")
    print("  VGG16 Transfer Learning + Fuzzy Logic")
    print("=" * 60)
    print("  1. Train Model")
    print("  2. Evaluate Model")
    print("  3. Predict Single Image")
    print("  4. Launch GUI")
    print("  5. Exit")
    print("=" * 60)


def main():
    """
    Main interactive menu loop.
    
    Provides options to:
    1. Train the hybrid model on training dataset
    2. Evaluate trained model on test dataset
    3. Predict severity for a single MRI image
    4. Exit the program
    """
    
    print_banner("Medical MRI Severity Classification System")
    
    # Create output directories
    create_output_directories()
    
    # Check if dataset exists
    if not check_dataset_structure():
        print("\nERROR: Dataset structure is incomplete.")
        print("Please ensure dataset/ folder exists with Training/ and Testing/ subdirectories")
        print("Each should contain: glioma/, meningioma/, notumor/, pituitary/ folders")
        sys.exit(1)
    
    print("✓ Dataset structure verified")
    
    while True:
        print_menu()
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            # Train model
            print("\nStarting model training...")
            print("This will take 10-20 minutes on CPU, much faster on GPU.")
            
            try:
                os.system('python train.py')
            except Exception as e:
                print(f"ERROR during training: {e}")
        
        elif choice == '2':
            # Evaluate model
            print("\nEvaluating model...")
            
            if not check_model_exists():
                print("Please train the model first (option 1)")
                continue
            
            try:
                os.system('python evaluate.py')
            except Exception as e:
                print(f"ERROR during evaluation: {e}")
        
        elif choice == '3':
            # Predict single image
            if not check_model_exists():
                print("Please train the model first (option 1)")
                continue
            
            image_path = input("\nEnter MRI image path: ").strip()
            
            if not validate_image_path(image_path):
                continue
            
            print("\nRunning prediction...")
            
            try:
                # Use forward slashes for quoted paths
                image_path = image_path.replace('\\', '/')
                os.system(f'python predict.py --image "{image_path}"')
            except Exception as e:
                print(f"ERROR during prediction: {e}")
        
        elif choice == '4':
            # Launch GUI
            print("\nLaunching GUI...")
            try:
                os.system('python gui.py')
            except Exception as e:
                print(f"ERROR launching GUI: {e}")
        
        elif choice == '5':
            # Exit
            print("\n" + "=" * 60)
            print("  Thank you for using Medical Image Classification System")
            print("=" * 60)
            print("Exiting... Goodbye!\n")
            break
        
        else:
            # Invalid choice
            print("✗ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)
