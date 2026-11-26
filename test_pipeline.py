"""
Quick test script to verify the pipeline works
"""

import sys
sys.path.insert(0, 'C:/Python313/Lib/site-packages')

import os
import cv2
import numpy as np
from preprocessing import full_preprocessing_pipeline

# Test with one image from each class
test_images = []
for class_name in ['glioma', 'meningioma', 'notumor', 'pituitary']:
    class_path = f'dataset/Training/{class_name}'
    if os.path.exists(class_path):
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            test_images.append(os.path.join(class_path, images[0]))

print(f"Testing with {len(test_images)} images...")

for img_path in test_images:
    print(f"Processing: {os.path.basename(img_path)}")
    result = full_preprocessing_pipeline(img_path)
    if result is not None:
        print(f"  ✓ Success: shape {result.shape}")
    else:
        print(f"  ✗ Failed")

print("Test completed!")