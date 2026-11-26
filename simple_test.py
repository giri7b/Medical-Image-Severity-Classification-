"""
Simple test without TensorFlow
"""

import cv2
import numpy as np
import os

# Test basic image loading
test_file = None
for class_name in ['glioma', 'meningioma', 'notumor', 'pituitary']:
    class_path = f'dataset/Training/{class_name}'
    if os.path.exists(class_path):
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            test_file = os.path.join(class_path, images[0])
            break

if test_file:
    print(f"Testing with: {test_file}")
    img = cv2.imread(test_file)
    if img is not None:
        print(f"✓ Image loaded: shape {img.shape}")
        # Basic resize
        resized = cv2.resize(img, (224, 224))
        print(f"✓ Resized: shape {resized.shape}")
        print("Basic preprocessing works!")
    else:
        print("✗ Failed to load image")
else:
    print("No test image found")