"""
Medical Image Preprocessing Pipeline
Specialized for brain MRI images - converts to 224x224 RGB for VGG16
"""

import cv2
import numpy as np
from tqdm import tqdm
import os


def resize_and_convert(image_path, size=(224, 224)):
    """
    Read image and convert to RGB 224x224 format required by VGG16.
    
    VGG16 requires RGB input at exactly 224x224 pixels (NOT grayscale).
    This is the first step in the preprocessing pipeline.
    
    Args:
        image_path: Path to the MRI image file
        size: Target size tuple (height, width), default 224x224
        
    Returns:
        RGB resized image as numpy array shape (224, 224, 3)
        Returns None if image cannot be loaded
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert BGR to RGB (OpenCV reads in BGR format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def remove_noise(image):
    """
    Remove noise from image using median blur filter.
    
    Median filter is superior to Gaussian blur for medical images
    because it preserves tumor edges while removing noise.
    Morphological operations can distort tumor boundaries badly.
    
    Args:
        image: RGB image array to denoise
        
    Returns:
        Denoised RGB image of same shape
    """
    # Ensure uint8 format for cv2 operations
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Apply median blur (kernel size 3x3)
    denoised = cv2.medianBlur(image, 3)
    
    return denoised


def skull_strip(image):
    """
    Remove skull and non-brain tissue from brain MRI using adaptive thresholding.
    
    Process:
    1. Create grayscale version for masking only
    2. Apply Otsu's automatic thresholding to separate brain from background
    3. Use morphological closing to fill small holes in mask
    4. Find the largest contour (brain region)
    5. Apply mask to all 3 RGB channels to remove skull
    
    This preserves the RGB image while removing surrounding skull tissue.
    
    Args:
        image: RGB image array (from resize_and_convert)
        
    Returns:
        Skull-stripped RGB image of same shape as input
        Returns original image if skull strip fails
    """
    try:
        # Temporarily convert to grayscale just for finding the mask
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Otsu's automatic thresholding
        _, thresh = cv2.threshold(gray, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological closing to fill holes in the mask
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, return original image unchanged
        if len(contours) == 0:
            return image
        
        # Find the largest contour (the brain region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create an empty mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Draw the largest contour filled on the mask
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Apply mask to each RGB channel
        skull_stripped = cv2.bitwise_and(image, image, mask=mask)
        
        return skull_stripped
        
    except Exception as e:
        print(f"Skull strip failed: {e}, returning original image")
        return image


def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) in LAB space.
    
    CLAHE enhances local contrast in medical images.
    Working in LAB color space instead of RGB preserves color information
    while only enhancing the brightness (L channel).
    
    Process:
    1. Convert RGB to LAB color space
    2. Extract L (lightness) channel
    3. Apply CLAHE to L channel only
    4. Merge back and convert to RGB
    
    Args:
        image: RGB image array (from remove_noise and skull_strip)
        
    Returns:
        Contrast-enhanced RGB image of same shape
    """
    # Convert RGB to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split into L, A, B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE only to L channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge channels back
    merged = cv2.merge([l_enhanced, a, b])
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb


def sharpen_edges(image):
    """
    Sharpen image edges using unsharp masking.
    
    Unsharp masking enhances tumor boundaries by:
    1. Creating a blurred version (removes details)
    2. Subtracting blur from original (highlights differences)
    
    This makes tumor edges more defined for better feature extraction.
    
    Args:
        image: RGB image array (from apply_clahe)
        
    Returns:
        Sharpened RGB image of same shape
    """
    # Create blurred version using Gaussian blur with sigma=3
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    
    # Apply unsharp masking: original*1.5 - blur*0.5
    # This emphasizes differences between original and blurred
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    
    # Clip values to valid range [0, 255] and convert to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


def normalize_for_vgg(image):
    """
    Apply VGG16-specific normalization using ImageNet preprocessing.
    
    VGG16 was trained on ImageNet using specific mean subtraction:
    - Subtracts RGB means: [103.93, 116.77, 123.68]
    - Does NOT just divide by 255
    
    This normalization is CRITICAL for VGG16 to work correctly.
    Must use keras.applications.vgg16.preprocess_input, not manual normalization.
    
    Args:
        image: RGB image array shape (224, 224, 3) as uint8
        
    Returns:
        Preprocessed image of shape (224, 224, 3) ready for VGG16
    """
    try:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        
        # Convert to float32 for preprocessing
        image_float = image.astype(np.float32)
        
        # Apply VGG16-specific preprocessing (subtracts ImageNet means)
        preprocessed = preprocess_input(image_float)
        
        return preprocessed
    except ImportError:
        # Fallback: simple normalization if TensorFlow not available
        print("Warning: TensorFlow not available, using simple normalization")
        return image.astype(np.float32) / 255.0


def augment_image(image):
    """
    Generate 5 augmented versions of the image for training data augmentation.
    
    Data augmentation helps the model generalize better by showing different
    versions of the same image during training.
    
    Returns:
        List of 5 augmented images:
        1. Original image unchanged
        2. Horizontally flipped
        3. Rotated +10 degrees
        4. Rotated -10 degrees
        5. Brightness increased by 20%
        
    All images maintain the same shape (224, 224, 3).
    
    Args:
        image: RGB image array shape (224, 224, 3), should be BEFORE normalization
        
    Returns:
        List of 5 augmented images (all uint8, before normalization)
    """
    # Ensure uint8 for augmentations
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    augmented_images = []
    
    # 1. Original image
    augmented_images.append(image.copy())
    
    # 2. Horizontal flip
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    
    # 3. Rotation +10 degrees
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 10, 1.0)
    rotated_pos = cv2.warpAffine(image, rotation_matrix, (width, height))
    augmented_images.append(rotated_pos)
    
    # 4. Rotation -10 degrees
    rotation_matrix = cv2.getRotationMatrix2D(center, -10, 1.0)
    rotated_neg = cv2.warpAffine(image, rotation_matrix, (width, height))
    augmented_images.append(rotated_neg)
    
    # 5. Brightness boost (20% increase)
    brightness_boosted = np.clip(image.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
    augmented_images.append(brightness_boosted)
    
    return augmented_images


def full_preprocessing_pipeline(image_path):
    """
    Complete preprocessing pipeline: all 6 steps in sequence.
    
    Process:
    1. Load and convert to RGB 224x224
    2. Remove noise with median filter
    3. Remove skull tissue
    4. Enhance contrast with CLAHE
    5. Sharpen edges
    6. Apply VGG16 normalization
    
    All steps preserve the RGB format and 224x224 size.
    VGG16 will receive the exact format it expects.
    
    Args:
        image_path: Path to MRI image file
        
    Returns:
        Preprocessed image of shape (224, 224, 3) ready for VGG16
        Returns None if preprocessing fails at any step
    """
    try:
        # Step 1: Load and convert to RGB 224x224
        image = resize_and_convert(image_path)
        if image is None:
            return None
        
        # Step 2: Remove noise
        image = remove_noise(image)
        
        # Step 3: Remove skull
        image = skull_strip(image)
        
        # Step 4: Enhance contrast
        image = apply_clahe(image)
        
        # Step 5: Sharpen edges
        image = sharpen_edges(image)
        
        # Step 6: VGG16 normalization
        image = normalize_for_vgg(image)
        
        return image
        
    except Exception as e:
        print(f"Preprocessing pipeline failed for {image_path}: {e}")
        return None


def load_dataset(dataset_path, augment=True, max_per_class=None):
    """
    Load and preprocess entire dataset with optional augmentation.

    Process:
    1. Walk through class folders (glioma, meningioma, pituitary, notumor)
    2. For each image: apply full preprocessing pipeline
    3. If augment=True: create 5 augmented versions of each image then normalize
    4. If augment=False: just preprocess normally
    5. Collect all X (images) and y (labels) into numpy arrays
    6. Show class distribution stats

    Class mapping:
        notumor    -> 0 (Normal)
        meningioma -> 1 (Mild)
        pituitary  -> 2 (Moderate)
        glioma     -> 3 (Severe)

    Args:
        dataset_path: Path to dataset root folder (e.g., 'dataset/Training/')
        augment: If True, create 5x more samples with augmentation
        max_per_class: Optional maximum number of images to load per class (useful for low-memory testing)

    Returns:
        X: numpy array of shape (N, 224, 224, 3) - preprocessed images
        y: numpy array of shape (N,) - severity labels (0-3)
    """
    severity_map = {
        'notumor': 0,      # Normal
        'meningioma': 1,   # Mild
        'pituitary': 2,    # Moderate
        'glioma': 3        # Severe
    }
    
    X = []
    y = []
    
    # Get all class folders
    class_folders = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))]
    
    total_images = 0
    for image_file in os.listdir(dataset_path):
        if os.path.isfile(os.path.join(dataset_path, image_file)):
            total_images += 1
    
    for class_folder in class_folders:
        if class_folder not in severity_map:
            continue
            
        class_path = os.path.join(dataset_path, class_folder)
        severity_label = severity_map[class_folder]
        
        # Get all image files in the class folder
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        if max_per_class is not None:
            image_files = image_files[:max_per_class]
        
        pbar = tqdm(image_files, desc=f"Loading {class_folder}", leave=False)
        
        for image_file in pbar:
            image_path = os.path.join(class_path, image_file)
            
            try:
                if augment:
                    # Load image without normalization for augmentation
                    full_img = resize_and_convert(image_path)
                    if full_img is None:
                        continue
                    
                    full_img = remove_noise(full_img)
                    full_img = skull_strip(full_img)
                    full_img = apply_clahe(full_img)
                    full_img = sharpen_edges(full_img)
                    
                    # Generate 5 augmented versions
                    augmented = augment_image(full_img)
                    
                    # Normalize each augmented version
                    for aug_img in augmented:
                        normalized = normalize_for_vgg(aug_img)
                        X.append(normalized)
                        y.append(severity_label)
                else:
                    # No augmentation, just preprocess normally
                    img = full_preprocessing_pipeline(image_path)
                    if img is None:
                        continue
                    
                    X.append(img)
                    y.append(severity_label)
                    
            except Exception as e:
                print(f"Skipping unreadable image: {image_path}")
                continue
    
    X = np.array(X)
    y = np.array(y)
    
    # Print class distribution
    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(X)}")
    print("Class distribution:")
    for class_name, class_idx in severity_map.items():
        count = np.sum(y == class_idx)
        percentage = (count / len(y) * 100) if len(y) > 0 else 0
        print(f"  {class_name:12} ({class_idx}): {count:5} samples ({percentage:5.1f}%)")
    
    return X, y
