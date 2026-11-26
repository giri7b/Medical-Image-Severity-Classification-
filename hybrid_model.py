"""
Hybrid VGG16 + Fuzzy Logic Model
Complete system combining CNN classification with fuzzy logic reasoning
"""

import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from preprocessing import full_preprocessing_pipeline
from transfer_model import (build_vgg16_feature_extractor, compile_model,
                           train_model, fine_tune_model, 
                           build_feature_extractor_only,
                           save_model, load_saved_model)
from fuzzy_system import build_fuzzy_system, fuzzy_classify


class HybridVGG16Fuzzy:
    """
    Hybrid model combining VGG16 CNN with Fuzzy Logic System.
    
    Architecture:
    1. VGG16 CNN: Extracts 256-dim feature vectors from MRI images
    2. Fuzzy Logic: Maps feature vectors to severity classes using domain rules
    
    This hybrid approach combines:
    - CNN strength: Learns complex visual patterns from data
    - Fuzzy strength: Incorporates medical domain knowledge
    
    Training:
    Phase 1: Train custom head on frozen VGG16 base
    Phase 2: Fine-tune last 4 VGG16 layers with smaller learning rate
    """
    
    def __init__(self):
        """Initialize hybrid model components."""
        self.full_model = None              # Full VGG16 with classification head
        self.feature_extractor = None       # Extracts 256-dim feature vectors
        self.fuzzy_sim = None               # Fuzzy logic inference system
        self.label_map = {
            0: 'Normal',
            1: 'Mild',
            2: 'Moderate',
            3: 'Severe'
        }
        self.is_trained = False
    
    def build(self):
        """
        Build model architecture: VGG16 + custom head + fuzzy system.
        
        Initializes all components:
        1. Build VGG16 with custom classification head
        2. Compile model with optimizer and loss
        3. Build fuzzy inference system
        4. Print model summary
        """
        print("Building VGG16 + Fuzzy Logic Hybrid Model...")
        
        # Build VGG16 feature extractor with custom classification head
        self.full_model = build_vgg16_feature_extractor()
        
        # Compile model
        self.full_model = compile_model(self.full_model)
        
        # Build fuzzy logic system
        self.fuzzy_sim = build_fuzzy_system()
        
        # Print model architecture
        print("\nModel Architecture:")
        self.full_model.summary()
        
        print("\n✓ Model built successfully!")
    
    def train(self, X_train, y_train, X_val, y_val, class_weight=None):
        """
        Train hybrid model in 2 phases.
        
        Phase 1: Train custom classification head (30 epochs)
        - VGG16 base FROZEN (retain ImageNet knowledge)
        - Only train Dense(512) → Dense(256) → Dense(4) layers
        - Learning rate: 0.0001
        
        Phase 2: Fine-tune VGG16 (10 epochs)
        - Unfreeze last 4 VGG16 convolutional layers
        - Allow adaptation to brain MRI domain
        - Learning rate: 0.00001 (10x smaller to avoid destabilizing)
        - Boost accuracy by ~2-3%
        
        Args:
            X_train: Training images (N, 224, 224, 3)
            y_train: Training labels (N,) with values 0-3
            X_val: Validation images (M, 224, 224, 3)
            y_val: Validation labels (M,) with values 0-3
            
        Returns:
            Combined training history from both phases
        """
        print("\n" + "="*60)
        print("  TRAINING HYBRID MODEL")
        print("="*60)
        
        # Phase 1: Train with frozen VGG16 base
        print("\n[Phase 1/2] Training classification head (VGG16 frozen)...")
        self.full_model, history1 = train_model(
            self.full_model, X_train, y_train, X_val, y_val,
            class_weight=class_weight
        )
        
        # Phase 2: Fine-tune VGG16 layers
        print("\n[Phase 2/2] Fine-tuning VGG16 layers...")
        self.full_model, history2 = fine_tune_model(
            self.full_model, X_train, y_train, X_val, y_val,
            class_weight=class_weight
        )
        
        # Build feature extractor from trained model
        # This extracts 256-dim vectors for fuzzy classification
        self.feature_extractor = build_feature_extractor_only(self.full_model)
        
        self.is_trained = True
        
        print("\n✓ Training completed!")
        
        # Combine histories
        return {
            'phase1': history1,
            'phase2': history2
        }
    
    def predict_single(self, image_path):
        """
        Predict severity for a single MRI image.
        
        Process:
        1. Preprocess image through full pipeline
        2. Get CNN classification (4-class softmax)
        3. Extract 256-dim feature vector
        4. Run through fuzzy logic system
        5. Return complete prediction results
        
        Results include:
        - Fuzzy classification (final decision)
        - CNN classification and confidence
        - Fuzzy input values (transparency)
        - Raw fuzzy score (0-3 continuous)
        
        Args:
            image_path: Path to MRI image file
            
        Returns:
            Dict with complete prediction results, or None if preprocessing fails
        """
        # Preprocess image
        processed_image = full_preprocessing_pipeline(image_path)
        
        if processed_image is None:
            print(f"ERROR: Could not preprocess image {image_path}")
            return None
        
        # Reshape to batch of 1
        image_batch = np.expand_dims(processed_image, axis=0)
        
        # Get CNN predictions (4 class probabilities)
        cnn_predictions = self.full_model.predict(image_batch, verbose=0)
        cnn_probabilities = cnn_predictions[0]
        cnn_class = int(np.argmax(cnn_probabilities))
        cnn_confidence = float(np.max(cnn_probabilities))
        
        # Extract 256-dim feature vector
        feature_vector = self.feature_extractor.predict(image_batch, verbose=0)[0]
        
        # Classify using fuzzy system
        fuzzy_class, fuzzy_score, fuzzy_inputs, fuzzy_fallback = fuzzy_classify(
            feature_vector, self.fuzzy_sim
        )
        
        # Combine CNN and fuzzy outputs with a more conservative hybrid strategy
        if cnn_confidence >= 0.75:
            final_class = cnn_class
            final_label = self.label_map[final_class]
            method = 'CNN high confidence'
        elif fuzzy_fallback:
            final_class = cnn_class
            final_label = self.label_map[final_class]
            method = 'CNN fallback'
        elif fuzzy_class == cnn_class:
            final_class = int(fuzzy_class)
            final_label = self.label_map[final_class]
            method = 'Consensus'
        elif fuzzy_score >= 2.6 and cnn_confidence < 0.75:
            final_class = int(fuzzy_class)
            final_label = self.label_map[final_class]
            method = 'Fuzzy strong high'
        elif fuzzy_score <= 0.4 and cnn_confidence < 0.75:
            final_class = int(fuzzy_class)
            final_label = self.label_map[final_class]
            method = 'Fuzzy strong low'
        else:
            final_class = cnn_class
            final_label = self.label_map[final_class]
            method = 'CNN preference'

        # Build result dictionary
        result = {
            'image_path': image_path,
            'fuzzy_class': int(fuzzy_class),
            'fuzzy_label': self.label_map[int(fuzzy_class)],
            'fuzzy_score': float(fuzzy_score),
            'fuzzy_inputs': {
                'tumor_intensity': float(fuzzy_inputs['tumor_intensity']),
                'texture_irregularity': float(fuzzy_inputs['texture_irregularity']),
                'boundary_clarity': float(fuzzy_inputs['boundary_clarity'])
            },
            'cnn_class': int(cnn_class),
            'cnn_label': self.label_map[int(cnn_class)],
            'cnn_confidence': cnn_confidence,
            'cnn_probabilities': {
                'Normal': float(cnn_probabilities[0]),
                'Mild': float(cnn_probabilities[1]),
                'Moderate': float(cnn_probabilities[2]),
                'Severe': float(cnn_probabilities[3])
            },
            'final_class': final_class,
            'final_label': final_label,
            'decision_method': method,
            'fuzzy_fallback': bool(fuzzy_fallback)
        }
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Predict severity for multiple MRI images.
        
        Args:
            image_paths: List of paths to MRI image files
            
        Returns:
            List of prediction result dicts (skips None results)
        """
        results = []
        
        pbar = tqdm(image_paths, desc="Predicting", leave=True)
        for image_path in pbar:
            result = self.predict_single(image_path)
            if result is not None:
                results.append(result)
        
        return results
    
    def evaluate(self, X_test, y_test, use_fuzzy=True):
        """
        Evaluate model on test set, using fuzzy classification (default) or CNN only.

        Args:
            X_test: Test images (N, 224, 224, 3)
            y_test: Test labels (N,) with values 0-3
            use_fuzzy: If True, use fuzzy prediction; otherwise use CNN output

        Returns:
            Tuple: (accuracy, predictions, true_labels)
            - accuracy: Float accuracy percentage (0-1)
            - predictions: Predicted class for each image (0-3)
            - true_labels: True labels for each image
        """
        # Get CNN softmax probabilities for all test images
        cnn_probs = self.full_model.predict(X_test, batch_size=32, verbose=0)
        cnn_preds = np.argmax(cnn_probs, axis=1)

        if not use_fuzzy:
            accuracy = np.mean(cnn_preds == y_test)
            return accuracy, cnn_preds, y_test

        # Extract feature vectors in batches
        feature_vectors = self.feature_extractor.predict(
            X_test, batch_size=32, verbose=0
        )

        # Classify each feature vector with fuzzy system
        predictions = []
        final_predictions = []

        for i in tqdm(range(len(feature_vectors)), desc="Fuzzy Classifying"):
            fuzzy_class, fuzzy_score, _, fuzzy_fallback = fuzzy_classify(
                feature_vectors[i], self.fuzzy_sim
            )
            cnn_class = int(cnn_preds[i])
            cnn_confidence = float(np.max(cnn_probs[i]))

            if fuzzy_fallback:
                predictions.append(int(cnn_class))
            else:
                predictions.append(int(fuzzy_class))

            if cnn_confidence >= 0.75:
                final_predictions.append(cnn_class)
            elif fuzzy_fallback:
                final_predictions.append(cnn_class)
            elif fuzzy_class == cnn_class:
                final_predictions.append(int(fuzzy_class))
            elif fuzzy_score >= 2.6 and cnn_confidence < 0.75:
                final_predictions.append(int(fuzzy_class))
            elif fuzzy_score <= 0.4 and cnn_confidence < 0.75:
                final_predictions.append(int(fuzzy_class))
            else:
                final_predictions.append(cnn_class)

        predictions = np.array(final_predictions)
        accuracy = np.mean(predictions == y_test)

        return accuracy, predictions, y_test
    
    def save(self, path='saved_model/'):
        """
        Save trained models to disk.
        
        Saves:
        - Full model (VGG16 + classification head): vgg16_model.h5
        - Feature extractor (outputs 256-dim vectors): feature_extractor.h5
        
        Args:
            path: Directory where models will be saved
        """
        os.makedirs(path, exist_ok=True)
        
        # Save full model (prefer Keras native format)
        full_model_path = os.path.join(path, 'vgg16_model.keras')
        save_model(self.full_model, full_model_path)
        
        # Save feature extractor
        feature_path = os.path.join(path, 'feature_extractor.keras')
        save_model(self.feature_extractor, feature_path)
        
        print(f"✓ Models saved to {path}")
    
    def load(self, path='saved_model/'):
        """
        Load trained models from disk.
        
        Loads:
        - Full model from vgg16_model.h5
        - Feature extractor from feature_extractor.h5
        - Rebuilds fuzzy system
        
        Args:
            path: Directory where models are saved
        """
        # Load full model
        full_model_path = os.path.join(path, 'vgg16_model.keras')
        if not os.path.exists(full_model_path):
            full_model_path = os.path.join(path, 'vgg16_model.h5')
        self.full_model = load_saved_model(full_model_path)
        
        # Load feature extractor
        feature_path = os.path.join(path, 'feature_extractor.keras')
        if not os.path.exists(feature_path):
            feature_path = os.path.join(path, 'feature_extractor.h5')
        self.feature_extractor = load_saved_model(feature_path)
        
        # Rebuild fuzzy system
        self.fuzzy_sim = build_fuzzy_system()
        
        self.is_trained = True
        
        print(f"✓ Models loaded from {path}")
