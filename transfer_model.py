"""
VGG16 Transfer Learning Model
Builds a VGG16 feature extractor with custom classification head for brain MRI severity
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, 
                                     BatchNormalization, Dropout, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                        ModelCheckpoint)
import os


def build_vgg16_feature_extractor():
    """
    Build VGG16 transfer learning model with custom classification head.
    
    VGG16 Architecture:
    - Pre-trained on ImageNet with millions of images
    - Extract only the convolutional base (remove top classification layers)
    - Freeze all VGG16 weights (transfer learning)
    - Add custom top classification layers:
      * GlobalAveragePooling2D (reduces spatial dimensions)
      * Dense(512) with BatchNorm and Dropout
      * Dense(256) with BatchNorm and Dropout (feature vector layer)
      * Dense(4) softmax (classify into 4 severity classes)
    
    The feature extraction power comes from the 14M+ ImageNet-trained parameters.
    We only train the top custom layers initially.
    
    Returns:
        Compiled Keras Model
        - Input: (224, 224, 3) RGB images
        - Output: 4-class probability distribution over [Normal, Mild, Moderate, Severe]
    """
    # Load VGG16 WITHOUT the top classification layers
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(224, 224, 3))
    
    # FREEZE all VGG16 base layers - we don't train the ImageNet features
    base_model.trainable = False
    
    # Build custom classification head on top of VGG16 base
    x = base_model.output
    
    # Global pooling to reduce spatial dimensions (7x7x512 -> 512)
    x = GlobalAveragePooling2D()(x)
    
    # First dense layer with high capacity (512 neurons)
    x = Dense(512, activation='relu', name='dense_0')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Feature vector layer (256 dimensions)
    # This layer outputs the feature vectors for fuzzy logic system
    x = Dense(256, activation='relu', name='dense_1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer: 4 severity classes with softmax probability
    output = Dense(4, activation='softmax', name='output')(x)
    
    # Create complete model
    model = Model(inputs=base_model.input, outputs=output)
    
    return model


def build_feature_extractor_only(trained_full_model):
    """
    Extract feature vector generation model from fully trained model.
    
    Uses the Dense(256) layer (named 'dense_1') as the output layer.
    This extracts 256-dimensional feature vectors for fuzzy classification.
    
    The complete pipeline is:
    VGG16 base (14M params) → Dense(512) → Dense(256) → Fuzzy Logic System
                                                    ↑
                                            Extract here
    
    Args:
        trained_full_model: Fully trained VGG16 model from train_model()
        
    Returns:
        New Model that outputs 256-dim feature vectors
        - Input: (224, 224, 3) RGB images
        - Output: (256,) feature vectors
    """
    # Find and get the Dense(256) feature vector layer
    # It's named 'dense_1' during model construction
    feature_layer = trained_full_model.get_layer('dense_1')
    
    # Create new model from input to feature layer
    feature_extractor = Model(inputs=trained_full_model.input,
                              outputs=feature_layer.output)
    
    return feature_extractor


def compile_model(model):
    """
    Compile model with optimizer, loss function, and metrics.
    
    Configuration:
    - Optimizer: Adam with lr=0.0001 (good initial learning rate)
    - Loss: sparse_categorical_crossentropy (for integer labels)
    - Metrics: accuracy to monitor training
    
    Args:
        model: Uncompiled Keras Model
        
    Returns:
        Compiled model ready for training
    """
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, class_weight=None):
    """
    Train the custom head on frozen VGG16 base (Phase 1 of training).
    
    Training strategy:
    - Frozen VGG16 base prevents overfitting to small skull-stripped MRI dataset
    - Learn only the custom classification head (512 → 256 → 4)
    - Use early stopping with validation patience of 7 epochs
    - Reduce learning rate if validation loss plateaus
    - Save best model based on validation accuracy
    
    Callbacks:
    - EarlyStopping: Stop if val_accuracy doesn't improve for 7 epochs
    - ReduceLROnPlateau: Reduce LR by 50% when val_loss plateaus after 3 epochs
    - ModelCheckpoint: Save best model based on highest val_accuracy
    
    Args:
        model: Compiled Keras model with frozen VGG16 base
        X_train: Training images (N, 224, 224, 3)
        y_train: Training labels (N,) with values 0-3
        X_val: Validation images (M, 224, 224, 3)
        y_val: Validation labels (M,) with values 0-3
        
    Returns:
        Trained model and history object with training curves
    """
    # Define callbacks for training
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
    
    checkpoint = ModelCheckpoint(
        'saved_model/best_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    )
    
    # Train for max 30 epochs (early stopping will likely stop sooner)
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=30,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    return model, history


def fine_tune_model(model, X_train, y_train, X_val, y_val, class_weight=None):
    """
    Fine-tune model by unfreezing last 4 VGG16 layers (Phase 2 of training).
    
    After training the custom head thoroughly, we unfreeze the top VGG16 layers
    and continue training with a MUCH smaller learning rate (10x smaller).
    
    This allows the model to adapt VGG16's ImageNet features to brain MRI domain.
    Fine-tuning typically boosts accuracy by 2-3%.
    
    Last 4 VGG16 layers are the highest-level semantic features (closest to output).
    Unfreezing earlier layers could degrade ImageNet features.
    
    Args:
        model: Already trained model from train_model()
        X_train: Training images (N, 224, 224, 3)
        y_train: Training labels (N,) with values 0-3
        X_val: Validation images (M, 224, 224, 3)
        y_val: Validation labels (M,) with values 0-3
        
    Returns:
        Fine-tuned model and history object
    """
    # Unfreeze last 4 layers of VGG16
    for layer in model.layers[-4:]:
        layer.trainable = True
    
    # Recompile with much smaller learning rate (10x smaller than initial)
    model.compile(
        optimizer=Adam(learning_rate=0.00001),  # 10x smaller than 0.0001
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune for 10 more epochs
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        verbose=1
    )
    
    return model, history


def save_model(model, path='saved_model/vgg16_model.h5'):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained Keras model to save
        path: File path where model will be saved
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")


def load_saved_model(path='saved_model/vgg16_model.h5'):
    """
    Load a saved model from disk.
    
    Args:
        path: Path to saved model file
        
    Returns:
        Loaded Keras model
    """
    model = tf.keras.models.load_model(path)
    print(f"Model loaded from {path}")
    return model
