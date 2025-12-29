#!/usr/bin/env python3
"""
Train multiple CNN models for ensemble
"""

import numpy as np
import tensorflow as tf
from cnn_model import CNNRecognizer
from cnn_train import CNNTrainer
import os

def train_ensemble(X_train, X_val, y_train, y_val, num_models=5):
    """Train multiple models with different seeds"""
    models = []
    
    for i in range(num_models):
        print(f"\n{'='*50}")
        print(f"Training Model {i+1}/{num_models}")
        print(f"{'='*50}")
        
        # Set random seed for reproducibility
        seed = 42 + i * 100
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Build model
        recognizer = CNNRecognizer()
        model = recognizer.build_cnn_advanced()
        model = recognizer.compile_model(model, learning_rate=0.001)
        
        # Train
        trainer = CNNTrainer(
            model, 
            model_save_path=f'models/cnn_ensemble_{i+1}.h5'
        )
        
        history = trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=30,
            batch_size=64,
            use_augmentation=True
        )
        
        models.append(model)
        
        # Save final accuracy
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Model {i+1} Validation Accuracy: {val_accuracy*100:.2f}%")
    
    return models

# Load your data and call:
# models = train_ensemble(X_train, X_val, y_train, y_val, num_models=5)