import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

class ModelTrainer:
    def __init__(self, model, model_save_path='models/digit_recognizer.h5'):
        self.model = model
        self.model_save_path = model_save_path
        self.history = None
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    def get_callbacks(self):
        """Define training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=30, batch_size=32, verbose=1):
        """Train the model"""
        callbacks = self.get_callbacks()
        
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history
        return history
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        history = self.history.history
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, path=None):
        """Save the trained model"""
        if path is None:
            path = self.model_save_path
        
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path=None):
        """Load a trained model"""
        from tensorflow.keras.models import load_model
        
        if path is None:
            path = self.model_save_path
        
        if os.path.exists(path):
            self.model = load_model(path)
            print(f"Model loaded from {path}")
            return self.model
        else:
            print(f"No model found at {path}")
            return None