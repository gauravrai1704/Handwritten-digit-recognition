import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import os
import time

class CNNTrainer:
    def __init__(self, model, model_save_path='models/cnn_digit_recognizer.h5'):
        self.model = model
        self.model_save_path = model_save_path
        self.history = None
        
        # Create directories
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results/cnn', exist_ok=True)
    
    def get_callbacks(self, patience=15, monitor='val_accuracy'):
        """Get advanced callbacks for CNN training"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        callbacks = [
            # Reduce learning rate when validation accuracy plateaus
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=5,
                min_lr=0.000001,
                verbose=1,
                mode='max'
            ),
            
            # Early stopping
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            # Save best model
            ModelCheckpoint(
                filepath=self.model_save_path,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            ),
            
            # Log training to CSV
            CSVLogger(f'logs/training_{timestamp}.csv'),
            
            # TensorBoard callback (optional)
            # TensorBoard(log_dir=f'logs/tensorboard_{timestamp}', histogram_freq=1)
        ]
        
        return callbacks
    
    def train_with_augmentation(self, X_train, y_train, X_val, y_val, 
                                augmentor, epochs=50, batch_size=64):
        """Train with data augmentation"""
        # Create data generators
        train_generator = augmentor.augment_data(X_train, y_train, batch_size)
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // batch_size
        
        print(f"Training with data augmentation...")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Batch size: {batch_size}")
        
        # Train model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        self.history = history
        return history
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=64, use_augmentation=False, augmentor=None):
        """Train the CNN model"""
        if use_augmentation and augmentor:
            return self.train_with_augmentation(X_train, y_train, X_val, y_val, 
                                               augmentor, epochs, batch_size)
        
        print(f"Training without augmentation...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}")
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        self.history = history
        return history
    
    def plot_training_history(self, save_path='results/cnn/training_history.png'):
        """Plot comprehensive training history"""
        if self.history is None:
            print("No training history available.")
            return
        
        history = self.history.history
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot learning rate
        if 'lr' in history:
            ax3.plot(history['lr'], label='Learning Rate')
            ax3.set_title('Learning Rate Schedule', fontsize=14)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # Plot accuracy difference
        if len(history['accuracy']) == len(history['val_accuracy']):
            diff = np.array(history['accuracy']) - np.array(history['val_accuracy'])
            ax4.plot(diff, label='Train-Val Difference', color='red')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax4.set_title('Train-Validation Gap', fontsize=14)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy Difference')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('CNN Training History', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model on test data"""
        print("Evaluating CNN model...")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        
        # Predict and get confidence scores
        predictions = self.model.predict(X_test, verbose=0)
        
        return loss, accuracy, predictions
    
    def save_model(self, path=None):
        """Save the trained model"""
        if path is None:
            path = self.model_save_path
        
        self.model.save(path)
        print(f"CNN model saved to {path}")
    
    def load_model(self, path=None):
        """Load a trained model"""
        from tensorflow.keras.models import load_model
        
        if path is None:
            path = self.model_save_path
        
        if os.path.exists(path):
            self.model = load_model(path)
            print(f"CNN model loaded from {path}")
            return self.model
        else:
            print(f"No model found at {path}")
            return None