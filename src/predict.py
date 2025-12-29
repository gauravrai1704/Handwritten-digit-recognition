import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

class DigitPredictor:
    def __init__(self, model_path='models/digit_recognizer.h5'):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except:
            print(f"Could not load model from {self.model_path}")
            print("Please train the model first or check the path.")
    
    def predict_digit(self, image_array):
        """Predict digit from image array"""
        if self.model is None:
            print("Model not loaded. Cannot make predictions.")
            return None
        
        # Preprocess the image
        if len(image_array.shape) == 2:
            image_array = image_array.reshape(1, 28, 28, 1)
        
        # Normalize
        image_array = image_array.astype('float32') / 255.0
        
        # Make prediction
        prediction = self.model.predict(image_array, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_digit, confidence, prediction
    
    def predict_batch(self, X_test):
        """Predict digits for a batch of images"""
        if self.model is None:
            print("Model not loaded. Cannot make predictions.")
            return None
        
        predictions = self.model.predict(X_test, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        
        return predictions, predicted_labels
    
    def visualize_predictions(self, X_test, predictions, num_samples=5):
        """Visualize predictions"""
        predicted_labels = np.argmax(predictions, axis=1)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            img = X_test[i].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Predicted: {predicted_labels[i]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def process_external_image(self, image_path):
        """Process an external image for prediction"""
        # Load and preprocess external image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Could not load image from {image_path}")
            return None
        
        # Resize to 28x28
        img = cv2.resize(img, (28, 28))
        
        # Invert colors if needed (white digit on black background)
        img = 255 - img
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        return img