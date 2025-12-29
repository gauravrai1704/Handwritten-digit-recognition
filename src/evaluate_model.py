import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

class ModelEvaluator:
    def __init__(self, model):
        self.model = model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        
        return loss, accuracy
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        
        return predictions, predicted_labels
    
    def plot_confusion_matrix(self, y_true, y_pred, classes=range(10)):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def get_classification_report(self, y_true, y_pred, classes=range(10)):
        """Get detailed classification report"""
        report = classification_report(y_true, y_pred, 
                                      target_names=[str(i) for i in classes])
        print(report)
        return report
    
    def analyze_misclassifications(self, X, y_true, y_pred, num_samples=5):
        """Analyze misclassified samples"""
        misclassified_idx = np.where(y_true != y_pred)[0]
        
        if len(misclassified_idx) == 0:
            print("No misclassifications found!")
            return
        
        print(f"Total misclassifications: {len(misclassified_idx)}")
        print(f"Misclassification rate: {len(misclassified_idx)/len(y_true)*100:.2f}%")
        
        # Plot some misclassified samples
        num_samples = min(num_samples, len(misclassified_idx))
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        
        if num_samples == 1:
            axes = [axes]
        
        for i, idx in enumerate(misclassified_idx[:num_samples]):
            img = X[idx].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return misclassified_idx