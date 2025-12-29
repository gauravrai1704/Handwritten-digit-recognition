import numpy as np
from tensorflow.keras.models import load_model

class ModelEnsemble:
    def __init__(self, model_paths):
        self.models = []
        for path in model_paths:
            self.models.append(load_model(path))
    
    def predict(self, X):
        """Make ensemble predictions"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        # Average predictions
        avg_predictions = np.mean(predictions, axis=0)
        
        return avg_predictions
    
    def predict_with_voting(self, X):
        """Hard voting ensemble"""
        all_predictions = []
        
        for model in self.models:
            pred = model.predict(X, verbose=0)
            pred_labels = np.argmax(pred, axis=1)
            all_predictions.append(pred_labels)
        
        # Stack predictions
        all_predictions = np.array(all_predictions)
        
        # Vote for each sample
        final_predictions = []
        for sample_preds in all_predictions.T:
            values, counts = np.unique(sample_preds, return_counts=True)
            final_predictions.append(values[np.argmax(counts)])
        
        return np.array(final_predictions)