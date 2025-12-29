#!/usr/bin/env python3
"""
Compare MLP vs CNN performance
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def compare_models(X_test, y_test):
    """Compare MLP and CNN models"""
    results = {}
    
    # Load MLP model
    try:
        mlp_model = load_model('models/digit_recognizer.h5')
        mlp_loss, mlp_acc = mlp_model.evaluate(X_test, y_test, verbose=0)
        results['MLP'] = {'accuracy': mlp_acc, 'loss': mlp_loss}
        print(f"MLP Accuracy: {mlp_acc*100:.2f}%")
    except:
        print("MLP model not found")
    
    # Load CNN model
    try:
        cnn_model = load_model('models/cnn_digit_recognizer.h5')
        cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
        results['CNN'] = {'accuracy': cnn_acc, 'loss': cnn_loss}
        print(f"CNN Accuracy: {cnn_acc*100:.2f}%")
    except:
        print("CNN model not found")
    
    # Plot comparison
    if len(results) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        models = list(results.keys())
        accuracies = [results[m]['accuracy']*100 for m in models]
        
        bars = ax1.bar(models, accuracies, color=['#ff6b6b', '#4ecdc4'])
        ax1.set_title('Model Accuracy Comparison', fontsize=14)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim([95, 100])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{acc:.2f}%', ha='center', va='bottom')
        
        # Loss comparison
        losses = [results[m]['loss'] for m in models]
        bars2 = ax2.bar(models, losses, color=['#ff6b6b', '#4ecdc4'])
        ax2.set_title('Model Loss Comparison', fontsize=14)
        ax2.set_ylabel('Loss')
        
        for bar, loss in zip(bars2, losses):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{loss:.4f}', ha='center', va='bottom')
        
        plt.suptitle('MLP vs CNN Performance Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return results

# Run with your test data
# results = compare_models(X_val, y_val)