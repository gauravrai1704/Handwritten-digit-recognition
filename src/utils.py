import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def create_submission(predictions, output_path='submission.csv'):
    """Create submission file for Kaggle"""
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission file created: {output_path}")
    return submission

def visualize_dataset(X, y, num_samples=10):
    """Visualize samples from the dataset"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = X[i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {np.argmax(y[i]) if len(y[i].shape) > 0 else y[i]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def check_data_distribution(y):
    """Check the distribution of labels"""
    if len(y.shape) > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y
    
    unique, counts = np.unique(y_labels, return_counts=True)
    
    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts)
    plt.title('Distribution of Digits in Dataset')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.xticks(range(10))
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Label distribution:")
    for digit, count in zip(unique, counts):
        print(f"Digit {digit}: {count} samples ({count/len(y_labels)*100:.1f}%)")

def save_results(history, test_accuracy, save_dir='results'):
    """Save training results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training history as CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'{save_dir}/training_history.csv', index=False)
    
    # Save test results
    with open(f'{save_dir}/test_results.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
        f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%\n")
        f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%\n")
    
    print(f"Results saved to {save_dir}/")