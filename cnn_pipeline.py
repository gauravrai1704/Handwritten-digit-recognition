import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import sys

# Add src to path
sys.path.append('src')

from cnn_model import CNNRecognizer
from cnn_train import CNNTrainer
from data_augmentation import DataAugmentor

def load_and_preprocess_data():
    """Load and preprocess MNIST data for CNN"""
    print("📊 Loading and preprocessing data for CNN...")
    
    # Load data (adjust paths as needed)
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Separate features and labels
    X = train_data.iloc[:, 1:].values
    y = train_data.iloc[:, 0].values
    
    # Check if test data has labels
    if test_data.shape[1] > 784:
        X_test = test_data.iloc[:, 1:].values
    else:
        X_test = test_data.values
    
    # Reshape and normalize
    X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encode labels
    y = to_categorical(y, num_classes=10)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    print(f"\n✅ Data preprocessing complete:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_val shape: {X_val.shape}")
    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   y_val shape: {y_val.shape}")
    
    return X_train, X_val, y_train, y_val, X_test

def visualize_sample_images(X_train, y_train, num_samples=16):
    """Visualize sample training images"""
    plt.figure(figsize=(10, 10))
    
    for i in range(num_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {np.argmax(y_train[i])}")
        plt.axis('off')
    
    plt.suptitle('Sample Training Images', fontsize=16)
    plt.tight_layout()
    plt.show()

def build_and_train_cnn(X_train, X_val, y_train, y_val, use_augmentation=True):
    """Build and train CNN model"""
    print("\n🏗️  Building CNN model...")
    
    # Initialize CNN recognizer
    cnn_recognizer = CNNRecognizer(input_shape=(28, 28, 1))
    
    # Choose CNN architecture
    print("\nSelect CNN architecture:")
    print("1. Simple CNN (faster training)")
    print("2. Advanced CNN (higher accuracy, target: 99.5%)")
    print("3. Efficient CNN (balanced)")
    
    try:
        choice = input("Enter choice (1-3, default 2): ").strip()
        if choice == '1':
            model = cnn_recognizer.build_cnn_simple()
            print("✓ Building simple CNN...")
        elif choice == '3':
            model = cnn_recognizer.build_efficient_cnn()
            print("✓ Building efficient CNN...")
        else:
            model = cnn_recognizer.build_cnn_advanced()
            print("✓ Building advanced CNN (target: 99.5% accuracy)...")
    except:
        model = cnn_recognizer.build_cnn_advanced()
        print("✓ Building advanced CNN (default)...")
    
    # Compile model
    model = cnn_recognizer.compile_model(model, learning_rate=0.001)
    
    # Print model summary
    print("\n📋 Model Summary:")
    model.summary()
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\n📊 Total parameters: {total_params:,}")
    
    # Initialize trainer
    trainer = CNNTrainer(model, model_save_path='models/cnn_digit_recognizer.h5')
    
    # Data augmentation
    augmentor = None
    if use_augmentation:
        print("\n🔧 Setting up data augmentation...")
        augmentor = DataAugmentor(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        
        # Visualize augmentations
        try:
            augmentor.visualize_augmentations(X_train[:1])
        except:
            print("⚠️  Could not visualize augmentations")
    
    # Training parameters
    epochs = 50
    batch_size = 64
    
    print(f"\n🚀 Starting CNN training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Using augmentation: {use_augmentation}")
    
    # Train model
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        use_augmentation=use_augmentation,
        augmentor=augmentor
    )
    
    return trainer, history, model

def evaluate_cnn_model(trainer, X_val, y_val):
    """Evaluate CNN model performance"""
    print("\n📊 Evaluating CNN model...")
    
    # Evaluate on validation set
    loss, accuracy, predictions = trainer.evaluate_model(X_val, y_val)
    
    print(f"\n🎯 CNN Model Performance:")
    print(f"   Validation Accuracy: {accuracy*100:.2f}%")
    print(f"   Validation Loss: {loss:.4f}")
    
    # Plot training history
    trainer.plot_training_history()
    
    # Get predictions
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_val, axis=1)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=range(10), yticklabels=range(10))
    plt.title('CNN Confusion Matrix', fontsize=16)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Analyze errors
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    if len(misclassified_idx) > 0:
        print(f"\n❌ Misclassifications: {len(misclassified_idx)}/{len(y_true)} "
              f"({len(misclassified_idx)/len(y_true)*100:.2f}%)")
        
        # Show some misclassifications
        num_samples = min(6, len(misclassified_idx))
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, idx in enumerate(misclassified_idx[:num_samples]):
            axes[i].imshow(X_val[idx].reshape(28, 28), cmap='gray')
            axes[i].set_title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}")
            axes[i].axis('off')
        
        plt.suptitle('CNN Misclassifications', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    return accuracy, predictions

def predict_on_test_set(model, X_test):
    """Make predictions on test set"""
    print("\n🔮 Making predictions on test set...")
    
    predictions = model.predict(X_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Visualize some predictions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(10):
        axes[i].imshow(X_test[i].reshape(28, 28), cmap='gray')
        confidence = np.max(predictions[i]) * 100
        axes[i].set_title(f"Pred: {predicted_labels[i]}\nConf: {confidence:.1f}%")
        axes[i].axis('off')
    
    plt.suptitle('CNN Test Set Predictions', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return predicted_labels

def save_results_and_model(trainer, accuracy, predicted_labels):
    """Save results and create submission"""
    # Save model
    trainer.save_model()
    
    # Create submission file
    submission = pd.DataFrame({
        'ImageId': range(1, len(predicted_labels) + 1),
        'Label': predicted_labels
    })
    
    submission.to_csv('submission_cnn.csv', index=False)
    print(f"\n✅ Submission file created: submission_cnn.csv")
    
    # Save accuracy report
    with open('results/cnn_accuracy.txt', 'w') as f:
        f.write(f"CNN Model Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Total test predictions: {len(predicted_labels)}\n")
    
    print(f"📊 Accuracy report saved: results/cnn_accuracy.txt")

def main():
    print("=" * 60)
    print("           CNN PIPELINE - HANDWRITTEN DIGIT RECOGNITION")
    print("                    Target Accuracy: 99.5%")
    print("=" * 60)
    
    try:
        # Step 1: Load and preprocess data
        X_train, X_val, y_train, y_val, X_test = load_and_preprocess_data()
        
        # Step 2: Visualize samples
        visualize_sample_images(X_train, y_train)
        
        # Step 3: Build and train CNN
        trainer, history, model = build_and_train_cnn(X_train, X_val, y_train, y_val)
        
        # Step 4: Evaluate model
        accuracy, predictions = evaluate_cnn_model(trainer, X_val, y_val)
        
        # Step 5: Make predictions on test set
        predicted_labels = predict_on_test_set(model, X_test)
        
        # Step 6: Save results
        save_results_and_model(trainer, accuracy, predicted_labels)
        
        print("\n" + "=" * 60)
        print("🎉 CNN PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        if accuracy >= 0.99:
            print(f"\n🏆 EXCELLENT! Achieved {accuracy*100:.2f}% accuracy! 🏆")
        elif accuracy >= 0.985:
            print(f"\n👍 Great! Achieved {accuracy*100:.2f}% accuracy!")
        else:
            print(f"\n📈 Good start! Achieved {accuracy*100:.2f}% accuracy.")
            print("   Try training for more epochs or tuning hyperparameters.")
        
        print(f"\n📁 Model saved: models/cnn_digit_recognizer.h5")
        print(f"📄 Submission: submission_cnn.csv")
        print(f"📊 Results: results/cnn_accuracy.txt")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()