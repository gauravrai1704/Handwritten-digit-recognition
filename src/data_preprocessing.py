import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class DataPreprocessor:
    def __init__(self, train_path, test_path, has_labels_in_test=False):
        """
        Args:
            train_path: Path to training CSV file
            test_path: Path to test CSV file
            has_labels_in_test: Whether test CSV has label column (first column)
        """
        self.train_path = train_path
        self.test_path = test_path
        self.has_labels_in_test = has_labels_in_test
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.X_test = None
        
    def load_data(self):
        """Load training and test data"""
        print("Loading data...")
        train_data = pd.read_csv(self.train_path)
        test_data = pd.read_csv(self.test_path)
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Train columns: {train_data.columns[:5].tolist()}...")
        
        # Separate features and labels for training data
        # Assuming first column is label
        X = train_data.iloc[:, 1:].values  # All columns except first
        y = train_data.iloc[:, 0].values   # First column
        
        # For test data, check if it has labels
        if self.has_labels_in_test and test_data.shape[1] > 784:
            # Test data has labels (like MNIST test set)
            X_test = test_data.iloc[:, 1:].values
            print("✓ Test data has labels column")
        else:
            # Test data has only features (like Kaggle test set)
            X_test = test_data.values
            print("✓ Test data has only features")
        
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"X_test shape: {X_test.shape}")
        
        # Check if dimensions are correct
        if X.shape[1] != 784:
            print(f"⚠️ Warning: Expected 784 features, got {X.shape[1]}")
            # Try to reshape if it's 28x28 flattened
            if X.shape[1] == 28 * 28:
                print("✓ 28x28 flattened images detected")
            else:
                print(f"❌ Unexpected number of features: {X.shape[1]}")
        
        return X, y, X_test
    
    def preprocess(self, X, y, X_test):
        """Preprocess the data"""
        # Normalize pixel values to [0, 1]
        X = X.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Check and fix shape
        if len(X.shape) == 2:
            if X.shape[1] == 784:  # Flattened 28x28 images
                X = X.reshape(-1, 28, 28, 1)
            elif X.shape[1] == 28 * 28:  # Generic 28x28 flattened
                X = X.reshape(-1, 28, 28, 1)
            else:
                print(f"⚠️ Can't reshape X with shape {X.shape} to 28x28")
                # Try to find square dimensions
                import math
                sqrt = int(math.sqrt(X.shape[1]))
                if sqrt * sqrt == X.shape[1]:
                    print(f"✓ Reshaping to {sqrt}x{sqrt}")
                    X = X.reshape(-1, sqrt, sqrt, 1)
                else:
                    raise ValueError(f"Cannot reshape X with {X.shape[1]} features")
        
        # Same for test data
        if len(X_test.shape) == 2:
            if X_test.shape[1] == 784:  # Flattened 28x28 images
                X_test = X_test.reshape(-1, 28, 28, 1)
            elif X_test.shape[1] == 28 * 28:  # Generic 28x28 flattened
                X_test = X_test.reshape(-1, 28, 28, 1)
            else:
                print(f"⚠️ Can't reshape X_test with shape {X_test.shape} to 28x28")
                # Try to find square dimensions
                import math
                sqrt = int(math.sqrt(X_test.shape[1]))
                if sqrt * sqrt == X_test.shape[1]:
                    print(f"✓ Reshaping to {sqrt}x{sqrt}")
                    X_test = X_test.reshape(-1, sqrt, sqrt, 1)
                else:
                    raise ValueError(f"Cannot reshape X_test with {X_test.shape[1]} features")
        
        # One-hot encode labels
        num_classes = len(np.unique(y))
        print(f"Number of unique labels: {num_classes}")
        y = to_categorical(y, num_classes=num_classes)
        
        # Split into training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
        )
        
        print(f"✅ Preprocessing complete:")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   X_val shape: {X_val.shape}")
        print(f"   X_test shape: {X_test.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   y_val shape: {y_val.shape}")
        
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.X_test = X_test
        
        return X_train, X_val, y_train, y_val, X_test
    
    def get_preprocessed_data(self):
        """Main method to get all preprocessed data"""
        X, y, X_test = self.load_data()
        return self.preprocess(X, y, X_test)