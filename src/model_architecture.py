from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

class DigitRecognizer:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_simple_model(self):
        """Build a simple feedforward neural network"""
        model = Sequential([
            Input(shape=self.input_shape),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_improved_model(self):
        """Build an improved model with regularization"""
        model = Sequential([
            Input(shape=self.input_shape),
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, model=None, optimizer='adam', 
                     loss='categorical_crossentropy', metrics=['accuracy']):
        """Compile the model"""
        if model is None:
            model = self.model
            
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def summary(self):
        """Print model summary"""
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")