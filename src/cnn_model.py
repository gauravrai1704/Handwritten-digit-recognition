from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

class CNNRecognizer:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_cnn_simple(self):
        """Simple CNN architecture - good starting point"""
        model = Sequential([
            # First Conv Block
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                   input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Conv Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Conv Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.25),
            
            # Dense Layers
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_cnn_advanced(self):
        """Advanced CNN with more features for higher accuracy"""
        model = Sequential([
            # First Conv Block
            Conv2D(32, (5, 5), activation='relu', padding='same', 
                   input_shape=self.input_shape, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(32, (5, 5), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Conv Block
            Conv2D(64, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2), strides=(2, 2)),
            Dropout(0.25),
            
            # Third Conv Block
            Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2), strides=(2, 2)),
            Dropout(0.25),
            
            # Dense Layers
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_efficient_cnn(self):
        """Efficient CNN with fewer parameters but good performance"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                   input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.2),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.4),
            
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, model=None, learning_rate=0.001):
        """Compile CNN model with Adam optimizer"""
        if model is None:
            model = self.model
        
        optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self):
        """Get callbacks for training"""
        callbacks = [
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def summary(self):
        """Print model summary"""
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_cnn_*() first.")