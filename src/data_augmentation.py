import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class DataAugmentor:
    def __init__(self, rotation_range=10, zoom_range=0.1, 
                 width_shift_range=0.1, height_shift_range=0.1):
        self.datagen = ImageDataGenerator(
            rotation_range=rotation_range,
            zoom_range=zoom_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            fill_mode='nearest'
        )
    
    def augment_data(self, X, y, batch_size=32):
        """Create augmented data generator"""
        return self.datagen.flow(X, y, batch_size=batch_size, shuffle=True)
    
    def visualize_augmentations(self, X_sample, num_augmentations=9):
        """Visualize augmented images"""
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        
        # Take one image and augment it
        img = X_sample[np.random.randint(0, len(X_sample))]
        img = img.reshape(1, 28, 28, 1)
        
        i = 0
        for batch in self.datagen.flow(img, batch_size=1):
            ax = axes[i // 3, i % 3]
            ax.imshow(batch[0].reshape(28, 28), cmap='gray')
            ax.axis('off')
            ax.set_title(f'Augmentation {i+1}')
            
            i += 1
            if i >= num_augmentations:
                break
        
        plt.suptitle('Data Augmentation Examples', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def get_augmented_dataset(self, X_train, y_train, augmentation_factor=2):
        """Create augmented dataset"""
        # Generate augmented data
        augmented_images = []
        augmented_labels = []
        
        for img, label in zip(X_train, y_train):
            img = img.reshape(1, 28, 28, 1)
            label = label.reshape(1, -1)
            
            # Generate augmented versions
            for _ in range(augmentation_factor):
                for batch in self.datagen.flow(img, label, batch_size=1):
                    augmented_images.append(batch[0][0])
                    augmented_labels.append(batch[1][0])
                    break
        
        # Convert to arrays
        X_augmented = np.array(augmented_images)
        y_augmented = np.array(augmented_labels)
        
        # Combine with original
        X_combined = np.concatenate([X_train, X_augmented])
        y_combined = np.concatenate([y_train, y_augmented])
        
        # Shuffle
        indices = np.random.permutation(len(X_combined))
        X_combined = X_combined[indices]
        y_combined = y_combined[indices]
        
        print(f"Original dataset: {len(X_train)} samples")
        print(f"Augmented dataset: {len(X_augmented)} samples")
        print(f"Combined dataset: {len(X_combined)} samples")
        
        return X_combined, y_combined