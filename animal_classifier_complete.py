"""
ANIMAL IMAGE CLASSIFICATION SYSTEM - COMPLETE TRAINING SCRIPT
==============================================================
Dataset: 15 Animal Classes (224x224x3 images)
Classes: Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, 
         Horse, Kangaroo, Lion, Panda, Tiger, Zebra

Features:
- Transfer Learning with MobileNetV2
- Data Augmentation
- Early Stopping & Model Checkpointing
- Comprehensive Evaluation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class AnimalClassifier:
    def __init__(self, dataset_path, img_size=(224, 224)):
        """
        Initialize the Animal Classification System
        
        Args:
            dataset_path: Path to dataset folder containing 15 animal subfolders
            img_size: Target image size (default: 224x224)
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.classes = [
            'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 
            'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 
            'Panda', 'Tiger', 'Zebra'
        ]
        self.num_classes = len(self.classes)
        self.model = None
        self.history = None
        
    def load_data(self):
        """Load images from directory structure"""
        print("\n" + "="*70)
        print("[STEP 1] LOADING DATASET")
        print("="*70)
        print(f"Path: {self.dataset_path}")
        print(f"Expected Classes: {self.num_classes}")
        print("-"*70)
        
        X = []
        y = []
        class_counts = {}
        
        for class_idx, animal in enumerate(self.classes):
            animal_path = os.path.join(self.dataset_path, animal)
            
            if not os.path.exists(animal_path):
                print(f"⚠ Warning: {animal_path} not found, skipping...")
                continue
                
            images = [f for f in os.listdir(animal_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            class_counts[animal] = len(images)
            print(f"📁 {animal:12s}: {len(images):3d} images")
            
            for img_name in images:
                img_path = os.path.join(animal_path, img_name)
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img)
                    
                    # Verify shape
                    if img_array.shape == (*self.img_size, 3):
                        X.append(img_array)
                        y.append(class_idx)
                    else:
                        print(f"    ⚠ Skipping {img_name}: Invalid shape {img_array.shape}")
                        
                except Exception as e:
                    print(f"    ✗ Error loading {img_path}: {e}")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        print("-"*70)
        print(f"✓ Total images loaded: {len(X)}")
        print(f"✓ Image shape: {X.shape[1:]}")
        print(f"✓ Classes distribution: {dict(class_counts)}")
        
        if len(X) == 0:
            raise ValueError("No images loaded! Check dataset path.")
            
        return X, y
    
    def preprocess_data(self, X, y):
        """Split and preprocess data"""
        print("\n" + "="*70)
        print("[STEP 2] PREPROCESSING DATA")
        print("="*70)
        
        # Split: 70% train, 15% validation, 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Normalize pixel values to [0, 1]
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        X_test = X_test / 255.0
        
        # Convert labels to categorical (one-hot encoding)
        y_train_cat = keras.utils.to_categorical(y_train, self.num_classes)
        y_val_cat = keras.utils.to_categorical(y_val, self.num_classes)
        y_test_cat = keras.utils.to_categorical(y_test, self.num_classes)
        
        print(f"✓ Training set:   {len(X_train):4d} images ({len(X_train)/len(X)*100:.1f}%)")
        print(f"✓ Validation set: {len(X_val):4d} images ({len(X_val)/len(X)*100:.1f}%)")
        print(f"✓ Test set:       {len(X_test):4d} images ({len(X_test)/len(X)*100:.1f}%)")
        print(f"✓ Pixel values normalized to [0, 1]")
        print(f"✓ Labels converted to categorical format")
        
        return (X_train, y_train_cat), (X_val, y_val_cat), (X_test, y_test_cat)
    
    def build_model(self, use_transfer_learning=True, fine_tune=False):
        """
        Build CNN model with optional transfer learning
        
        Args:
            use_transfer_learning: Use MobileNetV2 as base model
            fine_tune: Unfreeze base model layers for fine-tuning
        """
        print("\n" + "="*70)
        print("[STEP 3] BUILDING MODEL")
        print("="*70)
        
        if use_transfer_learning:
            print("✓ Using Transfer Learning (MobileNetV2)")
            print("✓ Base model: MobileNetV2 (ImageNet weights)")
            
            # Load pre-trained MobileNetV2
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3),
                pooling='avg'
            )
            
            # Control fine-tuning
            if fine_tune:
                print("✓ Fine-tuning: Unfreezing top 20 layers of base model")
                base_model.trainable = True
                for layer in base_model.layers[:-20]:
                    layer.trainable = False
            else:
                print("✓ Freezing base model layers")
                base_model.trainable = False
            
            # Build complete model
            model = keras.Sequential([
                base_model,
                layers.Dense(256, activation='relu', name='dense_256'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu', name='dense_128'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax', name='output')
            ])
            
        else:
            print("✓ Building Custom CNN from scratch")
            
            model = keras.Sequential([
                # First Convolutional Block
                layers.Conv2D(32, (3, 3), activation='relu', 
                            input_shape=(*self.img_size, 3), name='conv1'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Second Convolutional Block
                layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Third Convolutional Block
                layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Fourth Convolutional Block
                layers.Conv2D(128, (3, 3), activation='relu', name='conv4'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Classifier Head
                layers.Flatten(),
                layers.Dense(512, activation='relu', name='dense_512'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax', name='output')
            ])
        
        # Compile model
        if fine_tune:
            optimizer = keras.optimizers.Adam(learning_rate=1e-5)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=0.0001)
            
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
        )
        
        self.model = model
        
        print(f"\n✓ Model compiled successfully")
        print(f"✓ Total parameters: {model.count_params():,}")
        print(f"✓ Trainable parameters: {sum([keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        # Print model summary
        print("\n" + "-"*70)
        model.summary()
        print("-"*70)
        
        return model
    
    def train(self, train_data, val_data, epochs=50, batch_size=32, use_augmentation=True):
        """
        Train the model
        
        Args:
            train_data: Tuple (X_train, y_train)
            val_data: Tuple (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size for training
            use_augmentation: Enable data augmentation
        """
        print("\n" + "="*70)
        print("[STEP 4] TRAINING MODEL")
        print("="*70)
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_animal_model.keras',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if use_augmentation:
            print("✓ Data augmentation enabled")
            print("  - Rotation range: 20°")
            print("  - Width/Height shift: 20%")
            print("  - Horizontal flip: enabled")
            print("  - Zoom range: 20%")
            
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                fill_mode='nearest'
            )
            
            train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
            
            self.history = self.model.fit(
                train_generator,
                steps_per_epoch=len(X_train) // batch_size,
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            print("✓ Data augmentation disabled")
            
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        print("\n✓ Training completed!")
        print(f"✓ Best validation accuracy: {max(self.history.history['val_accuracy']):.4f}")
        
        return self.history
    
    def evaluate(self, test_data):
        """Evaluate model performance on test set"""
        print("\n" + "="*70)
        print("[STEP 5] EVALUATING MODEL")
        print("="*70)
        
        X_test, y_test = test_data
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics_names = self.model.metrics_names
        
        print("\nTest Set Performance:")
        for name, value in zip(metrics_names, results):
            print(f"  • {name}: {value:.4f}")
        
        # Detailed predictions
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        print("\n" + "-"*70)
        print("Detailed Classification Report:")
        print("-"*70)
        print(classification_report(
            y_true_classes, y_pred_classes, 
            target_names=self.classes, 
            digits=4
        ))
        
        return y_true_classes, y_pred_classes
    
    def plot_results(self, y_true, y_pred, save_path='training_results.png'):
        """Plot training history and confusion matrix"""
        print("\n" + "="*70)
        print("[STEP 6] GENERATING VISUALIZATIONS")
        print("="*70)
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Training & Validation Accuracy
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(self.history.history['accuracy'], label='Training', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training & Validation Loss
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(self.history.history['loss'], label='Training', linewidth=2, color='orange')
        ax2.plot(self.history.history['val_loss'], label='Validation', linewidth=2, color='red')
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate (if available)
        if 'lr' in self.history.history:
            ax3 = plt.subplot(2, 2, 3)
            ax3.plot(self.history.history['lr'], linewidth=2, color='green')
            ax3.set_title('Learning Rate', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Confusion Matrix
        ax4 = plt.subplot(2, 2, 4)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes,
            ax=ax4,
            cbar_kws={'label': 'Count'}
        )
        ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Predicted Label')
        ax4.set_ylabel('True Label')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax4.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Results saved to '{save_path}'")
        
        # Show per-class accuracy
        print("\nPer-Class Accuracy:")
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i, class_name in enumerate(self.classes):
            accuracy = cm_normalized[i, i]
            print(f"  • {class_name:12s}: {accuracy:.2%}")
        
        plt.show()
    
    def predict_image(self, image_path):
        """
        Predict the animal in a single image
        
        Returns:
            predicted_class: Name of predicted animal
            confidence: Confidence score (0-1)
            top3_predictions: List of (class_name, confidence) tuples
            original_image: PIL Image object
        """
        if self.model is None:
            raise ValueError("Model not loaded! Train or load a model first.")
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize(self.img_size)
        
        # Preprocess for model
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_predictions = [(self.classes[i], float(predictions[i])) for i in top_3_indices]
        
        return self.classes[predicted_idx], float(confidence), top_3_predictions, img
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction with bar chart"""
        prediction, confidence, top3, img = self.predict_image(image_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Display image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(
            f'Predicted: {prediction}\nConfidence: {confidence:.2%}', 
            fontsize=14, 
            fontweight='bold',
            color='#27ae60' if confidence > 0.8 else '#f39c12' if confidence > 0.6 else '#e74c3c'
        )
        
        # Display confidence bars
        classes = [t[0] for t in top3]
        scores = [t[1] for t in top3]
        colors = ['#27ae60', '#3498db', '#95a5a6']
        
        bars = ax2.barh(range(len(classes)), scores, color=colors)
        ax2.set_yticks(range(len(classes)))
        ax2.set_yticklabels(classes)
        ax2.set_xlabel('Confidence Score')
        ax2.set_title('Top 3 Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        
        # Add percentage labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax2.text(
                score + 0.01, 
                i, 
                f'{score:.1%}', 
                va='center', 
                fontweight='bold'
            )
        
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        plt.show()
        
        return prediction, confidence, top3
    
    def save_model(self, model_path='animal_classifier_model.keras', 
                   classes_path='class_names.json'):
        """Save the trained model and class names"""
        if self.model is None:
            raise ValueError("No model to save! Train a model first.")
        
        self.model.save(model_path)
        print(f"\n✓ Model saved to '{model_path}'")
        
        # Save class names
        with open(classes_path, 'w') as f:
            json.dump(self.classes, f)
        print(f"✓ Class names saved to '{classes_path}'")
        
        # Save model info
        info = {
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'classes': self.classes,
            'total_params': self.model.count_params()
        }
        with open('model_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        print(f"✓ Model info saved to 'model_info.json'")
    
    def load_saved_model(self, model_path='animal_classifier_model.keras', 
                        classes_path='class_names.json'):
        """Load a saved model and class names"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = keras.models.load_model(model_path)
        
        with open(classes_path, 'r') as f:
            self.classes = json.load(f)
        self.num_classes = len(self.classes)
        
        print(f"✓ Model loaded from '{model_path}'")
        print(f"✓ Classes loaded: {self.num_classes}")
        print(f"✓ Ready for prediction!")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("ANIMAL IMAGE CLASSIFICATION SYSTEM")
    print("Deep Learning with Transfer Learning")
    print("="*70)
    
    # ========================================================================
    # CONFIGURATION - UPDATE THIS PATH TO YOUR DATASET
    # ========================================================================
    dataset_path = r"C:\Users\VIAMRSH P S\Desktop\Projects-20240722T093004Z-001\Projects\animal_classification\Animal Classification\dataset"
    
    # Training configuration
    USE_TRANSFER_LEARNING = True  # Set False for custom CNN
    FINE_TUNE = False             # Set True to unfreeze base model (slower but better)
    EPOCHS = 30
    BATCH_SIZE = 32
    USE_AUGMENTATION = True
    
    print(f"\nConfiguration:")
    print(f"  Dataset Path: {dataset_path}")
    print(f"  Transfer Learning: {USE_TRANSFER_LEARNING}")
    print(f"  Fine Tuning: {FINE_TUNE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Data Augmentation: {USE_AUGMENTATION}")
    
    try:
        # Initialize classifier
        classifier = AnimalClassifier(dataset_path)
        
        # Load data
        X, y = classifier.load_data()
        
        # Preprocess
        train_data, val_data, test_data = classifier.preprocess_data(X, y)
        
        # Build model
        classifier.build_model(
            use_transfer_learning=USE_TRANSFER_LEARNING,
            fine_tune=FINE_TUNE
        )
        
        # Train
        classifier.train(
            train_data, 
            val_data, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE,
            use_augmentation=USE_AUGMENTATION
        )
        
        # Evaluate
        y_true, y_pred = classifier.evaluate(test_data)
        
        # Visualize results
        classifier.plot_results(y_true, y_pred)
        
        # Save model
        classifier.save_model()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("1. Model saved as 'animal_classifier_model.keras'")
        print("2. Run 'python animal_predictor.py' to classify new images")
        print("3. Or use: classifier.visualize_prediction('path/to/image.jpg')")
        
        # Return trained classifier for immediate use
        return classifier
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        raise


if __name__ == "__main__":
    classifier = main()