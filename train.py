"""
Bird Species Classification Model - Training Script
Train a CNN model to classify 50 bird species
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime

# ============== CONFIGURATION ==============
DATA_DIR = 'bird_dataset'  # Your dataset folder
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
NUM_CLASSES = 50  # Number of bird species

# Model save path
MODEL_NAME = f'bird_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras'
BEST_MODEL = 'best_bird_model.keras'

print("=" * 50)
print("Bird Species Classification Training")
print("=" * 50)
print(f"Image Size: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Classes: {NUM_CLASSES}")
print("=" * 50)

# ============== DATA AUGMENTATION ==============
print("\n[1/6] Setting up data augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# ============== LOAD DATA ==============
print("[2/6] Loading training data...")

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Save class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}
np.save('class_labels.npy', class_labels)
print(f"Found {len(class_labels)} classes")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# ============== BUILD MODEL ==============
print("\n[3/6] Building model architecture...")

def create_model(num_classes):
    """Create CNN model for bird classification"""
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 5
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

model = create_model(NUM_CLASSES)
model.summary()

# ============== COMPILE MODEL ==============
print("\n[4/6] Compiling model...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
)

# ============== CALLBACKS ==============
print("[5/6] Setting up callbacks...")

callbacks = [
    # Save best model
    ModelCheckpoint(
        BEST_MODEL,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Early stopping
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# ============== TRAIN MODEL ==============
print("\n[6/6] Starting training...")
print("This will take 2-6 hours depending on your hardware.")
print("=" * 50)

start_time = datetime.now()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

end_time = datetime.now()
training_time = end_time - start_time

print("\n" + "=" * 50)
print("TRAINING COMPLETED!")
print("=" * 50)
print(f"Training time: {training_time}")
print(f"Best model saved as: {BEST_MODEL}")
print(f"Final model saved as: {MODEL_NAME}")

# Save final model
model.save(MODEL_NAME)

# ============== PLOT RESULTS ==============
print("\nGenerating training plots...")

plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Top-5 Accuracy plot
plt.subplot(1, 3, 3)
plt.plot(history.history['top_5_accuracy'], label='Training Top-5')
plt.plot(history.history['val_top_5_accuracy'], label='Validation Top-5')
plt.title('Top-5 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("Training plots saved as: training_history.png")

# ============== FINAL METRICS ==============
print("\n" + "=" * 50)
print("FINAL METRICS")
print("=" * 50)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
print("=" * 50)

print("\n✓ Training complete! Now run predict.py to test your model.")