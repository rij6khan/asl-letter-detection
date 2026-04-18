"""
ASL Project — Step 1: Train on Sign Language MNIST
=====================================================
Run this FIRST. It teaches the AI the basics using
the MNIST CSV dataset from Kaggle.

Before running, download from Kaggle:
  sign_mnist_train.csv
  sign_mnist_test.csv
And place them in the same folder as this script.

Then run: python step1_train_mnist.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0=all, 1=info, 2=warnings, 3=errors only
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt


print("=" * 50)
print("  ASL Model Training — Step 1 (MNIST)")
print("=" * 50)

# --------------------------------------------------
# CHECK FILES EXIST
# --------------------------------------------------
if not os.path.exists('sign_mnist_train.csv'):
    print("\n ERROR: 'sign_mnist_train.csv' not found!")
    print("Please download from:")
    print("https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
    exit(1)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
print("\n Loading MNIST data...")
train_df = pd.read_csv('sign_mnist_train.csv')
test_df  = pd.read_csv('sign_mnist_test.csv')
print(f"  Training samples: {len(train_df):,}")
print(f"  Test samples:     {len(test_df):,}")

# --------------------------------------------------
# PREPARE FEATURES AND LABELS
# --------------------------------------------------
y_train = train_df['label'].values
y_test  = test_df['label'].values
X_train = train_df.drop('label', axis=1).values
X_test  = test_df.drop('label', axis=1).values

# Normalize 0-255 → 0.0-1.0
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# Reshape flat 784 → 28×28×1
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
lb = LabelBinarizer()
y_train_enc = lb.fit_transform(y_train)
y_test_enc  = lb.transform(y_test)
num_classes = y_train_enc.shape[1]

print(f"\n Image shape: {X_train.shape[1:]}")
print(f"  Number of classes: {num_classes}")
print(f"  Classes: {list(lb.classes_)}")

# --------------------------------------------------
# BUILD MODEL
# --------------------------------------------------
print("\n  Building neural network...")

model = keras.Sequential([
    # Block 1
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),

    # Block 2
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),

    # Block 3
    keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),

    # Dense head
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),

    # Output
    keras.layers.Dense(num_classes, activation='softmax')
], name='ASL_MNIST_Model')

model.summary()

# --------------------------------------------------
# COMPILE
# --------------------------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------------------------------
# DATA AUGMENTATION
# --------------------------------------------------
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
print("\n Starting training (this may take 5-10 minutes)...\n")

history = model.fit(
    datagen.flow(X_train, y_train_enc, batch_size=64),
    epochs=30,
    validation_data=(X_test, y_test_enc),
    callbacks=callbacks,
    steps_per_epoch=len(X_train) // 64
)

# --------------------------------------------------
# EVALUATE
# --------------------------------------------------
print("\n Evaluating on test data...")
loss, accuracy = model.evaluate(X_test, y_test_enc, verbose=0)
print(f"\n Test Accuracy: {accuracy*100:.2f}%")

# --------------------------------------------------
# SAVE FINAL MODEL
# --------------------------------------------------
model.save('asl_model_mnist.h5')
print(" Model saved: asl_model_mnist.h5")

# Save label classes so we can use them later
np.save('label_classes.npy', lb.classes_)
print(" Labels saved: label_classes.npy")

# --------------------------------------------------
# PLOT TRAINING HISTORY
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Accuracy Over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Loss Over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print(" Graph saved: training_history.png")

print("\n Step 1 complete! Now run: python step2_train_real_images.py")