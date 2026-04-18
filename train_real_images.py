"""
ASL Project — Step 2: Fine-tune with Real ASL Photos
======================================================
Run this AFTER step1_train_mnist.py.
It takes the already-trained model and makes it smarter
using real hand photos from the asl_dataset folder.

Run: python train_real_images.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # hide TF startup messages

import numpy as np
import tensorflow as tf
import keras
from PIL import Image

print("=" * 50)
print("  ASL Model Training - Step 2 (Real Photos)")
print("=" * 50)

DATASET_PATH = 'asl_dataset'
IMAGE_SIZE   = (28, 28)
BATCH_SIZE   = 32
EPOCHS       = 25

# --------------------------------------------------
# CHECK PREREQUISITES
# --------------------------------------------------
if not os.path.exists('best_model.h5'):
    print("\n ERROR: 'best_model.h5' not found!")
    print("Please run train.py first.")
    exit(1)

if not os.path.exists(DATASET_PATH):
    print(f"\n ERROR: '{DATASET_PATH}' folder not found!")
    exit(1)

# --------------------------------------------------
# LOAD REAL PHOTO DATASET
# --------------------------------------------------
print(f"\n Loading images from '{DATASET_PATH}'...")

images = []
labels = []

# Get all subfolders sorted alphabetically
folders = sorted([
    f for f in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, f))
])

# Only keep single-letter folders
letter_folders = [f for f in folders if len(f) == 1 and f.isalpha()]
print(f"  Found {len(letter_folders)} letter categories: {letter_folders}")

label_map = {name: i for i, name in enumerate(letter_folders)}

for folder_name in letter_folders:
    folder_path = os.path.join(DATASET_PATH, folder_name)
    count = 0

    for img_file in os.listdir(folder_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(folder_path, img_file)

        try:
            img = Image.open(img_path).convert('L').resize(IMAGE_SIZE)
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(label_map[folder_name])
            count += 1
        except Exception as e:
            print(f"  Skipped {img_file}: {e}")

    print(f"  Loaded {count:3d} images for '{folder_name}'")

X = np.array(images).reshape(-1, 28, 28, 1)
y = np.array(labels)
print(f"\n  Total: {len(X)} images, {len(label_map)} classes")

# --------------------------------------------------
# SHUFFLE DATA  ← THIS IS THE KEY FIX
# --------------------------------------------------
# Data is loaded folder by folder (all a's, then all b's...)
# We MUST shuffle before splitting into train/validation
# otherwise train gets some letters and validation gets others!

print("\n  Shuffling data...")
indices = np.random.permutation(len(X))   # create shuffled index list
X = X[indices]                             # reorder images using shuffled indices
y = y[indices]                             # reorder labels  using same shuffled indices
print("  Shuffle complete!")

# --------------------------------------------------
# LOAD PRE-TRAINED MODEL
# --------------------------------------------------
print("\n Loading pre-trained model...")
model = keras.models.load_model('best_model.h5')

# Check if output matches our number of classes
# .units gives the number of neurons in a Dense layer
current_output = model.layers[-1].units
needed_output  = len(letter_folders)

print(f"  Model outputs: {current_output} | Classes needed: {needed_output}")

if current_output != needed_output:
    print(f"  Mismatch detected! Rebuilding output layer...")
    # Build a new Sequential model using all layers except the last one
    new_model = tf.keras.Sequential(model.layers[:-1])
    # Add a new output layer with the correct number of classes
    # Give it a unique name to avoid conflicts
    new_model.add(tf.keras.layers.Dense(
        needed_output,
        activation='softmax',
        name='new_output'      # unique name avoids "name already exists" error
    ))
    model = new_model
    print("  Output layer rebuilt successfully!")

# --------------------------------------------------
# COMPILE WITH LOWER LEARNING RATE
# --------------------------------------------------
# We use a SMALL learning rate (0.0001) because:
# The model already learned from MNIST — we don't want to
# "forget" that knowledge, just gently update it with new photos
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------------------------------
# DATA AUGMENTATION
# --------------------------------------------------
# Creates slightly modified versions of each image during training
# This helps the model generalize to new, unseen hands
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,       # rotate up to 15 degrees
    zoom_range=0.15,         # zoom in/out up to 15%
    width_shift_range=0.1,   # shift left/right up to 10%
    height_shift_range=0.1,  # shift up/down up to 10%
    horizontal_flip=False,   # NEVER flip — flipping changes the sign meaning!
    validation_split=0.2     # use 20% of data for validation
)

# Create training and validation generators
# subset='training' → uses the first 80% of data
# subset='validation' → uses the last 20% of data
train_gen = datagen.flow(X, y, batch_size=BATCH_SIZE, subset='training',   shuffle=True)
val_gen   = datagen.flow(X, y, batch_size=BATCH_SIZE, subset='validation', shuffle=False)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
print("\n Fine-tuning on real photos...\n")

callbacks = [
    # Stop early if model stops improving (saves time)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    # Save the best version automatically
    tf.keras.callbacks.ModelCheckpoint(
        'asl_model_final.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# --------------------------------------------------
# EVALUATE
# --------------------------------------------------
print("\n Evaluating...")
_, accuracy = model.evaluate(X, y, verbose=0)
print(f"\n Overall Accuracy on Real Photos: {accuracy*100:.2f}%")

# --------------------------------------------------
# SAVE
# --------------------------------------------------
model.save('asl_model_final.h5')
np.save('label_map.npy', label_map)
print("\n Final model saved: asl_model_final.h5")
print(" Label map saved:   label_map.npy")

print("\n Step 2 complete!")
print("  Your model is ready!")
print("  Next: connect asl_model_final.h5 to your frontend app!")
