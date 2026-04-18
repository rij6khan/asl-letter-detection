"""
ASL Project — Step 3: Test Your Model
======================================
Use this to test your trained model on any hand image!

Run: python step3_test_single_image.py path/to/hand_image.jpg
  Or: python step3_test_single_image.py   (uses a sample from your dataset)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0=all, 1=info, 2=warnings, 3=errors only
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import sys


print("=" * 50)
print("  ASL Model — Test Single Image")
print("=" * 50)

# Standard ASL labels (no J=9, no Z=25 because they need motion)
LABELS_MNIST = [
    'A','B','C','D','E','F','G','H','I','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y'
]

# --------------------------------------------------
# FIND MODEL FILE
# --------------------------------------------------
if os.path.exists('asl_model_final.h5'):
    model_path = 'asl_model_final.h5'
    print(f"\n Using final model: {model_path}")
elif os.path.exists('best_model.h5'):
    model_path = 'best_model.h5'
    print(f"\n Using MNIST model: {model_path}")
else:
    print("\n No model file found! Please run the training scripts first.")
    exit(1)

model = keras.models.load_model(model_path)
num_classes = model.layers[-1].units

# --------------------------------------------------
# GET IMAGE PATH
# --------------------------------------------------
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    # Use a sample image from the dataset
    sample_folder = os.path.join('asl_dataset', 'a')
    if os.path.exists(sample_folder):
        sample_file = os.listdir(sample_folder)[0]
        image_path  = os.path.join(sample_folder, sample_file)
        print(f"\n No image specified — using sample: {image_path}")
    else:
        print("\n Please provide an image path as argument.")
        print("   Example: python step3_test_single_image.py hand_a.jpg")
        exit(1)

if not os.path.exists(image_path):
    print(f"\n Image not found: {image_path}")
    exit(1)

# --------------------------------------------------
# PREPARE IMAGE
# --------------------------------------------------
print(f"\n  Loading image: {image_path}")
img = Image.open(image_path).convert('L').resize((28, 28))
img_array = np.array(img) / 255.0
img_array = img_array.reshape(1, 28, 28, 1)

# --------------------------------------------------
# PREDICT
# --------------------------------------------------
predictions = model.predict(img_array, verbose=0)
probs = predictions[0]

# Determine labels to use
if num_classes == 24:
    labels = LABELS_MNIST
elif os.path.exists('label_map.npy'):
    label_map = np.load('label_map.npy', allow_pickle=True).item()
    labels = [k.upper() for k, v in sorted(label_map.items(), key=lambda x: x[1])]
else:
    labels = [str(i) for i in range(num_classes)]

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
top_idx    = np.argmax(probs)
top_letter = labels[top_idx] if top_idx < len(labels) else "?"
top_conf   = probs[top_idx] * 100

print(f"\n{'='*30}")
print(f"   PREDICTION:  {top_letter}")
print(f"   CONFIDENCE:  {top_conf:.1f}%")
print(f"{'='*30}")

# Show all top guesses
print("\nTop 5 guesses:")
top5 = np.argsort(probs)[-5:][::-1]
for rank, idx in enumerate(top5, 1):
    letter = labels[idx] if idx < len(labels) else "?"
    bar    = "█" * int(probs[idx] * 30)
    print(f"  {rank}. {letter}  {bar} {probs[idx]*100:.1f}%")

print()