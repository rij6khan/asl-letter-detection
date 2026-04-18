"""
ASL Project — Flask Prediction Server
======================================
This script creates a local web server that:
1. Loads your trained AI model (asl_model_final.h5)
2. Listens for image frames from your browser (index.html)
3. Runs the prediction and sends back the result

Run: python server.py
Then open index.html in your browser.

Install required packages first:
  pip install flask flask-cors
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # hide TF startup messages

from flask import Flask, request, jsonify
from flask_cors import CORS              # allows browser to call this server
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

# --------------------------------------------------
# CREATE THE FLASK APP
# --------------------------------------------------
# Flask is a lightweight web server framework
# Think of it like a tiny post office — it receives requests
# and sends back responses
app = Flask(__name__)

# CORS = Cross-Origin Resource Sharing
# Without this, browsers block requests to different ports
# Our HTML is on one port, this server is on port 5000
# CORS tells the browser: "yes, that's allowed"
CORS(app)

# --------------------------------------------------
# LOAD MODEL AT STARTUP (only once, not per request)
# --------------------------------------------------
print("=" * 50)
print("  ASL Prediction Server")
print("=" * 50)

# Check model file exists
if not os.path.exists('asl_model_final.h5'):
    print("\n ERROR: asl_model_final.h5 not found!")
    print("Please run train_real_images.py first.")
    exit(1)

print("\n Loading model...")
model = tf.keras.models.load_model('asl_model_final.h5')
print("  Model loaded!")

# Load the label map (maps index → letter)
# e.g. {0: 'a', 1: 'b', 2: 'c', ...}
label_map     = np.load('label_map.npy', allow_pickle=True).item()
idx_to_label  = {v: k.upper() for k, v in label_map.items()}
# label_map is   {'a': 0, 'b': 1, ...}
# idx_to_label is {0: 'A', 1: 'B', ...}

print(f"  {len(idx_to_label)} classes: {list(idx_to_label.values())}")
print(f"\n Server ready!")
print(f"  API running at: http://localhost:5000")
print(f"  Open your index.html in a browser")
print(f"\n  Press Ctrl+C to stop\n")

# --------------------------------------------------
# ROUTE: Health Check
# --------------------------------------------------
# A simple endpoint to check the server is running
# Visit http://localhost:5000/health in your browser to test
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'asl_model_final.h5',
        'classes': len(idx_to_label)
    })

# --------------------------------------------------
# ROUTE: Predict
# --------------------------------------------------
# This is the main endpoint your browser calls
# It receives an image, runs it through the model, returns the letter
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Get the image data from the request
        # The browser sends JSON: {"image": "base64encodeddata..."}
        data    = request.json
        img_b64 = data['image']

        # Step 2: Remove the data URL prefix if it's there
        # Canvas.toDataURL() returns: "data:image/jpeg;base64,/9j/4AAQ..."
        # We only want the part after the comma
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]

        # Step 3: Decode base64 → raw bytes → PIL Image
        # base64 is a way to send binary data (like images) as text
        img_bytes = base64.b64decode(img_b64)
        img       = Image.open(io.BytesIO(img_bytes))

        # Step 4: Preprocess — exactly the same as training!
        # If training used 28x28 grayscale normalized images,
        # prediction MUST use the same format
        img       = img.convert('L')          # convert to grayscale
        img       = img.resize((28, 28))      # resize to 28x28
        img_array = np.array(img) / 255.0     # normalize 0-255 → 0.0-1.0
        img_array = img_array.reshape(1, 28, 28, 1)  # add batch + channel dims

        # Step 5: Run the prediction
        # model.predict returns an array of probabilities for each class
        # e.g. [[0.01, 0.87, 0.02, ...]] for 26 letters
        predictions = model.predict(img_array, verbose=0)[0]

        # Step 6: Find the best prediction
        top_idx    = int(np.argmax(predictions))      # index with highest probability
        top_letter = idx_to_label.get(top_idx, '?')  # convert index → letter
        top_conf   = float(predictions[top_idx])      # the probability (0.0 to 1.0)

        # Step 7: Build a dictionary of ALL letter confidences
        # The browser can use this to show exact confidence for any target letter
        all_confidences = {
            idx_to_label.get(i, '?'): float(predictions[i])
            for i in range(len(predictions))
        }

        # Step 8: Return the result as JSON
        return jsonify({
            'letter':     top_letter,          # best guess letter
            'confidence': top_conf,            # confidence for best guess (0.0-1.0)
            'all':        all_confidences      # confidence for every letter
        })

    except Exception as e:
        # If anything goes wrong, return an error message
        # This prevents the server from crashing on bad input
        print(f"  Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


# --------------------------------------------------
# START THE SERVER
# --------------------------------------------------
if __name__ == '__main__':
    # debug=False for production (True shows detailed errors in browser)
    # port=5000 is the standard Flask port
    # host='0.0.0.0' means accept connections from any device on local network
    app.run(debug=False, port=5000, host='0.0.0.0')
