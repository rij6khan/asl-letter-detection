"""
ASL Webcam — Landmark-based Prediction (MATCHES YOUR NOTEBOOK)

Run:
    python3.9 webcam_asl.py
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# ===============================
# CONFIG
# ===============================
MODEL_PATH = 'model.pth'
LANDMARK_MODEL = 'hand_landmarker.task'

LETTERS = ['a','b','c','d','e','f','g','h','i','k','l','m',
           'n','o','p','q','r','s','t','u','v','x','y']

# ===============================
# CNN MODEL (MUST MATCH NOTEBOOK)
# ===============================
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, 3)
        self.conv2 = nn.Conv1d(32, 128, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===============================
# LOAD MODEL
# ===============================
if not os.path.exists(MODEL_PATH):
    print("ERROR: model.pth not found")
    exit()

device = torch.device("cpu")

model = CNNModel(len(LETTERS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded ✅")

# ===============================
# MEDIAPIPE SETUP
# ===============================
base_options = python.BaseOptions(
    model_asset_path=LANDMARK_MODEL,
    delegate=python.BaseOptions.Delegate.CPU
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("Webcam started — press Q to quit")

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb)
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        coords = []
        h, w = frame.shape[:2]

        for lm in hand:
            coords.extend([lm.x, lm.y, lm.z])

            # draw points
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(frame, (px, py), 5, (0,255,0), -1)

        # predict
        input_tensor = torch.tensor(
            np.array(coords).reshape(1, 21, 3),
            dtype=torch.float32
        )

        with torch.no_grad():
            output = model(input_tensor)[0]
            probs = torch.softmax(output, dim=0).numpy()

        idx = np.argmax(probs)
        letter = LETTERS[idx].upper()
        conf = probs[idx]

        cv2.putText(frame, f"{letter} ({conf:.2f})",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0,255,0), 3)

    else:
        cv2.putText(frame, "No Hand",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0,0,255), 3)

    cv2.imshow("ASL Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
