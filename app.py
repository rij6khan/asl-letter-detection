#make sure to have flask installed 
#to run locally:
# flask -- run OR flask run
#make sure to have the following files:
#- model.pth - the model weights
#- hand_landmarker.task - Mediapipe's hand-landmarker detection model


#libraries to import
from flask import Flask, render_template, Response
import cv2 #to show webcam
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

app = Flask(__name__)

#config
MODEL_PATH = 'model_new.pth' #saved model weights
LANDMARK_MODEL = 'hand_landmarker.task'

#LETTERS = ['a','b','c','d','e','f','g','h','i','k','l','m',
#           'n','o','p','q','r','s','t','u','v','x','y']
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

#CNN Model (put architecture of model here)
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

if not os.path.exists(MODEL_PATH):
    print("ERROR: model.pth not found")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel(len(LETTERS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded!")

#setup for mediapipe
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


#to set up the webcam to be viewed on website
def camera_setup():
    #using webcam setup in webcam_asl.py
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        exit()

    #print("Webcam started — press Q to quit")

    while True:
        success, frame = cap.read()
        if not success:
            break

        #print("frame captured")

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb)
        )
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            #print(result.hand_landmarks)
            hand = result.hand_landmarks[0]

            coords = []
            h, w = frame.shape[:2]
            palmX, palmY, palmZ = hand[0].x, hand[0].y, hand[0].z

            for lm in hand:
                coords.extend([lm.x-palmX, lm.y-palmY, lm.z-palmZ])

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

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield(b'--frame \r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# #loading the best model's weights
# def load_model():
#     return 
# 
# #run the classification based on those weights
# def classification():
#     return 


@app.route("/")
def website():
    return render_template('index.html')

@app.route("/video_feed")
def video_feed():
    return Response(camera_setup(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)