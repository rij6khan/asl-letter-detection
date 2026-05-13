# Real-Time ASL Letter Recognition via Landmark-Based Deep Learning 

Group Members: 
* Nicholas Offei
* Rija Khan
* Jacob Sze
* Piyaporn Puangprasert

This project focuses on utliizing hand landmarks from Google's Mediapipe to extract features and patterns to detect different static ASL letters from A-Y (excluding J to Z since they are motion based). 

## Project Structure: 
* app.py: webcamera setup for real-time testing - done in Flask
* data_analysis.ipynb: analysis of the dataset chosen
* hand_landmarker.task: hand landmarker model from Mediapipe
* model_acc.pth & model.pth: model weights based on best validation accuracy (model_acc.pth) and best validation loss (model.pth)
* model.ipynb: runs training and testing to classify letters based on dataset
* ASL_Report: final project report

## How to Run:
Make sure to install all required dependencies:
```pip install -r requirements.txt```

Then, run the app.py file as stated below:
```flask -- run```