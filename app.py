import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import pathlib
from pathlib import Path
import requests

model_url = 'https://raw.githubusercontent.com/nihalvarma14/updated-streamlit-tennis-app/blob/main/best.pt'
# Load your YOLOv5 model - ensure 'best.pt' is in the same directory or provide the path
# model_path = 'C:\\Users\\peric\\Desktop\\infosys_project\\best.pt'  # Update this path if your model file is elsewhere

# repo = "nihalvarma14/updated-streamlit-tennis-app"
# model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_path, source = 'local')
# model = torch.load(model_path, weights_only=True)

pathlib.PosixPath = pathlib.WindowsPath

repo_path = 'C:\\Users\\peric\\Desktop\\infosys_project'
model_path = 'C:\\Users\\peric\\Desktop\\infosys_project\\best.pt'  # Replace with your actual .pt file

# Load the custom YOLOv5 model
model = torch.hub.load(repo_path, 'custom', path=model_path, source='local')


# Streamlit app UI
st.title('Tennis Player Detection App')
st.write('Upload a tennis video to detect players in real-time.')

# File uploader for video input
uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video_path = str(temp_video.name)
    temp_video.write(uploaded_video.read())
    temp_video.close()

    # Open video capture
    cap = cv2.VideoCapture(temp_video.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #Perform detection
        results = model(frame)
        frame = np.squeeze(results.render())  # Draw the detection boxes on the frame

        #Convert BGR to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Display the frame in Streamlit
        stframe.image(frame, channels='RGB', use_column_width=True)

    cap.release()
    st.success('Video processing complete!')

st.write("Ensure 'best.pt' is in the same directory or provide the correct path in model_path.")