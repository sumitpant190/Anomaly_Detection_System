import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import tempfile
import time

# Load your pre-trained model
model = load_model(r'CNN_LSTM.h5')  # Update the path to your model

categories_labels = {'Fighting': 0, 'Shoplifting': 1, 'Abuse': 2, 'Arrest': 3, 'Shooting': 4, 'Robbery': 5, 'Explosion': 6}
labels_categories = {v: k for k, v in categories_labels.items()}  # reverse dictionary for label lookup

def predict_frame(frame, anomaly_threshold=0.5):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the image
    resized_frame = cv2.resize(gray_frame, (50, 50))
    
    # Reshape the image to 4D array for CNN and LSTM input
    image_cnn = resized_frame.reshape((1,) + resized_frame.shape + (1,))
    image_lstm = resized_frame.reshape((1,) + (-1, 1))
    
    # Use the model to predict the category of the frame
    prediction = model.predict([image_cnn, image_lstm])
    
    # Get the highest probability and corresponding label
    max_prob = np.max(prediction)
    label_index = np.argmax(prediction)
    label = labels_categories[label_index]
    
    # Check if it's an anomaly based on the threshold
    if max_prob > anomaly_threshold:
        return True, label, max_prob
    else:
        return False, None, max_prob

def process_video_live(video_path, anomaly_threshold=0.5, frame_skip=15):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected_anomalies = []
    
    previous_prediction = None
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        if frame_number % frame_skip == 0:  # Process every 'frame_skip' frame
            is_anomaly, label, prob = predict_frame(frame, anomaly_threshold)
            if is_anomaly and (label != previous_prediction):
                detected_anomalies.append((frame_number, label, prob))
                previous_prediction = label
            
            st.image(frame, channels="BGR", caption=f'Frame {frame_number}')
            if is_anomaly:
                st.write(f"Frame {frame_number}: {label} (Probability: {prob:.4f})")
            else:
                st.write(f"Frame {frame_number}: No anomaly detected")
            
            time.sleep(0.1)  # Adjust this to control playback speed
    
    cap.release()
    return detected_anomalies

# Streamlit UI
st.title('Real-Time Anomaly Detection in Video Frames')
uploaded_video = st.file_uploader('Upload a video file', type=['mp4', 'avi'])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    st.video(tfile.name)
    
    if st.button('Predict'):
        st.write('Processing video...')
        anomalies = process_video_live(tfile.name, anomaly_threshold=0.5, frame_skip=15)
        
        if anomalies:
            st.write('Anomalies detected:')
            for frame_num, label, prob in anomalies:
                st.write(f'Frame {frame_num}: {label} (Probability: {prob:.4f})')
        else:
            st.write('No anomalies detected.')
