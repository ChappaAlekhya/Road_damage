import torch
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the pretrained yolov11 model
model = YOLO('E:\Alekhya AEE\roaddamage\best.pt')  

# Class names and alternative labels
classes = ['D00', 'D10', 'D20', 'D40']
alt_names = {'D00': 'lateral_crack', 'D10': 'linear_cracks', 'D20': 'aligator_cracks', 'D40': 'potholes'}

def process_image(image):
    # Convert the image to RGB (Streamlit requires RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(image)  # Perform inference with the image
    
    # Draw bounding boxes and labels on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box (x1, y1, x2, y2)
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class index
            
            # Get class name
            label = classes[cls]
            alt_label = alt_names[label]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
            text = f"{alt_label}: {conf:.2f}"
            cv2.putText(image_rgb, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image_rgb

def main():
    st.title("YOLO Object Detection with Streamlit")
    
    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Process the image
        processed_image = process_image(image)
        
        # Display the processed image using Matplotlib
        st.image(processed_image, channels="RGB", use_column_width=True)

if __name__ == "__main__":
    main()