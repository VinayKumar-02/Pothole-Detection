import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('best.pt')

def detect_objects(frame):
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = box.conf[0]
            cls = box.cls[0]

            if int(cls) == 0:  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'pothole: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def main():
    st.markdown("<h1 style='text-align: center; color: black;'>POTHOLE DETECTION USING YOLO</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        
        detected_image = detect_objects(image)
        
        st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), caption='Detected Image', use_column_width=True)

if __name__ == "__main__":
    main()
