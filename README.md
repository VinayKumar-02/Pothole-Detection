# Pothole-Detection
Pothole Detection is an intelligent system that uses computer vision techniques to identify potholes in road images and videos. The system leverages the YOLOv8 object detection model, which is trained to recognize potholes with high accuracy, and provides real-time detection on user-uploaded images via a web interface built with Streamlit.

Features:

1. YOLOv8 Object Detection: Utilizes YOLOv8, a state-of-the-art deep learning model, for accurate and fast pothole detection in road images and videos.
2. Image/Video Processing: Employs OpenCV for processing road images and videos, detecting potholes in real-time.
3. Streamlit Web Interface: Users can upload images or videos, and the system will display the potholes identified in the media with bounding boxes and confidence scores.
4. Visualization: Performance of the model, including detection accuracy and loss over epochs, is visualized using Matplotlib during training.
5. Scalable Experimentation: Model training was conducted on Google Colab, utilizing the power of GPUs for efficient training and experimentation.

Technologies Use:

1. YOLOv8 (Ultralytics): For object detection and pothole identification.
2. OpenCV: For image and video processing.
3. Streamlit: For creating an interactive web interface for real-time pothole detection.
4. Matplotlib: For visualizing model performance metrics and training progress.
5. Google Colab: Used for training the model with scalable GPU resources.
