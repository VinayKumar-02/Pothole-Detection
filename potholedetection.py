# -*- coding: utf-8 -*-
"""PotholeDetection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1l9rwiReKfSckLAbh9vgNsim5B9z5M9-e
"""

import os
from IPython.display import Image, display

!nvidia-smi

!mkdir /content/Pothole

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/Pothole

HOME = os.getcwd()
print(HOME)

!mkdir {HOME}/datasets

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}/datasets

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="QUWMCW70c1dPy9pc03sQ")
project = rf.workspace("hiteshram").project("object-detection-bounding-box-ftfs5")
dataset = project.version(1).download("yolov5")

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}

# Commented out IPython magic to ensure Python compatibility.
# %cd {dataset.location}

!pip install ultralytics

import ultralytics

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}

!yolo task=detect mode=train model=yolov8m.pt data="/content/Pothole/datasets/Object-Detection-(Bounding-Box)-1/data.yaml" epochs=75 imgsz=640

!ls {HOME}//runs/detect/train

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}

Image(filename = f'{HOME}/runs/detect/train/confusion_matrix.png', width = 900)

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}

Image(filename = f'{HOME}/runs/detect/train/results.png', width = 600)

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}

Image(filename = f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width = 600)

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}

!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data="/content/Pothole/datasets/Object-Detection-(Bounding-Box)-1/data.yaml"

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source="/content/Pothole/datasets/Object-Detection-(Bounding-Box)-1/test/images"

Image('/content/Pothole/runs/detect/predict/142_jpg.rf.ef9f025b2536187f2dbbbdb80bc8bfb1.jpg')
