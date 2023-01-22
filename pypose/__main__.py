import os

import cv2
from numpy import ndarray

from pypose.model import YoloV7Pose
from pypose.webcam import process_webcam

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

image: ndarray = cv2.imread("./person.jpg")
model = YoloV7Pose("./yolov7-w6-pose.pt")

process_webcam(model)
