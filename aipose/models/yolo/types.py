from enum import Enum

from pydantic import BaseModel


class YoloPoseJoints(Enum):
    """
    Enumerates the different joints that can be detected by a Ultralytics YOLO
    Pose model.
    """

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class YoloPoseKeypoint(BaseModel):
    """
    Represents a YOLOv7 pose keypoint with floating point x and y coordinates and a
    confidence score.

    Attributes:
        x (float | int): The x-coordinate of the keypoint.
        y (float | int): The y-coordinate of the keypoint.
        conf (float | int): The confidence score of the keypoint.
    """

    x: float | int
    y: float | int
    conf: float | int
