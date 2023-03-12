from enum import Enum

from pydantic import BaseModel


class PredictionBoundingBoxXYWH(BaseModel):
    """
    Represents a bounding box prediction, defined by its X and Y coordinates,
    width, height, and confidence score.

    Attributes:
    - x (float | int): The X coordinate of the bounding box.
    - y (float | int): The Y coordinate of the bounding box.
    - width (float | int): The width of the bounding box.
    - height (float | int): The height of the bounding box.
    - confidence (float | int): The confidence score of the bounding box prediction.
    """

    x: float | int
    y: float | int
    width: float | int
    height: float | int
    confidence: float | int


class PredictionBoundingBoxXYXY(BaseModel):
    """
    Represents a bounding box prediction, defined by its minimum and maximum X and Y coordinates,
    along with a confidence score.

    Attributes:
    - xmin (float | int): The minimum X coordinate of the bounding box.
    - xmax (float | int): The maximum X coordinate of the bounding box.
    - ymin (float | int): The minimum Y coordinate of the bounding box.
    - ymax (float | int): The maximum Y coordinate of the bounding box.
    - confidence (float | int): The confidence score of the bounding box prediction.
    """

    xmin: float | int
    xmax: float | int
    ymin: float | int
    ymax: float | int
    confidence: float | int


class YoloV7PoseKeypoint(BaseModel):
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


class YoloV7PoseJoints(Enum):
    """
    Enumerates the different joints that can be detected by a YOLOv7 Pose model.
    """

    NOSE = 7
    LEFT_EYE = 10
    RIGHT_EYE = 13
    LEFT_EAR = 16
    RIGHT_EAR = 19
    LEFT_SHOULDER = 22
    RIGHT_SHOULDER = 25
    LEFT_ELBOW = 28
    RIGHT_ELBOW = 31
    LEFT_WRIST = 34
    RIGHT_WRIST = 37
    LEFT_HIP = 40
    RIGHT_HIP = 43
    LEFT_KNEE = 46
    RIGHT_KNEE = 49
    LEFT_ANKLE = 52
    RIGHT_ANKLE = 55


class YoloV7PoseKeypointsIndex(Enum):
    """
    Enumerates the different keypoint indices that correspond to the various joints detected by a YOLOv7 Pose model.
    """

    BATCH_ID = 0
    CLASS_ID = 1
    X = 2
    Y = 3
    WIDTH = 4
    HEIGHT = 5
    CONFIDENCE = 6
    X_NOSE = 7
    Y_NOSE = 8
    CONFIDENCE_NOSE = 9
    X_LEFT_EYE = 10
    Y_LEFT_EYE = 11
    CONFIDENCE_LEFT_EYE = 12
    X_RIGHT_EYE = 13
    Y_RIGHT_EYE = 14
    CONFIDENCE_RIGHT_EYE = 15
    X_LEFT_EAR = 16
    Y_LEFT_EAR = 17
    CONFIDENCE_LEFT_EAR = 18
    X_RIGHT_EAR = 19
    Y_RIGHT_EAR = 20
    CONFIDENCE_RIGHT_EAR = 21
    X_LEFT_SHOULDER = 22
    Y_LEFT_SHOULDER = 23
    CONFIDENCE_LEFT_SHOULDER = 24
    X_RIGHT_SHOULDER = 25
    Y_RIGHT_SHOULDER = 26
    CONFIDENCE_RIGHT_SHOULDER = 27
    X_LEFT_ELBOW = 28
    Y_LEFT_ELBOW = 29
    CONFIDENCE_LEFT_ELBOW = 30
    X_RIGHT_ELBOW = 31
    Y_RIGHT_ELBOW = 32
    CONFIDENCE_RIGHT_ELBOW = 33
    X_LEFT_WRIST = 34
    Y_LEFT_WRIST = 35
    CONFIDENCE_LEFT_WRIST = 36
    X_RIGHT_WRIST = 37
    Y_RIGHT_WRIST = 38
    CONFIDENCE_RIGHT_WRIST = 39
    X_LEFT_HIP = 40
    Y_LEFT_HIP = 41
    CONFIDENCE_LEFT_HIP = 42
    X_RIGHT_HIP = 43
    Y_RIGHT_HIP = 44
    CONFIDENCE_RIGHT_HIP = 45
    X_LEFT_KNEE = 46
    Y_LEFT_KNEE = 47
    CONFIDENCE_LEFT_KNEE = 48
    X_RIGHT_KNEE = 49
    Y_RIGHT_KNEE = 50
    CONFIDENCE_RIGHT_KNEE = 51
    X_LEFT_ANKLE = 52
    Y_LEFT_ANKLE = 53
    CONFIDENCE_LEFT_ANKLE = 54
    X_RIGHT_ANKLE = 55
    Y_RIGHT_ANKLE = 56
    CONFIDENCE_RIGHT_ANKLE = 57
