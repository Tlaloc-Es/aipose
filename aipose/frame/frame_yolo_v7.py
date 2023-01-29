from typing import List

from numpy import ndarray

from aipose.frame import FrameManagerBase
from aipose.model import Keypoints, YoloV7Pose


class FrameYoloV7(FrameManagerBase):
    def __init__(self):
        self.model = YoloV7Pose()

    def on_frame(self, frame: ndarray) -> ndarray:
        prediction, image_tensor = self.model(frame)
        proccessed_frame = self.on_predict(frame, prediction, image_tensor)
        if proccessed_frame is None:
            return frame
        else:
            return frame

    def on_predict(
        self, frame: ndarray, prediction: List[Keypoints], image_tensor: ndarray
    ) -> None | ndarray:
        raise NotImplementedError
