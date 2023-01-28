from typing import List

import cv2
import numpy as np
from numpy import ndarray

from aipose.model import Keypoints, YoloV7Pose
from aipose.plot import plot


class FrameManager:
    stop = False

    def __init__(self):
        pass

    def on_frame(self, frame: ndarray) -> ndarray:
        return frame

    def to_stop(self) -> bool:
        return self.stop


class FrameYoloV7(FrameManager):
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


class FramePlot(FrameManager):
    def __init__(self, model):
        self.model = model

    def on_frame(self, frame: ndarray) -> ndarray:
        prediction, image_tensor = self.model(frame)
        frame = self._plot(prediction, image_tensor)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _plot(self, prediction: List[Keypoints], image_tensor: ndarray) -> ndarray:
        frame = plot(
            image_tensor,
            np.array([value.get_raw_keypoint() for value in prediction]),
            plot_image=False,
            return_img=True,
        )
        return frame
