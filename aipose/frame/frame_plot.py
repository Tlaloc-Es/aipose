from typing import List

import cv2
import numpy as np
from numpy import ndarray

from aipose.frame import FrameManagerBase
from aipose.model import Keypoints
from aipose.plot import plot


class FramePlot(FrameManagerBase):
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
