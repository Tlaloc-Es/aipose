from typing import List

from numpy import ndarray

from aipose.frame import FrameManagerBase
from aipose.models.yolov7 import YoloV7Pose, YoloV7PoseKeypoints


class FrameYoloV7(FrameManagerBase):
    def __init__(self):
        self.model = YoloV7Pose()

    def on_frame(self, frame: ndarray) -> ndarray:
        prediction, image_tensor = self.model(frame)
        processed_frame = self.on_predict(frame, prediction, image_tensor)
        if processed_frame is None:
            return frame
        else:
            return processed_frame

    def on_predict(
        self,
        frame: ndarray,
        prediction: List[YoloV7PoseKeypoints],
        image_tensor: ndarray,
    ) -> None | ndarray:
        raise NotImplementedError
