from typing import List

import torch
from numpy import ndarray

from aipose.frame import FrameManagerBase
from aipose.models.yolov7.domain import YoloV7Pose, YoloV7PoseKeypoints


class FrameYoloV7(FrameManagerBase):
    """
    A subclass of FrameManagerBase that uses YOLOv7 model to make predictions on each frame.

    Attributes:
        model: The YOLOv7 model used for predictions.

    Methods:
        __init__(): Initializes a new instance of the FrameYoloV7 class and sets the model to YOLOv7Pose.
        before_read_frame(): Overrides the before_read_frame method of FrameManagerBase, empties the GPU cache before reading the next frame.
        frame_received(frame: ndarray) -> ndarray: Overrides the frame_received method of FrameManagerBase, applies the model to the frame and processes the result.
        on_predict(frame: ndarray, prediction: List[YoloV7PoseKeypoints]) -> None | ndarray:
            This method is called after the prediction is made, can be overridden in the subclass for additional processing of the result.
    """

    def __init__(self):
        """
        Initializes a new instance of the FrameYoloV7 class and sets the model to YOLOv7Pose.
        """
        self.model = YoloV7Pose()

    def before_read_frame(self):
        """
        Overrides the before_read_frame method of FrameManagerBase, empties the GPU cache before reading the next frame.
        """
        torch.cuda.empty_cache()

    def frame_received(self, frame: ndarray) -> ndarray:
        """
        Overrides the frame_received method of FrameManagerBase, applies the model to the frame and processes the result.

        Args:
            frame (ndarray): The frame to process.

        Returns:
            ndarray: The processed frame.
        """
        prediction = self.model(frame)
        processed_frame = self.on_predict(frame, prediction)
        if processed_frame is None:
            return frame
        else:
            return processed_frame

    def on_predict(
        self,
        frame: ndarray,
        prediction: List[YoloV7PoseKeypoints],
    ) -> None | ndarray:
        """
        This method is called after the prediction is made, can be overridden in the subclass for additional processing of the result.

        Args:
            frame (ndarray): The input frame.
            prediction (List[YoloV7PoseKeypoints]): The predicted keypoints.

        Returns:
            None | ndarray: If None, the processed frame is not returned and the original frame is returned instead.
            If an ndarray is returned, it should be the processed frame.
        """
        raise NotImplementedError
