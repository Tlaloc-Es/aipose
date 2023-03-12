from typing import List

import cv2
import numpy as np
from numpy import ndarray

from aipose.frame import FrameManagerBase
from aipose.models.yolov7.domain import YoloV7PoseKeypoints
from aipose.plot import plot


class FramePlot(FrameManagerBase):
    """
    A subclass of FrameManagerBase that plots the predictions of a given model on each frame.

    Attributes:
        model: The model used to generate the predictions.

    Methods:
        __init__(model): Initializes a new instance of the FramePlot class with the specified model.
        frame_received(frame: ndarray) -> ndarray: Overrides the frame_received method of FrameManagerBase, applies the model to the frame and plots the predictions.
        _plot(prediction: List[YoloV7PoseKeypoints], image_tensor: ndarray) -> ndarray: Plots the predicted keypoints on the input image.
    """

    def __init__(self, model):
        """
        Initializes a new instance of the FramePlot class with the specified model.

        Args:
            model: The model used to generate the predictions.
        """
        self.model = model

    def frame_received(self, frame: ndarray) -> ndarray:
        """
        Overrides the frame_received method of FrameManagerBase, applies the model to the frame and plots the predictions.

        Args:
            frame (ndarray): The frame to process.

        Returns:
            ndarray: The modified frame.
        """
        prediction, image_tensor = self.model(frame)
        frame = self._plot(prediction, image_tensor)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _plot(
        self, prediction: List[YoloV7PoseKeypoints], image_tensor: ndarray
    ) -> ndarray:
        """
        Plots the predicted keypoints on the input image.

        Args:
            prediction (List[YoloV7PoseKeypoints]): The predicted keypoints.
            image_tensor (ndarray): The input image.

        Returns:
            ndarray: The image with the predicted keypoints plotted.
        """
        frame = plot(
            image_tensor,
            np.array([value.raw_keypoint for value in prediction]),
            plot_image=False,
            return_img=True,
        )
        return frame
