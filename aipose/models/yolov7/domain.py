from enum import Enum
from typing import List

import numpy as np
import torch
from numpy import ndarray
from torchvision import transforms

from aipose.downloader import Downloader
from aipose.models.yolov7.types import (
    PredictionBoundingBoxXYWH,
    PredictionBoundingBoxXYXY,
    YoloV7PoseJoints,
    YoloV7PoseKeypoint,
    YoloV7PoseKeypointsIndex,
)
from aipose.utils import letterbox, non_max_suppression_kpt, output_to_keypoint


class YoloV7PoseKeypoints:
    """
    A class that represents pose keypoints and their confidences extracted by the YOLOv7 model.
    """

    _step_keypoint: int = 3
    raw_keypoints: List[float]

    def __init__(
        self,
        raw_keypoints: List[float],
        height_prediction: int,
        width_prediction: int,
        height_original: int,
        width_original: int,
    ):
        """
        Initialize a YoloV7PoseKeypoints instance.

        :param raw_keypoints: A list of raw keypoints and their confidences.
        :param height: The height of the image.
        :param width: The width of the image.
        """
        self.raw_keypoints = raw_keypoints
        self.raw_keypoints[
            YoloV7PoseKeypointsIndex.WIDTH.value
        ] = self._convert_dimension(
            self.raw_keypoints[YoloV7PoseKeypointsIndex.WIDTH.value],
            width_prediction,
            width_original,
        )
        self.raw_keypoints[
            YoloV7PoseKeypointsIndex.HEIGHT.value
        ] = self._convert_dimension(
            self.raw_keypoints[YoloV7PoseKeypointsIndex.HEIGHT.value],
            width_prediction,
            width_original,
        )
        self.raw_keypoints[
            YoloV7PoseKeypointsIndex.X.value : YoloV7PoseKeypointsIndex.Y.value + 1
        ] = self._convert_point(
            self.raw_keypoints[YoloV7PoseKeypointsIndex.X.value],
            self.raw_keypoints[YoloV7PoseKeypointsIndex.Y.value],
            width_prediction,
            height_prediction,
            width_original,
            height_original,
        )
        self.raw_keypoints[7:] = self._convert_keypoints_with_confidence(
            self.raw_keypoints[7:],
            width_prediction,
            height_prediction,
            width_original,
            height_original,
        )

    @property
    def batch_id(self) -> int:
        return self.raw_keypoints[0]

    @property
    def class_id(self) -> int:
        return self.raw_keypoints[1]

    @property
    def body_keypoints(self) -> List[float]:
        return self.raw_keypoints[7:]

    @property
    def raw_keypoint(self) -> List[float]:
        return self.raw_keypoints

    @property
    def human_confidence(self) -> List[float]:
        return self.raw_keypoints[YoloV7PoseKeypointsIndex.CONFIDENCE]

    @property
    def human_center(self) -> List[float]:
        x = self.raw_keypoints[YoloV7PoseKeypointsIndex.X.value]
        y = self.raw_keypoints[YoloV7PoseKeypointsIndex.Y.value]
        return (x, y)

    @property
    def prediction_bounding_box_xyxy(self) -> PredictionBoundingBoxXYXY:
        points = self.keypoints_coordinates
        xmin = min([x for x, y in points])
        xmax = max([x for x, y in points])
        ymin = min([y for x, y in points])
        ymax = max([y for x, y in points])
        confidence = self.raw_keypoints[6]

        return PredictionBoundingBoxXYXY(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, confidence=confidence
        )

    @property
    def prediction_bounding_box_xywh(self) -> PredictionBoundingBoxXYWH:
        return PredictionBoundingBoxXYWH(
            x=self.raw_keypoints[2],
            y=self.raw_keypoints[3],
            width=self.raw_keypoints[4],
            height=self.raw_keypoints[5],
            confidence=self.raw_keypoints[6],
        )

    @property
    def is_backwards(self) -> bool:
        return (
            self.get_keypoint(YoloV7PoseJoints.LEFT_EAR).x
            < self.get_keypoint(YoloV7PoseJoints.RIGHT_EAR).x
        )

    @property
    def keypoints_coordinates(self) -> ndarray:
        body_keypoints = self.body_keypoints
        points = np.array([*zip(body_keypoints[::3], body_keypoints[1::3])])
        return points

    def total_confidence_over(
        self, expected_confidence: float
    ) -> List[YoloV7PoseJoints]:
        """
        This method returns a list of YoloV7PoseJoints objects where the confidence level is greater than a given expected confidence level.


        Attributes:
            expected_confidence (float): the confidence level threshold to filter keypoints

        Returns:
            List[YoloV7PoseJoints]: a list of YoloV7PoseJoints objects where the confidence level is greater than the expected confidence level.
        """
        keypoints_over_confidence = []

        for pose_joint_index in YoloV7PoseJoints:
            keypoint = self.get_keypoint(pose_joint_index)
            if keypoint.conf > expected_confidence:
                keypoints_over_confidence.append(keypoint)

        return keypoints_over_confidence

    def get_keypoint(self, keypoint: YoloV7PoseJoints | int) -> YoloV7PoseKeypoint:
        if isinstance(keypoint, Enum):
            keypoint = keypoint.value
        return self._get_x_y_conf(keypoint)

    def cosine_similarity(self, pose: "YoloV7PoseKeypoints") -> float:
        a = self.get_points_normalize_by_bbox()
        b = pose.get_points_normalize_by_bbox()

        return round(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 4)

    def get_points_normalize_by_bbox(self) -> ndarray:
        bbox = self.prediction_bounding_box_xywh
        xmin, ymin, width, height = bbox.x, bbox.y, bbox.width, bbox.height

        body_keypoints = self.body_keypoints
        body_keypoints_normalized = np.zeros_like(body_keypoints)

        for i in range(0, len(body_keypoints), 3):
            x, y, confidence = (
                body_keypoints[i],
                body_keypoints[i + 1],
                body_keypoints[i + 2],
            )
            x_normalized = (x - xmin) / width
            y_normalized = (y - ymin) / height
            body_keypoints_normalized[i] = x_normalized
            body_keypoints_normalized[i + 1] = y_normalized
            body_keypoints_normalized[i + 2] = confidence

        return body_keypoints_normalized

    def turn(self) -> "YoloV7PoseKeypoints":
        self._swap_keypoints(YoloV7PoseJoints.LEFT_EYE, YoloV7PoseJoints.RIGHT_EYE)
        self._swap_keypoints(YoloV7PoseJoints.LEFT_EAR, YoloV7PoseJoints.RIGHT_EAR)
        self._swap_keypoints(
            YoloV7PoseJoints.LEFT_SHOULDER, YoloV7PoseJoints.RIGHT_SHOULDER
        )
        self._swap_keypoints(YoloV7PoseJoints.LEFT_ELBOW, YoloV7PoseJoints.RIGHT_ELBOW)
        self._swap_keypoints(YoloV7PoseJoints.LEFT_WRIST, YoloV7PoseJoints.RIGHT_WRIST)
        self._swap_keypoints(YoloV7PoseJoints.LEFT_HIP, YoloV7PoseJoints.RIGHT_HIP)
        self._swap_keypoints(YoloV7PoseJoints.LEFT_KNEE, YoloV7PoseJoints.RIGHT_KNEE)
        self._swap_keypoints(YoloV7PoseJoints.LEFT_ANKLE, YoloV7PoseJoints.RIGHT_ANKLE)
        return self

    def _swap_keypoints(
        self, keypoint_a: YoloV7PoseJoints | int, keypoint_b: YoloV7PoseJoints | int
    ):
        """
        Swap the values of two keypoints.

        :param a_index: The index of the first keypoint.
        :param b_index: The index of the second keypoint.
        """
        if isinstance(keypoint_a, Enum):
            a_index = keypoint_a.value

        if isinstance(keypoint_b, Enum):
            b_index = keypoint_b.value

        end_index_a = a_index + self._step_keypoint
        a_x = self.raw_keypoints[a_index:end_index_a][0]
        a_y = self.raw_keypoints[a_index:end_index_a][1]
        a_conf = self.raw_keypoints[a_index:end_index_a][2]

        end_index_b = b_index + self._step_keypoint
        b_x = self.raw_keypoints[b_index:end_index_b][0]
        b_y = self.raw_keypoints[b_index:end_index_b][1]
        b_conf = self.raw_keypoints[b_index:end_index_b][2]

        self.raw_keypoints[a_index:end_index_a][0] = b_x
        self.raw_keypoints[a_index:end_index_a][1] = b_y
        self.raw_keypoints[a_index:end_index_a][2] = b_conf

        self.raw_keypoints[b_index:end_index_b][0] = a_x
        self.raw_keypoints[b_index:end_index_b][1] = a_y
        self.raw_keypoints[b_index:end_index_b][2] = a_conf

    def _convert_dimension(self, value, original_dim, target_dim):
        converted_value = (value / original_dim) * target_dim
        return converted_value

    def _convert_point(self, x, y, width1, height1, width2, height2):
        new_x = (x / width1) * width2
        new_y = (y / height1) * height2
        return [new_x, new_y]

    def _convert_keypoints_with_confidence(
        self, keypoints_with_confidence, width1, height1, width2, height2
    ):
        converted_keypoints = []

        for i in range(0, len(keypoints_with_confidence), 3):
            x = keypoints_with_confidence[i]
            y = keypoints_with_confidence[i + 1]

            converted_point = self._convert_point(
                x, y, width1, height1, width2, height2
            )
            confidence = keypoints_with_confidence[i + 2]

            converted_keypoints.extend(converted_point + [confidence])

        return converted_keypoints

    def _get_x_y_conf(self, start_index: int) -> YoloV7PoseKeypoint:
        """
        Get the x, y, and confidence values for a single keypoint.

        :param start_index: The index at which to start reading values from the raw_keypoints list.
        :return: A YoloV7PoseKeypoint object representing a single keypoint.
        """
        end_index = start_index + self._step_keypoint
        x = self.raw_keypoints[start_index:end_index][0]
        y = self.raw_keypoints[start_index:end_index][1]
        conf = self.raw_keypoints[start_index:end_index][2]
        return YoloV7PoseKeypoint(x=x, y=y, conf=conf)

    def __str__(self) -> str:
        return str(self.raw_keypoints)

    def __repr__(self) -> str:
        return str(self.raw_keypoints)


class YoloV7Pose:
    aipose_model_path: str = ""
    _model_repo: str = "WongKinYiu/yolov7"
    aipose_model_hash: str = "62ca91ec6612b22bef0ab4c95f3e2d07"
    aipose_model_file_name: str = "yolov7-w6-pose.pt"
    model_url_download: str = "https://huggingface.co/Tlaloc-Es/yolov7-w6-pose.pt/resolve/main/yolov7-w6-pose.pt"  # noqa: E501

    def __init__(self):
        self.aipose_model_path = Downloader().download(
            self.aipose_model_file_name, self.aipose_model_hash, self.model_url_download
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        try:
            self.model = torch.hub.load(
                self._model_repo,
                "custom",
                f"{ self.aipose_model_path}",
                trust_repo=True,
            )
        except Exception as e:  # noqa: F841
            weigths = torch.load(self.aipose_model_path, map_location=self.device)

        self.model = weigths["model"]
        self.model.float().eval()

        if torch.cuda.is_available():
            self.model.half().to(self.device)

    def __call__(
        self, image: ndarray, conf_thres=0.25, iou_thres=0.65
    ) -> List[YoloV7PoseKeypoints]:
        # Resize and pad image
        image_letterbox = letterbox(image, 960, stride=64, auto=True)[
            0
        ]  # shape: (567, 960, 3)
        # Apply transforms
        size_image = image.shape
        size_prediction = image_letterbox.shape

        image_letterbox = transforms.ToTensor()(
            image_letterbox
        )  # torch.Size([3, 567, 960])
        if torch.cuda.is_available():
            image_letterbox = image_letterbox.half().to(self.device)
        # Turn image into batch
        image_letterbox = image_letterbox.unsqueeze(0)  # torch.Size([1, 3, 567, 960])
        with torch.no_grad():
            original_output, _ = self.model(image_letterbox)

        original_output = non_max_suppression_kpt(
            original_output,
            conf_thres,
            iou_thres,
            nc=self.model.yaml["nc"],
            nkpt=self.model.yaml["nkpt"],
            kpt_label=True,
        )

        with torch.no_grad():
            output = output_to_keypoint(original_output)

        return [
            YoloV7PoseKeypoints(
                prediction,
                size_prediction[0],
                size_prediction[1],
                size_image[0],
                size_image[1],
            )
            for prediction in output
        ]
