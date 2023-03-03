import hashlib
import logging
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
from numpy import ndarray
from pydantic import BaseModel
from torchvision import transforms
from tqdm import tqdm

from aipose.utils import letterbox, non_max_suppression_kpt, output_to_keypoint


class BBox(BaseModel):
    xmin: int
    xmax: int
    ymin: int
    ymax: int


class YoloV7PoseKeypoint(BaseModel):
    x: float | int
    y: float | int
    conf: float | int


class YoloV7PoseKeypoints:
    _step_keypoint: int = 3
    raw_keypoints: List[float]
    height: int = 0
    width: int = 0

    def __init__(self, raw_keypoints: List[float], height: int, width: int):
        self.raw_keypoints = raw_keypoints
        self.height = height
        self.width = width

    def _get_x_y_conf(self, start_index: int) -> YoloV7PoseKeypoint:
        end_index = start_index + self._step_keypoint
        x = self.raw_keypoints[start_index:end_index][0]
        y = self.raw_keypoints[start_index:end_index][1]
        conf = self.raw_keypoints[start_index:end_index][2]
        return YoloV7PoseKeypoint(x=x, y=y, conf=conf)

    def _replace_points(self, a_index: int, b_index: int):
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

    def total_confidence_over(self, expected_confidence: float):
        return [
            *filter(
                lambda x: x > expected_confidence,
                [
                    *map(
                        self.raw_keypoints.__getitem__,
                        [
                            9,
                            12,
                            15,
                            18,
                            21,
                            24,
                            27,
                            30,
                            33,
                            36,
                            39,
                            42,
                            45,
                            48,
                            51,
                            53,
                        ],
                    )
                ],
            )
        ]

    def get_nose(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(7)

    def get_left_eye(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(10)

    def get_right_eye(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(13)

    def get_left_ear(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(16)

    def get_right_ear(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(19)

    def get_left_shoulder(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(22)

    def get_right_shoulder(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(25)

    def get_left_elbow(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(28)

    def get_right_elbow(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(31)

    def get_left_wrist(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(34)

    def get_right_wrist(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(37)

    def get_left_hip(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(40)

    def get_right_hip(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(43)

    def get_left_knee(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(46)

    def get_right_knee(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(49)

    def get_left_ankle(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(52)

    def get_right_ankle(self) -> YoloV7PoseKeypoint:
        return self._get_x_y_conf(55)

    def get_body_keypoints(self) -> List[float]:
        return self.raw_keypoints[7:]

    def is_backwards(self) -> bool:
        return self.get_left_ear().x < self.get_right_ear().x

    def get_body_keypoints_normalize(self):
        body_keypoints = self.get_body_keypoints()
        body_keypoints_normalaized = body_keypoints / np.array(
            [self.width, self.height, 1] * 17
        )
        return body_keypoints_normalaized

    def get_raw_keypoint(self) -> List[float]:
        return self.raw_keypoints

    def get_points(self) -> ndarray:
        raw_points = self.get_body_keypoints()
        points = np.array([*zip(raw_points[::3], raw_points[1::3])])
        return points

    def get_confidences(self) -> ndarray:
        raw_points = self.get_body_keypoints_normalize()
        confidences = raw_points[2::3]
        return confidences

    def cosine_similarity(self, pose: "YoloV7PoseKeypoints") -> float:
        a = self.get_points_normalize_by_bbox()
        b = pose.get_points_normalize_by_bbox()

        return round(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 4)

    def get_points_normalize_by_bbox(self) -> ndarray:
        bbox_0 = self.calculate_bbox()
        xmin_0 = bbox_0.xmin
        ymin_0 = bbox_0.ymin
        bbox_0.xmax = bbox_0.xmax - xmin_0
        bbox_0.xmin = bbox_0.xmin - xmin_0
        bbox_0.ymax = bbox_0.ymax - ymin_0
        bbox_0.ymin = bbox_0.ymin - ymin_0
        keypoints_0 = self.get_body_keypoints()
        keypoints_0_normalized = keypoints_0 - np.array([xmin_0, ymin_0, 0] * 17)
        return keypoints_0_normalized

    def calculate_bbox(self) -> BBox:
        xmin = min([x for x, y in self.get_points()])
        xmax = max([x for x, y in self.get_points()])
        ymin = min([y for x, y in self.get_points()])
        ymax = max([y for x, y in self.get_points()])

        return BBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    def turn(self) -> "YoloV7PoseKeypoints":
        self._replace_points(10, 13)
        self._replace_points(16, 19)
        self._replace_points(22, 25)
        self._replace_points(28, 31)
        self._replace_points(34, 37)
        self._replace_points(40, 43)
        self._replace_points(46, 49)
        self._replace_points(52, 55)
        return self

    def __str__(self) -> str:
        return str(self.raw_keypoints)


class YoloV7Pose:
    aipose_model_path: str = ""
    aipose_path: str = ""
    _model_path: str = ""
    _model_repo: str = "WongKinYiu/yolov7"
    aipose_model_hash: str = "62ca91ec6612b22bef0ab4c95f3e2d07"
    aipose_model_file_name: str = "yolov7-w6-pose.pt"
    model_url_download: str = "https://huggingface.co/Tlaloc-Es/yolov7-w6-pose.pt/resolve/main/yolov7-w6-pose.pt"  # noqa: E501

    def __init__(self):
        home_path = Path.home()
        self.aipose_path = os.path.join(home_path, ".aipose")
        os.makedirs(os.path.join(home_path, self.aipose_path), exist_ok=True)

        self.aipose_model_path = os.path.join(
            self.aipose_path, self.aipose_model_file_name
        )

        self.download_yolo_w6_pose()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        try:
            self.model = torch.hub.load(
                self._model_repo, "custom", f"{self.aipose_path}", trust_repo=True
            )
        except Exception as e:  # noqa: F841
            weigths = torch.load(self.aipose_model_path, map_location=self.device)

        self.model = weigths["model"]
        self.model.float().eval()

        if torch.cuda.is_available():
            self.model.half().to(self.device)

    def download_yolo_w6_pose(self) -> None:

        if not os.path.isfile(self.aipose_model_path):
            self._download_yolo_w6_pose()
        current_model_hash = hashlib.md5(
            open(self.aipose_model_path, "rb").read()
        ).hexdigest()

        if self.aipose_model_hash != current_model_hash:
            self._download_yolo_w6_pose()

    def _download_yolo_w6_pose(self):
        logging.info("Downloding yolov7-w6-pose.pt")
        self.download_file(
            self.model_url_download,
            self.aipose_model_path,
        )

    def download_file(self, url, local_filename):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192)):
                    f.write(chunk)
        return local_filename

    def __call__(self, image: ndarray) -> Tuple[List[YoloV7PoseKeypoints], ndarray]:
        # Resize and pad image
        image = letterbox(image, 960, stride=64, auto=True)[0]  # shape: (567, 960, 3)
        # Apply transforms
        image = transforms.ToTensor()(image)  # torch.Size([3, 567, 960])
        if torch.cuda.is_available():
            image = image.half().to(self.device)
        # Turn image into batch
        image = image.unsqueeze(0)  # torch.Size([1, 3, 567, 960])
        with torch.no_grad():
            output, _ = self.model(image)

        output = non_max_suppression_kpt(
            output,
            0.25,
            0.65,
            nc=self.model.yaml["nc"],
            nkpt=self.model.yaml["nkpt"],
            kpt_label=True,
        )

        with torch.no_grad():
            output = output_to_keypoint(output)

        return [
            YoloV7PoseKeypoints(prediction, image.shape[2], image.shape[3])
            for prediction in output
        ], image
