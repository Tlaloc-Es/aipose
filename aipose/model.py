import hashlib
import logging
import os
from pathlib import Path
from typing import List, Tuple

import requests
import torch
from numpy import ndarray
from pydantic import BaseModel
from torchvision import transforms
from tqdm import tqdm

from aipose.utils import letterbox, non_max_suppression_kpt, output_to_keypoint


class Keypoint(BaseModel):
    x: float | int
    y: float | int
    conf: float | int


class Keypoints:
    _step_keypoint = 3
    raw_keypoints: List[float]

    def __init__(self, raw_keypoints: List[float]):
        self.raw_keypoints = raw_keypoints

    def _get_x_y_conf(self, start_index: int) -> Keypoint:
        end_index = start_index + self._step_keypoint
        x = self.raw_keypoints[start_index:end_index][0]
        y = self.raw_keypoints[start_index:end_index][1]
        conf = self.raw_keypoints[start_index:end_index][2]
        return Keypoint(x=x, y=y, conf=conf)

    def get_nose(self) -> Keypoint:
        return self._get_x_y_conf(7)

    def get_left_eye(self) -> Keypoint:
        return self._get_x_y_conf(10)

    def get_right_eye(self) -> Keypoint:
        return self._get_x_y_conf(13)

    def get_left_ear(self) -> Keypoint:
        return self._get_x_y_conf(16)

    def get_right_ear(self) -> Keypoint:
        return self._get_x_y_conf(19)

    def get_left_shoulder(self) -> Keypoint:
        return self._get_x_y_conf(22)

    def get_right_shoulder(self) -> Keypoint:
        return self._get_x_y_conf(25)

    def get_left_elbow(self) -> Keypoint:
        return self._get_x_y_conf(28)

    def get_right_elbow(self) -> Keypoint:
        return self._get_x_y_conf(31)

    def get_left_wrist(self) -> Keypoint:
        return self._get_x_y_conf(34)

    def get_right_wrist(self) -> Keypoint:
        return self._get_x_y_conf(37)

    def get_left_hip(self) -> Keypoint:
        return self._get_x_y_conf(40)

    def get_right_hip(self) -> Keypoint:
        return self._get_x_y_conf(43)

    def get_left_knee(self) -> Keypoint:
        return self._get_x_y_conf(46)

    def get_right_knee(self) -> Keypoint:
        return self._get_x_y_conf(49)

    def get_left_ankle(self) -> Keypoint:
        return self._get_x_y_conf(52)

    def get_right_ankle(self) -> Keypoint:
        return self._get_x_y_conf(54)

    def get_body_keypoints(self) -> List[float]:
        return self.raw_keypoints[7:]

    def get_raw_keypoint(self) -> List[float]:
        return self.raw_keypoints

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

    def __call__(self, image: ndarray) -> Tuple[List[Keypoints], ndarray]:
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

        return [Keypoints(prediction) for prediction in output], image
