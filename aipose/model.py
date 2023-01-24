from typing import List, Tuple

import torch
from numpy import ndarray
from pydantic import BaseModel
from torchvision import transforms

from aipose.utils import letterbox, non_max_suppression_kpt, output_to_keypoint


class Keypoint(BaseModel):
    x: float
    y: float
    conf: float


class Keypoints:
    _step_keypoint = 3
    raw_keypoints: List[float]

    def __init__(self, raw_keypoints: List[float]):
        self.raw_keypoints = raw_keypoints

    def _get_x_y_conf(self, start_index: int) -> Keypoint:
        end_index = start_index + self._step_keypoint
        x = self.raw_keypoints[start_index:end_index]
        y = self.raw_keypoints[start_index:end_index]
        conf = self.raw_keypoints[start_index:end_index]
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

    _model_repo: str = "WongKinYiu/yolov7"

    def __init__(self, path: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        try:
            self.model = torch.hub.load(
                self._model_repo, "custom", f"{path}", trust_repo=True
            )
        except Exception as e:  # noqa: F841
            weigths = torch.load(f"{path}", map_location=self.device)
        self.model = weigths["model"]
        self.model.float().eval()

        if torch.cuda.is_available():
            self.model.half().to(self.device)

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
