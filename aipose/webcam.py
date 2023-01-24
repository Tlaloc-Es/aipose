import cv2
import numpy as np
from numpy import ndarray

from aipose.model import YoloV7Pose
from aipose.plot import plot


def webcam_ia(frame: ndarray, model: YoloV7Pose) -> ndarray:
    prediction, image_tensor = model(frame)

    frame = plot(
        image_tensor,
        np.array([value.get_raw_keypoint() for value in prediction]),
        plot_image=False,
        return_img=True,
    )

    return frame


def process_webcam(model: YoloV7Pose):
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        import torch

        torch.cuda.empty_cache()

        ret, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = webcam_ia(frame, model)

        cv2.imshow("webCam", frame)
        if cv2.waitKey(1) == ord("s"):
            break

    capture.release()
    cv2.destroyAllWindows()
