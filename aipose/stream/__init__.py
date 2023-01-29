from typing import Any

import cv2
import torch

from aipose.frame.frame_manager_base import FrameManagerBase


def process_webcam(frame_manager: FrameManagerBase, webcam_input: int = 0):
    _process_stream(frame_manager, webcam_input)


def process_video(frame_manager: FrameManagerBase, path: str):
    _process_stream(frame_manager, path)


def _process_stream(frame_manager: FrameManagerBase, input: Any):
    if frame_manager is None:
        frame_manager = FrameManagerBase()

    capture = cv2.VideoCapture(input)

    while capture.isOpened():
        torch.cuda.empty_cache()

        ret, frame = capture.read()

        if frame is None:
            break

        frame = frame_manager.on_frame(frame)

        cv2.imshow("AIPOSE - Stream Video - Press 's' to exit", frame)

        if cv2.waitKey(1) == ord("s") or frame_manager.to_stop():
            break

    capture.release()
    cv2.destroyAllWindows()
