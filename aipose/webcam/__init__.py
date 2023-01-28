import cv2
import torch

from aipose.webcam.frame_managers import FrameManager


def process_webcam(frame_manager: FrameManager):
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        torch.cuda.empty_cache()

        ret, frame = capture.read()
        frame = frame_manager.on_frame(frame)

        cv2.imshow("AIPOSE - WebCam - Press 's' to exit", frame)

        if cv2.waitKey(1) == ord("s") or frame_manager.to_stop():
            break

    capture.release()
    cv2.destroyAllWindows()
