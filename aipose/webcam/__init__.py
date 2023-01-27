import cv2

from aipose.webcam.frame_managers import FrameManager


def process_webcam(frame_manaer: FrameManager = FrameManager()):
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        import torch

        torch.cuda.empty_cache()

        ret, frame = capture.read()
        frame = frame_manaer.on_frame(frame)

        cv2.imshow("webCam", frame)
        if cv2.waitKey(1) == ord("s"):
            break

    capture.release()
    cv2.destroyAllWindows()
