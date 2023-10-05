import cv2

from aipose.frame.frame_manager_base import FrameManagerBase


def process_webcam(frame_manager: FrameManagerBase, webcam_input: int = 0):
    """
    This function captures frames from the default webcam and applies the given frame
    manager. It then displays the frames in a window until the user exits the window.

    Args:
        frame_manager: A FrameManagerBase object that manages the processing of frames.
        webcam_input: An integer representing the index of the webcam to use.
                        Default is 0 for the default system webcam.

    Returns:
        None
    """

    _process_stream(frame_manager, webcam_input)


def process_video(frame_manager: FrameManagerBase, path: str):
    """
    This function reads frames from a video file and applies the given frame manager.
    It then displays the frames in a window until the user exits the window.

    Args:
        frame_manager: A FrameManagerBase object that manages the processing of frames.
        path: A string representing the path to the video file.

    Returns:
        None
    """
    _process_stream(frame_manager, path)


def process_rtsp(frame_manager: FrameManagerBase, rtsp_url: str):
    """
    This function reads frames from an RTSP stream and applies the given frame manager.
    It then displays the frames in a window until the user exits the window.

    Args:
        frame_manager: A FrameManagerBase object that manages the processing of frames.
        rtsp_url: A string representing the RTSP URL.

    Returns:
        None
    """

    _process_stream(frame_manager, rtsp_url)


def _process_stream(frame_manager: FrameManagerBase, source: str | int):
    """
    This is a helper function that processes a stream of frames from a video source and
    given frame manager. The source can be a webcam, a video file, or an RTSP stream.

    Args:
        frame_manager: A FrameManagerBase object that manages the processing of frames.
        source: An integer representing the index of the webcam to use,
               a string representing the path to the video file, or a string
               representing the RTSP URL.
    Returns:
        None
    """
    if frame_manager is None:
        frame_manager = FrameManagerBase()

    capture = cv2.VideoCapture(source)

    frame_manager.stream_started(source)

    while capture.isOpened():
        frame_manager.before_read_frame()
        ret, frame = capture.read()

        if not ret:
            break

        frame = frame_manager.frame_received(frame)
        cv2.imshow("AIPOSE - Stream Video - Press 'q' to exit", frame)
        if cv2.waitKey(1) == ord("q") or frame_manager.to_stop():
            break

    capture.release()
    cv2.destroyAllWindows()

    frame_manager.stream_ended()
