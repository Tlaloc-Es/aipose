from numpy import ndarray


class FrameManagerBase:
    """
    A base class for managing video frames.

    Attributes:
        stop (bool): A boolean flag indicating if frame reading should be stopped.

    Methods:
        __init__(): Initializes a new instance of the FrameManagerBase class.
        before_read_frame(): Method called before reading a new frame.
        frame_received(frame: ndarray) -> ndarray: Method called for each frame.
        to_stop() -> bool: Returns the current value of the stop flag.
        stream_started(source: str | int) -> None: Method called when a new stream is started.
        stream_ended() -> None: Method called when the video stream ends.
    """

    stop = False

    def __init__(self):
        """
        Initializes a new instance of the FrameManagerBase class.
        """

    def before_read_frame(self):
        """
        Method called before reading a new frame.
        """

    def frame_received(self, frame: ndarray) -> ndarray:
        """
        Method called for each frame.
        Args:
            frame (ndarray): A numpy array representing the frame.

        Returns:
            ndarray: A numpy array representing the modified frame.
        """
        return frame

    def to_stop(self) -> bool:
        """
        Returns the current value of the stop flag.
        Returns:
            bool: The current value of the stop flag.
        """
        return self.stop

    def stream_started(self, source: str | int) -> None:
        """
        Method called when a new video stream is started.
        Args:
            source (str | int): The source of the video stream (either a string path or
                an integer camera ID or a rtsp uri).
        """
        pass

    def stream_ended(self) -> None:
        """
        Method called when the video stream ends.
        """
        pass
