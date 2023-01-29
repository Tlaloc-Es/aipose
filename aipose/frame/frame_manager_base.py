from numpy import ndarray


class FrameManagerBase:
    stop = False

    def __init__(self):
        pass

    def on_frame(self, frame: ndarray) -> ndarray:
        return frame

    def to_stop(self) -> bool:
        return self.stop

    def on_ends_stream(self) -> None:
        pass
