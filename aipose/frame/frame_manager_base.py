from numpy import ndarray


class FrameManagerBase:
    stop = False

    def __init__(self):
        pass

    def before_read_frame(self):
        pass

    def on_frame(self, frame: ndarray) -> ndarray:
        return frame

    def to_stop(self) -> bool:
        return self.stop

    def on_starts_stream(self, source: str | int) -> None:
        pass

    def on_ends_stream(self) -> None:
        pass
