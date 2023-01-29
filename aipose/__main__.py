import logging

import click

from aipose.frame import FramePlot
from aipose.model import YoloV7Pose
from aipose.stream import process_webcam

logging.basicConfig(level=logging.INFO)


@click.command()
def webcam():
    model = YoloV7Pose()
    frame_plot = FramePlot(model)
    process_webcam(frame_plot)


if __name__ == "__main__":
    webcam()
