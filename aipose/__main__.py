import logging

import click

from aipose.model import YoloV7Pose
from aipose.webcam import process_webcam
from aipose.webcam.frame_managers import FramePlot

logging.basicConfig(level=logging.INFO)


@click.command()
def webcam():
    model = YoloV7Pose()
    frame_plot = FramePlot(model)
    process_webcam(frame_plot)


if __name__ == "__main__":
    webcam()
