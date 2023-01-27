import logging
import os
from pathlib import Path

import click

from aipose.model import YoloV7Pose
from aipose.webcam import process_webcam

logging.basicConfig(level=logging.INFO)


@click.command()
def webcam():
    home_path = Path.home()
    aipose_path = os.path.join(home_path, ".aipose")
    os.makedirs(os.path.join(home_path, aipose_path), exist_ok=True)

    model = YoloV7Pose(aipose_path)

    process_webcam(model)


if __name__ == "__main__":
    webcam()
