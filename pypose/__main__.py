import os

import click

from pypose.model import YoloV7Pose
from pypose.webcam import process_webcam


@click.command()
def webcam():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    model = YoloV7Pose("./yolov7-w6-pose.pt")

    process_webcam(model)


if __name__ == "__main__":
    webcam()
