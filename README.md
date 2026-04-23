# AIPOSE

<div align="center">

[![Downloads](https://static.pepy.tech/personalized-badge/aipose?period=month&units=international_system&left_color=grey&right_color=blue&left_text=PyPi%20Downloads)](https://pepy.tech/project/aipose)
[![Stars](https://img.shields.io/github/stars/Tlaloc-Es/aipose?color=yellow&style=flat)](https://github.com/Tlaloc-Es/aipose/stargazers)
[![Documentation Status](https://readthedocs.org/projects/aipose/badge/?version=latest)](https://aipose.readthedocs.io/en/latest/?badge=latest)

</div>

<p align="center">
    <img src="https://raw.githubusercontent.com/Tlaloc-Es/aipose/master/logo.png" width="200" height="240"/>
</p>

Library to use pose estimation in your projects easily.

## Installation [![PyPI](https://img.shields.io/pypi/v/aipose.svg)](https://pypi.org/project/aipose/)

You can install `aipose` from [Pypi](https://pypi.org/project/aipose/). It's going to install the library itself and its prerequisites as well.

```bash
pip install aipose
```

You can install `aipose` from its source code.

```bash
git clone https://github.com/Tlaloc-Es/aipose.git
cd aipose
pip install -e .
```

## Run demo

Use the following command to run a demo with your cam and YOLOv7 pose estimator,

```bash
posewebcam
```

### Testing without a physical webcam (Linux)

You can simulate a webcam using `v4l2loopback` and `ffmpeg`:

**1. Install dependencies:**

```bash
sudo apt install v4l2loopback-dkms ffmpeg
```

**2. Load the virtual device:**

```bash
sudo modprobe v4l2loopback devices=1 video_nr=0 card_label="FakeWebcam" exclusive_caps=1
```

**3. Feed it with a video or image (in a separate terminal):**

```bash
# Loop a video file
ffmpeg -re -stream_loop -1 -i your_video.mp4 -vf format=yuv420p -f v4l2 /dev/video0

# Or use a static image
ffmpeg -re -loop 1 -i your_image.jpg -vf format=yuv420p -f v4l2 /dev/video0
```

**4. Run the demo in another terminal:**

```bash
posewebcam
```

**5. Unload the virtual device when done:**

```bash
sudo modprobe -r v4l2loopback
```

## Running over a video results

<p align="center">
    <img src="https://raw.githubusercontent.com/Tlaloc-Es/aipose/master/video.gif" width="250" height="200" />
    <img src="https://raw.githubusercontent.com/Tlaloc-Es/aipose/master/video_processed.gif" width="250" height="200" />
</ p>

## How to use

You can check the section notebooks in the repository to check the usage of the library or you can ask in the [Issues section](https://github.com/Tlaloc-Es/aipose/issues).

The examples are:

- [How to draw key points in a video](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/video.ipynb)
- [How to draw key points in a video and store it](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/process_and_save_video.ipynb)
- [How to draw key points in a webcam](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/webcam.ipynb)
- [How to draw key points in a picture](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/plot_keypoints.ipynb)
- [How to capture a frame to apply your business logic](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/custom%20manager.ipynb)
- [How to stop the video stream when anybody raises hands with YOLOv7](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/process_keypoints.ipynb)
- [How to calculate pose similarity with YOLOv7](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/pose_similarity.ipynb)
- [How to turn the pose with YOLOv7](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/pose_similarity.ipynb)
- [How to train a pose classificator](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/Pose_Classificator.ipynb)

## References

- https://github.com/RizwanMunawar/yolov7-pose-estimation

## Support

You can do a donation with the following link.

<a href="https://www.buymeacoffee.com/tlaloc" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

Or you can try to make a pull request with your improvements to the repo.

## Source of videos and images

- Video: [Mikhail Nilov](https://www.pexels.com/video/a-woman-exercising-using-an-exercise-ball-6739975/)

### Images (Pexels)

- [Roman Davayposmotrim](https://www.pexels.com/photo/woman-wearing-black-sports-bra-reaching-floor-while-standing-35987/)
- [Vlada Karpovich](https://www.pexels.com/photo/a-woman-doing-yoga-4534689/)
- [Lucas Pezeta](https://www.pexels.com/photo/woman-doing-yoga-2121049/)
- [Cliff Booth](https://www.pexels.com/photo/photo-of-woman-in-a-yoga-position-4057839/)
- [MART PRODUCTION](https://www.pexels.com/photo/photo-of-a-woman-meditating-8032834/)
- [Antoni Shkraba](https://www.pexels.com/photo/woman-in-blue-tank-top-and-black-leggings-doing-yoga-4662330/)
- [Elina Fairytale](https://www.pexels.com/photo/woman-in-pink-tank-top-and-blue-leggings-bending-her-body-3823074/)
- [Anna Shvets](https://www.pexels.com/photo/graceful-woman-performing-variation-of-setu-bandha-sarvangasana-yoga-pose-5012071/)
- [Miriam Alonso](https://www.pexels.com/photo/calm-young-asian-woman-doing-supine-hand-to-big-toe-yoga-asana-7593010/)
- [Anete Lusina](https://www.pexels.com/photo/concentrated-woman-standing-in-tree-pose-on-walkway-4793290/)
