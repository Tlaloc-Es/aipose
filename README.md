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
## Instalation [![PyPI](https://img.shields.io/pypi/v/aipose.svg)](https://pypi.org/project/aipose/)

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

## Running over a video results

<p align="center">
    <img src="https://raw.githubusercontent.com/Tlaloc-Es/aipose/master/video.gif" width="250" height="200" />
    <img src="https://raw.githubusercontent.com/Tlaloc-Es/aipose/master/video_processed.gif" width="250" height="200" />
</ p>

## How to use

You can check the section notebooks in the repository to check the usage of the library or you can ask in the [Issues section](https://github.com/Tlaloc-Es/aipose/issues).

The examples are:

* [How to draw key points in a video](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/video.ipynb)
* [How to draw key points in a video and store it](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/process_and_save_video.ipynb)
* [How to draw key points in a webcam](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/webcam.ipynb)
* [How to draw key points in a picture](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/plot_keypoints.ipynb)
* [How to capture a frame to apply your business logic](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/custom%20manager.ipynb)
* [How to stop the video stream when anybody raises hands with YOLOv7](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/process_keypoints.ipynb)
* [How to calculate pose similarity with YOLOv7](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/pose_similarity.ipynb)
* [How to turn the pose with YOLOv7](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/pose_similarity.ipynb)
* [How to train a pose classificator](https://github.com/Tlaloc-Es/aipose/blob/master/notebooks/Pose_Classificator.ipynb)

## References

* https://github.com/RizwanMunawar/yolov7-pose-estimation

## Support

You can do a donation with the following link.

<a href="https://www.buymeacoffee.com/tlaloc" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

Or you can try to make a pull request with your improvements to the repo.

## Source of videos and images

* Video by Mikhail Nilov: https://www.pexels.com/video/a-woman-exercising-using-an-exercise-ball-6739975/

In folder notebooks/poses/

* [Photo by Roman Davayposmotrim: https://www.pexels.com/photo/woman-wearing-black-sports-bra-reaching-floor-while-standing-35987/](notebooks/poses/pexels-roman-davayposmotrim-35987.jpg)
* [Photo by Vlada Karpovich: https://www.pexels.com/photo/a-woman-doing-yoga-4534689/](pexels-roman-davayposmotrim-35987.jpg)
* [Photo by Lucas Pezeta: https://www.pexels.com/photo/woman-doing-yoga-2121049/](notebooks/poses/pexels-lucas-pezeta-2121049)
* [Photo by Cliff  Booth: https://www.pexels.com/photo/photo-of-woman-in-a-yoga-position-4057839/](pnotebooks/poses/exels-cliff-booth-4057839.jpg)
* [Photo by Cliff  Booth: https://www.pexels.com/photo/photo-of-woman-meditating-alone-4056969/](notebooks/poses/pexels-cliff-booth-4056969.jpg)
* [Photo by MART  PRODUCTION: https://www.pexels.com/photo/photo-of-a-woman-meditating-8032834/](notebooks/poses/pexels-mart-production-8032834.jpg)
* [Photo by Antoni Shkraba: https://www.pexels.com/photo/woman-in-blue-tank-top-and-black-leggings-doing-yoga-4662330/](notebooks/poses/pexels-antoni-shkraba-4662330.jpg)
* [Photo by MART  PRODUCTION: https://www.pexels.com/photo/woman-wearing-a-sports-bra-8032742/](notebooks/poses/pexels-mart-production-8032742.jpg)
* [Photo by Elina Fairytale: https://www.pexels.com/photo/woman-in-pink-tank-top-and-blue-leggings-bending-her-body-3823074/](notebooks/poses/pexels-elina-fairytale-3823074.jpg)
* [Photo by Cliff  Booth: https://www.pexels.com/photo/photo-of-woman-stretching-her-legs-4057525/](notebooks/poses/pexels-cliff-booth-4057525.jpg)
* [Photo by Mikhail Nilov: https://www.pexels.com/photo/woman-standing-in-a-bending-position-on-a-box-6740089/](notebooks/poses/pexels-mikhail-nilov-6740089.jpg)
* [Photo by cottonbro studio: https://www.pexels.com/photo/woman-in-black-sports-bra-and-black-panty-doing-yoga-4323290/](notebooks/poses/pexels-cottonbro-studio-4323290.jpg)
* [Photo by ArtHouse Studio: https://www.pexels.com/photo/photo-of-man-bending-his-body-4334910/](notebooks/poses/pexels-arthouse-studio-4334910.jpg)
* [Photo by Anna Shvets: https://www.pexels.com/photo/graceful-woman-performing-variation-of-setu-bandha-sarvangasana-yoga-pose-5012071/](notebooks/poses/pexels-anna-shvets-5012071.jpg)
* [Photo by Miriam Alonso: https://www.pexels.com/photo/calm-young-asian-woman-doing-supine-hand-to-big-toe-yoga-asana-7593010/](notebooks/poses/pexels-miriam-alonso-7593010.jpg)
* [Photo by Miriam Alonso: https://www.pexels.com/photo/anonymous-sportswoman-doing-stretching-exercise-during-yoga-session-7593002/](notebooks/poses/pexels-miriam-alonso-7593002.jpg)
* [Photo by Miriam Alonso: https://www.pexels.com/photo/fit-young-asian-woman-preparing-for-handstand-during-yoga-training-at-home-7593004/](notebooks/poses/pexels-miriam-alonso-7593004.jpg)
* [Photo by Anete Lusina: https://www.pexels.com/photo/concentrated-woman-standing-in-tree-pose-on-walkway-4793290/](notebooks/poses/pexels-anete-lusina-4793290.jpg)
* [Photo by Miriam Alonso: https://www.pexels.com/photo/faceless-sportive-woman-stretching-back-near-wall-7592982/](notebooks/poses/pexels-miriam-alonso-7592982.jpg)

