# Demo Code for "Talking Head(?) Anime from a Single Image 4: Improved Model and Its Distillation"

This repository contains demo programs for the "Talking Head(?) Anime from a Single Image 4: Improved Model and Its Distillation" project. Roughly, the project is about a machine learning model that can animate an anime character given only one image. However, the model is too slow to run in real-time. So, it also proposes an algorithm to use the model to train a small machine learning model that is specialized to a character image that can anime the character in real time.

This demo code has two parts.

* **Improved model.** This part gives a model similar to [Version 3](https://github.com/pkhungurn/talking-head-anime-3-demo) of the porject. It has one demo program:

  * The `full_manual_poser` allows the user to manipulate a character's facial expression and body rotation through a graphical user interface.

  There are no real-time demos because the new model is too slow for that.

* **Distillation.** This part allows the user to train small models (which we will refer to as **student models**) to mimic that behavior of the full system with regards to a specific character image. It also allows the user to run these models under various interfaces. The demo programs are:

  * The `distiller` trains a student model given a configuration file, a $512 \times 512$ RGBA character image, and a mask of facial organs.
  * The `distiller_ui` provides a user-friendly interface to the distiller, allowing you to create training configurations and providing useful documentation.
  * The `student_manual_poser` allows the user to control trained student models with a graphical user interface.
  * The `student_ifacialmocap_puppeteer` allows the user to control trained student models with their facial movement, which is captured by the [iFacialMocap](https://www.ifacialmocap.com/) software. To run this software, you must have an iOS device and, of course, iFacialMocap.
  * The `student_mediapipe_puppeteer` allows the user to control trained student models with their facial movement, which is captured by the [Mediapipe FaceLandmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) model. To run this software, you need a web camera.  

## Preemptive FAQs

### What is the program to control character images with my facial movement?

There is no such program in this release. If you want one, try the `ifacialmocap_puppeteer` of [Version 3](https://github.com/pkhungurn/talking-head-anime-3-demo).

### What is the `student_ifacialmocap_puppeteer` and `student_mediapipe_puppeteer` then?

These programs allow you to control "student models" with your movement in real time with moderate hardware requirement.

### What is a student model?

A student model is a small (< 2MB) and fast machine learning model that knows how to animate a specific character image. You can find two student models in the `data/character_models` directory. The [two](https://pkhungurn.github.io/talking-head-anime-4/supplementary/webcam-demo/index.html) [demos](https://pkhungurn.github.io/talking-head-anime-4/supplementary/manual-poser-demo/index.html) on the project website feature 13 students models.

### So, for this release, you can only control two fixed characters in real time?

No. You can create your own student models by using the `distiller`.

### How do I use the `distiller`?

Unless you want to automate the process of training student models, please use the `distiller_ui` instead. It will guide you through the required data preparation and will invoke the `distiller` on your behalf when you are ready.

### How long does it take to create a student model?

Last time I tried, it was about 30 hours on a computer with an Nvidia RTX A6000 GPU.

### Why is this release so hard to use?

[Version 3](https://github.com/pkhungurn/talking-head-anime-3-demo) is arguably easier to use because you can give it an animate and you can control it with your facial movment immediately. However, I was not satisfied with its image quality and speed. 

In this release, I explore a new way of doing things. I added a new preprocessing stage (i.e., training the student models) that has to be done one time per character image. It allows the image to be animated much faster at a higher image quality level.

In other words, it makes the user's life difficult but the engineer/researcher happy. Patient users who are willing to go through the steps, though, would be rewarded with faster animation.

### Can I use a student model from a web browser?

No. A student model created by the `distiller` is a [PyTorch](https://pytorch.org/) model, which cannot run directly in the browser. It needs to be converted to the appropriate format ([TensorFlow.js](https://www.tensorflow.org/js)) first, and the [web](https://pkhungurn.github.io/talking-head-anime-4/supplementary/webcam-demo/index.html) [demos](https://pkhungurn.github.io/talking-head-anime-4/supplementary/manual-poser-demo/index.html) use the converted models. However, The conversion code is not included in this repository. I will not release it unless I change my mind.

## Hardware Requirements

All programs require a recent and powerful Nvidia GPU to run. I developed the programs on a machine with an Nvidia RTX A6000. However, anything after the GeForce RTX 2080 should be fine.

The `student_ifacialmocap_puppeteer` program requires an iOS device that is capable of computing [blend shape parameters](https://developer.apple.com/documentation/arkit/arfaceanchor/2928251-blendshapes) from a video feed. This means that the device must be able to run iOS 11.0 or higher and must have a TrueDepth front-facing camera. (See [this page](https://developer.apple.com/documentation/arkit/content_anchors/tracking_and_visualizing_faces) for more info.) In other words, if you have the iPhone X or something better, you should be all set. Personally, I have used an iPhone 12 mini.

The `student_mediapipe_puppeteer` program requires a web camera.

## Software Requirements

### GPU Driver and CUDA Toolkit

Please update your GPU's device driver and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) that is compatible with your GPU and is newer than the version you will be installing in the next subsection.

### Python and Python Libraries

All programs are written in the [Python](https://www.python.org/) programming languages. The following libraries are required:

* `python` 3.10.11
* `torch` 1.13.1 with CUDA support
* `torchvision` 0.14.1
* `tensorboard` 2.15.1
* `opencv-python` 4.8.1.78
* `wxpython` 4.2.1
* `numpy-quaternion` 2022.4.2
* `pillow` 9.4.0
* `matplotlib` 3.6.3
* `einops` 0.6.0
* `mediapipe` 0.10.3
* `numpy` 1.26.3
* `scipy` 1.12.0
* `omegaconf` 2.3.0

Instead of installing these libraries yourself, you should follow the recommended method to set up a Python environment in the next section.

### iFacialMocap

If you want to use ``ifacialmocap_puppeteer``, you will also need to an iOS software called [iFacialMocap](https://www.ifacialmocap.com/) (a 980 yen purchase in the App Store). Your iOS and your computer must use the same network. For example, you may connect them to the same wireless router.

## Creating Python Environment

### Installing Python

Please install [Python 3.10.11](https://www.python.org/downloads/release/python-31011/). 

I recommend using [`pyenv`](https://github.com/pyenv/pyenv) (or [`pyenv-win`](https://github.com/pyenv-win/pyenv-win) for Windows users) to manage multiple Python versions on your system. If you use `pyenv`, this repository has a `.python-version` file that indicates it would use Python 3.10.11. So, you will be using Python 3.10.11 automatically once you `cd` into the repository's directory.

Make sure that you can run Python from the command line.

### Installing Poetry

Please install [Poetry](https://python-poetry.org/) 1.7 or later. We will use it to automatically install the required libraries. Again, make sure that you can run it from the command line.

### Cloning the Repository

Please clone the repository to an arbitrary directory in your machine.

### Instruction for Linux/OSX Users

1. Open a shell.
2. `cd` to the directory you just cloned the repository too
   ```
   cd SOMEWHERE/talking-head-anime-4-demo
   ```
3. Use Python to create a virtual environment under the `venv` directory.
   ```
   python -m venv venv --prompt talking-head-anime-4-demo
   ```
4. Activate the newly created virtual environment. You can either use the script I provide:
   ```
   source bin/activate-venv.sh
   ```
   or do it yourself:
   ```
   source venv/bin/activate   
   ```
5. Use Poetry to install libraries.
   ```
   cd poetry
   poetry install
   ```

### Instruction for Windows Users

1. Open a shell.
2. `cd` to the directory you just cloned the repository too
   ```
   cd SOMEWHERE\talking-head-anime-4-demo
   ```
3. Use Python to create a virtual environment under the `venv` directory.
   ```
   python -m venv venv --prompt talking-head-anime-4-demo
   ```
4. Activate the newly created virtual environment. You can either use the script I provide:
   ```
   bin\activate-venv.bat
   ```
   or do it yourself:
   ```
   venv\Scripts\activate   
   ```
5. Use Poetry to install libraries.
   ```
   cd poetry
   poetry install
   ```

## Using the programs

