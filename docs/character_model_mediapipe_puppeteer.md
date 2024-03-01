# `character_model_mediapipe_puppeteer`

allows the user to control trained student models with their facial movement, which is captured by a web camera and processed by the [Mediapipe FaceLandmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) model.

## Web Camera

Please make sure that, before you invoke the program, your computer has a web camera plugged in. The program will use a web camera, but it does not allow you to specify which. In case your machine has more than one web camera, you can turn off all camera except the one that you want to use. 

You can also inspect the [source code](../src/tha4/app/character_model_mediapipe_puppeteer.py) and change the 

```
    video_capture = cv2.VideoCapture(0)
```

line to choose a particular camera that you want to use.

## Invoking the Program

Make sure you have (1) created a Python environment and (2) downloaded model files as instruction in the [main README file](../README.md).

### Instruction for Linux/OSX Users

1. Open a shell.
2. `cd` to the repository's directory.
   ```
   cd SOMEWHERE/talking-head-anime-4-demo
   ```
3. Run the program.
   ```
   bin/run src/tha4/app/character_model_mediapipe_puppeteer.py
   ```   

### Instruction for Windows Users

1. Open a shell.
2. `cd` to the repository's directory.
   ```
   cd SOMEWHERE\talking-head-anime-4-demo
   ```
3. Run the program.
   ```
   bin\run.bat src\tha4\app\character_model_mediapipe_puppeteer.py
   ```   
