# GSoC 2022 : Audio Visual Speech Recognition using OpenCV

## Overview
Mentor : Liubov Batanina @l-bat and Jia WU\
Proposal for the project : [https://summerofcode.withgoogle.com/programs/2022/projects/vSBH0gCK](https://summerofcode.withgoogle.com/programs/2022/projects/vSBH0gCK)\
Link for Pull Request : [https://github.com/opencv/opencv/pull/22181](https://github.com/opencv/opencv/pull/22181)\
Link to pre-trained model : [https://drive.google.com/drive/folders/1oO5vUbzHFmovKTIaDyMSc_ivHb5EBhjB?usp=sharing](https://drive.google.com/drive/folders/1oO5vUbzHFmovKTIaDyMSc_ivHb5EBhjB?usp=sharing)
Code to export onnx model from torch: [https://gist.github.com/spazewalker/b8ab3eabc96ffcb30218cbb6f6ea09b3](https://gist.github.com/spazewalker/b8ab3eabc96ffcb30218cbb6f6ea09b3)

OpenCV is used extensively in computer vision, tackling almost all major computer vision problems. Many Modern CV applications work with video frames. However, these applications can be improved by using an ensemble of video frames with the accompanying audio signal, increasing the efficiency of the signal used for prediction. OpenCV is actively working on the support for audio processing, thereby making the proposed application possible out of the box in OpenCV. After the incorporation of the speech recognition sample in OpenCV, The goal of this project is to add real-time speech recognition samples using audio and visual data streams by using an ensemble of lip-reading techniques and speech recognition techniques.

## Objectives
The objectives were:
* Export the ensembled pytorch models in onnx format for prediction
* Implement Processing functions in OpenCV and Numpy
* Comparing the results from OpenCV sample and original model
* Demonstrate the use of AudioIO using `CAP_MSMF` for simultaneous audio and video processing in OpenCV

## Model Selection

## Model Exporting

## Implementation details
### Running models on realtime camera input and checking model generalization

### Exporting torch model to onnx

### Processing Functions
#### Video Pre-processing
#### Other
#### Usage of AudioIO to retreve audio and video simultaneously in openCV
## Future work
The goals included an OpenCV sample for AV speech-recognition using AudioIO module and a pre-trained onnx model running in real time. The "Real time" part is constrained by the compute power of the system. However the video file inference in opencv works at par with the torch model.

## What does this repo contain?
This repository contains the Jupyter notebooks that were used to experiment with the code along with some scripts to make life easier. Most of the work was done inside the colab notebooks, links to which are given below:\
- https://colab.research.google.com/drive/1JBVswK5ujWJZj2wnWzUyxqxbc3jkgFod?usp=sharing
- https://colab.research.google.com/drive/1eFQiULlT-8zE4CxLIRt8KzKVbeo51-v0?usp=sharing
- https://colab.research.google.com/drive/1awBCZ5O6uAT32cHvufNWad5m6q26TuqQ?usp=sharing
- https://colab.research.google.com/drive/1awBCZ5O6uAT32cHvufNWad5m6q26TuqQ?usp=sharing
Since these were only for experementing with models, the code is pretty messy.
