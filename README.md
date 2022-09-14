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
We looked at the State of the Art models for Lip reading. And identified two models:\
- https://arxiv.org/abs/2201.01763 : https://github.com/facebookresearch/av_hubert
- https://arxiv.org/abs/2209.01383v1 : https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
Both of the models use a DL arch. to predict a logit of 500 labels. This is not very good in terms of doing speech recognition, as the normla human vocabulary for english consists of upwards of 5000 words, But since we're working with lip reading and doing a phoneme level prediction is not so easy, this is a good (State of the Art) technique of predicting words spoken. Moreover, in this case, predicting word boundries is also an issue. These models were trained on a BBC dataset called "Lip Reading in the Wild". which essentially has videos comtaining single words, so the model is very good in predicting those words, but the models doesn't generalize well. More research in a better dataset is required to make it more accurate. For the purpose of demosntrating the simultaneous use of audio and video frames, these models are good enough.\ Upon comparision on various testing examples taken from youtube, the AVHuber model by Facebook (1st) works better than the TCN model(2nd). 

## Model Exporting
Model exporting refers to the conversion of pytorch .pth model to ONNX model by using `torch.onnx`. The procedure traces all the computations happening while calling a model's forward function and makes a computation graph for the whole calculation, and finally write the graph using nodes from ONNX framework. \ In our case, the AVHubert model is written as a part of fairseq framwork and it uses functions and torch modules defined in the fairseq modeule, which are notoriously very hard to export to ONNX format. There's a whole thread for this on fairseq's github page. We anyway tried to export the AVHubert model by hardcoding various features, modules and pieces of code, However after spending a lot of time on that, we realised that it's not working for this model. We eventually had to switch to the TCN model, which, when exported was behaving weirdly in opencv and we were not able to forward the model because of some nodes being not correct. This was, however, fixed easily by hardcoding and rewriting the main class. We were able to export the audio and video models from this repo. But the output given by the model uwas not same as the pytorch model. It tok us some time to figure out that in OpenCV dnn module, it automatically tries to fuse layers and because of that we were getting wrong result. We eventually fixed it by disabling the layer fusion.

## Implementation details
### Running models on realtime camera input and checking model generalization
We tried running the models on various video snippits taken from news telecasts from youtube. Both models perform decently, considering the limited vocabolary of the models. The input of "we're going to check back in with" was predicted as "we're going to try to pack in" by the AVHuber model. Where's the TCN model for an input of "It's that simple" predicted, "Against Simply". Both of these are pretty impressive considering the models only worked on video frames and no audio. \ Final model exported by our script ensembles the audio and video model using a simple weighted sum of both the outputs. The weight can be specified while exporting the model. The ensembling is handled inside the onnx graph and we don't have to tajke care of that in the actual functions in opencv. \ For the model to work with openCV, we also had to write some preprocessing for video. Firstly, the model expects the audio to be normalized. Secondly, the model expects a Black and White mouth ROI in 98x98 frame size, with mouth region aligned to the center. This was originally done using dlib in the original repo. Which essentially uses the face landmarks to identify the mouth region, align it by transforming the image and coordinates and crops the image into 98x98 size portion of mouth ROI. We chose to do this using YUNet model, which is available in OpenCV in ONNX format for face detection and facial keypoint identification. Given the different format of the keypoints used by both YUNet and dlib, it took some trial and error to figure out a good way to do the same thing using YUnet in OpenCV. Eventually I implemented all the preprocessing operations in numpy.  

### Exporting torch model to onnx
The actual code to export model can be found inside a github gist at: https://gist.github.com/spazewalker/b8ab3eabc96ffcb30218cbb6f6ea09b3

### Processing Functions
The preprocessing function takes the keypoints identified by YUNet and extracts the two kaypoints for lips and uses an image rotation and transformation to align these keypoints to center and simultaneously transforms other keypoints. these keypoints are eventually used to crop a mouth ROI for each frame. Normalised audio is simply fed into the model in the raw format, with no preprocessing such as mel features, stft etc.

### Usage of AudioIO to retrieve audio and video simultaneously in openCV
Here is a piece of code to extract the audio and video simultaneously frame by frame in OpenCV using `CAP_MSMF`.
```python
import numpy as np
import cv2 as cv
source = 'test.mp4' # pass 0 to read from webcam
params = np.asarray([cv.CAP_PROP_AUDIO_STREAM, 0, # read audio from stream 0
                cv.CAP_PROP_VIDEO_STREAM, 0, # read video frames from stream 0
                cv.CAP_PROP_AUDIO_DATA_DEPTH, cv.CV_32F,
                cv.CAP_PROP_AUDIO_SAMPLES_PER_SECOND, self.samplingRate
                ])
cap.open(source, cv.CAP_MSMF, params)
if not cap.isOpened():
            print('Cannot open video source')
            exit(1)

audioBaseIndex = int(self.cap.get(cv.CAP_PROP_AUDIO_BASE_INDEX))
audioChannels = int(self.cap.get(cv.CAP_PROP_AUDIO_TOTAL_CHANNELS))
if cap.isOpened():
    while self.cap.grab():
        ret, frame = cap.retrieve()
        audioFrame = np.asarray([])
        audioFrame = cap.retrieve(audioFrame, audioBaseIndex)
        audioFrame = audioFrame[1][0] if audioFrame is not None else None
        # process video and audio here
```
## Future work
The goals included an OpenCV sample for AV speech-recognition using AudioIO module and a pre-trained onnx model running in real time. The "Real time" part is constrained by the compute power of the system. However the video file inference in opencv works at par with the torch model. In the future, we can try to work on tyhe AVHuber model and trying to rewrite the whole config og the model while preserving the weights to export it to ONNX.

## What does this repo contain?
This repository contains the Jupyter notebooks that were used to experiment with the code along with some scripts to make life easier. Most of the work was done inside the colab notebooks, links to which are given below:\
- https://colab.research.google.com/drive/1JBVswK5ujWJZj2wnWzUyxqxbc3jkgFod?usp=sharing
- https://colab.research.google.com/drive/1eFQiULlT-8zE4CxLIRt8KzKVbeo51-v0?usp=sharing
- https://colab.research.google.com/drive/1awBCZ5O6uAT32cHvufNWad5m6q26TuqQ?usp=sharing
- https://colab.research.google.com/drive/1awBCZ5O6uAT32cHvufNWad5m6q26TuqQ?usp=sharing
Since these were only for experementing with models, the code is pretty messy.
