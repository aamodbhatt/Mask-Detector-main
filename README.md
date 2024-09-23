# Mask Detection using TensorFlow

This project implements a Convolutional Neural Network (CNN) in Python using TensorFlow to classify images as either containing a mask or not. The model is trained on a dataset of images, and the classification task is performed on new images after resizing and normalizing the data.

## Introduction
 This project uses a Convolutional Neural Network (CNN) to classify images into two categories:

1. Images of people wearing a mask.
2. Images of people not wearing a mask.

It resizes the images to a standard shape (50x50 pixels), normalizes the pixel values, and uses TensorFlow to train and predict the results

Install the required dependencies before running the project: 
```
pip install tensorflow numpy pillow
```

## Training the Model
The script automatically processes the dataset and trains the CNN for 100 epochs. The training images are normalized to have pixel values in the range `[0, 1]`.

## Example

Example output for prediction: 
```
Prediction for mask image: [[0.99, 0.01]]
Prediction for no-mask image: [[0.02, 0.98]]
```
