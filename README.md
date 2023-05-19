# Handwritten-Digit-Recognizer

This code utilizes Convolutional Neural Networks (ConvNets) to achieve a classification accuracy of over 99% in identifying handwritten numbers.

The model was trained using the Handwritten Number Dataset from Kaggle. This dataset, consisting of 70,000 images with dimensions of 28x28 pixels, and labeled with numbers from 0 to 9, was used to develop a program capable of recognizing handwritten numbers on paper. This program is well-suited for real-life scenarios.

During training, 60,000 images from the dataset were used, while the remaining images were evenly split between the validation and test datasets.

Keras API with Tensorflow in the backend was used to build the Deep Neural Network. OpenCV was used to detect multiple digits in a single image using Contours.
Model Architecture:
Conv2D=>MaxPool2D=>Conv2D=>Conv2D=>MaxPool2D=>Dense=>Dense

Training Accuracy: 99.97%

Validation Accuracy: 99.28%

Test Accuracy: 99.30%
## Getting Started

### Dependencies

* Jupyter Notebook required

* Python Libraries

    - Imutils
    - Tensorflow
    - Keras
    - Numpy
    - cv2
    - os
    - Scikit-Learn
    - Matplotlib

### Installing

* Download Jupyter Notebook

* No further installation


### Executing program

There are 4 phases of this program, run each of them.

* Train a Deep Convolutional Neural Network to recognize numbers 0-9 using the Handwritten Number Dataset

* Take a Test image with numbers written on a white plain sheet.

* Localize and detect multiple numbers in the in the Binary Threshold image using Contours Detection method of OpenCV 

* Classify the selected Contours using the Trained DCNN model

## Help

Installing the libraries beforehand will solve most issues

## Authors

Contributors names and contact info 
ex. [@shubham241201](https://github.com/shubham241201)

## Version History

* 0.1
    * Initial Release

## License

GNU General Public License v3.0
