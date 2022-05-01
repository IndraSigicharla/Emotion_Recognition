# Emotion Recognition

Object Detection using Haar cascade classifiers is an effective object detection method. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images. The algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier.  

However, such a trained classifier already exists in OpenCV's implementation of "haarcascade_frontalface_default.xml" which is included with the rest of the source code.

The Machine Learning model is trained on the FER-2013 dataset that is available on Kaggle, and all the images from that have been put into the data.zip included with the source code. ([https://www.kaggle.com/datasets/msambare/fer2013])  
