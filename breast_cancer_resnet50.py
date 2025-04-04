import os   # provides a way to interact with the operating system like interacting with files, running commands, etc.
import cv2  # OpenCV library in python, for image processing object detection, video analysis
import numpy as np  # for numerical computation, provides support for multi-dimensional arrays and matrices
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, MaxPool2D
from tensorflow.keras.layers import Flatten

# from tensorflow.keras.utils import np_utils

from sklearn.metrics import accuracy_score
