# Testing image classification with CNN
# By Kawisorn Kamtue


import os, argparse
import cv2, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
from model.vgg_model import *
from keras.utils import np_utils
K.set_image_data_format('channels_last')
from keras import applications
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import theano.ifelse
from theano.ifelse import IfElse, ifelse
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
import sys
import math

input_image1 = "test1.jpg"
input_image2 = "test9.jpg"

model = applications.VGG19(weights="imagenet", include_top=False)
# model = applications.VGG19(weights="imagenet")
# model = VGG16(weights="imagenet")

# Freeze the layers which you don't want to train.
# Train only classifiers
# for layer in model.layers:
#     layer.trainable = False

image1 = image_utils.load_img(input_image1, target_size=(224, 224))
image1 = image_utils.img_to_array(image1)
image1 = np.expand_dims(image1, axis=0)
image1 = preprocess_input(image1)

image2 = image_utils.load_img(input_image2, target_size=(224, 224))
image2 = image_utils.img_to_array(image2)
image2 = np.expand_dims(image2, axis=0)
image2 = preprocess_input(image2)

x1 = model.predict(image1)
x2 = model.predict(image2)

# print(x1.shape)

def euclidean(x, y):
    d = math.sqrt(np.sum([(a - b) ** 2 for a, b in zip(x, y)]))

    return d

print("Euclidean distance: ", input_image1, "vs", input_image2, euclidean(x1, x2))

# def chi2_distance(histA, histB, eps=1e-10):
#     # compute the chi-squared distance
#     d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
#                       for (a, b) in zip(histA, histB)])
#
#     # return the chi-squared distance
#     return d
#
# print("Chi squared distance: ", chi2_distance(x1, x2))