# Testing image classification with CNN
# By Kawisorn Kamtue


import os, argparse
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
from model.vgg_model import *
from keras.utils import np_utils
K.set_image_data_format('channels_last')
from keras import applications
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import theano.ifelse
from theano.ifelse import IfElse, ifelse

def get_image_features(image_path):
    # normalize images
    img = cv2.resize(cv2.imread(image_path), (256, 256)).astype(np.float32)
    x = img_to_array(img)  # this is a Numpy array with shape (256, 256,3)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1,256, 256,3)
    return x
def train_model():

    # Train model and save
    img_width, img_height = 256, 256
    train_data_dir = "data/train"
    validation_data_dir = "data/val"
    nb_train_samples = 50
    nb_validation_samples = 5
    batch_size = 3
    epochs = 20

    model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height,3))


    # Freeze the layers which you don't want to train. 
    # Train only classifiers
    for layer in model.layers:
        layer.trainable = False

    #Adding custom Layers e.g. classifiers
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(3, activation="softmax")(x) # set to num of classes we have

    # creating the final model 
    model_final = Model(input = model.input, output = predictions)

    # compile the model 
    model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

    # Initiate the train and test generators with data Augumentation 
    train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)

    test_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)

    train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size, 
    class_mode = "categorical")

    validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    class_mode = "categorical")
    
    print(validation_generator.class_indices) ## to show indice for each class

    # Save the model according to the conditions  
    checkpoint = ModelCheckpoint("final_model.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    # Train the model 

    model_final.fit_generator(
    train_generator,
    samples_per_epoch = nb_train_samples,
    epochs = epochs,
    validation_data = validation_generator,
    nb_val_samples = nb_validation_samples,
    callbacks = [checkpoint, early])

def load_final_model():
    # Load our model from .h5 file
    model = load_model('final_model.h5')
    return model


if __name__ == "__main__":
    train = True # set to true if you want to train
    if train:
        train_model()
    model = load_final_model() # need to have .h5 file first
    img = cv2.imread('test2.jpg')
    X = get_image_features('test2.jpg')
    Y = model.predict(X,batch_size=5)
    print Y
    #see line 94
    if np.argmax(Y) == 0:
        cv2.imshow('Angkor',img)
    elif np.argmax(Y) == 1:
        cv2.imshow('Chiangmai',img)
    else:
        cv2.imshow('Eiffel',img)
    cv2.waitKey(0)