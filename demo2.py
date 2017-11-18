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
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import theano.ifelse
from theano.ifelse import IfElse, ifelse

# def get_image_features(image_path):
#     # normalize images
#     im = cv2.resize(cv2.imread(image_path), (256, 256)).astype(np.float32)
#     """[b,g,r] = get_mean()
#     im[:,:,0] -= b
#     im[:,:,1] -= g
#     im[:,:,2] -= r"""
#     mean_pixel = [103.939, 116.779, 123.68]
#
#     for c in range(3):
#         im[c, :, :] = im[c, :, :] - mean_pixel[c]
#
#     im = im.transpose((2,0,1)) # change from BGR to RGB
#     im = np.expand_dims(im, axis=0) # change to 1 3 224 224
#     return im
# def prep_data_X():
#     nb_train = 20
#     X_train = np.zeros((nb_train,256,256,3))
#     path = "./dataset/"
#     i=0
#     for image_file_name in os.listdir("./dataset"):
#         if image_file_name.endswith(".jpg") or image_file_name.endswith(".jpeg"):
#             image_file_name = path+image_file_name
#             im = get_image_features(image_file_name)
#             X_train[i,:,:,:]=im
#             i=i+1
#     return X_train
# def prep_data_Y():
#     #to be modified
#     Y = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
#     return np_utils.to_categorical(Y)
#
# def train_model():
#     model = vgg16_model()
#     Y_train = prep_data_Y()
#     X_train = prep_data_X()
#     print X_train.shape
#     print "\n"
#     print Y_train.shape
#     nb_epoch = 10
#     batch_size = 5
#     model.fit(X_train, Y_train,
#               batch_size=batch_size,
#               nb_epoch=nb_epoch,
#               shuffle=True
#               )
#     print "Training complete \n"
#     model.save_weights('weight.h5')
#     return model
#
# def load_model():
#     weight_path = 'weight.h5'
#     model = vgg16_model(weight_path)
#     return model
# '''if __name__ == "__main__":
#     model = train_model()
#     #model = load_model()
#
#     Y_predict=model.predict(get_image_features('test1.jpg'),batch_size=5)
#     print Y_predict
#     #img = cv2.imread('test1.jpg')
#     #print Y_predict
#     #if np.argmax(Y_predict) == 0:
#     #    cv2.imshow('Dog',img)
#     #else:
#     #    cv2.imshow('Cat',img)
#     #cv2.waitKey(0)
# '''

if __name__ == "__main__":
    img_width, img_height = 256, 256
    train_data_dir = "data2/train"
    validation_data_dir = "data2/val"
    nb_train_samples = 30
    nb_validation_samples = 8
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
    predictions = Dense(3, activation="softmax")(x) # currently have two classes

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
    

    # Save the model according to the conditions  
    checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    """Y_train = prep_data_Y()
    X_train = prep_data_X()
    print X_train.shape"""

    # Train the model 


    model_final.fit_generator(
    train_generator,
    samples_per_epoch = nb_train_samples,
    epochs = epochs,
    validation_data = validation_generator,
    nb_val_samples = nb_validation_samples,
    callbacks = [checkpoint, early])
