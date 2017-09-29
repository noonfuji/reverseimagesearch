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
K.set_image_data_format('channels_first')


def get_mean():
	# calculate the mean of training data
    # normalize images
    blue_mean = np.zeros((224, 224), dtype='float')
    green_mean = np.zeros((224, 224), dtype='float')
    red_mean = np.zeros((224, 224), dtype='float')
    path = "./dataset/"
    nb_train = 0
    for image_file_name in os.listdir("./dataset"):
        if image_file_name.endswith(".jpg") or image_file_name.endswith(".jpeg"):
            nb_train = nb_train+1
            image_file_name = path+image_file_name
            im = cv2.resize(cv2.imread(image_file_name), (224, 224))
            blue_mean = blue_mean + im[:,:,0]
            green_mean = green_mean + im[:,:,1]
            red_mean = red_mean + im[:,:,2]
    return [blue_mean.sum()/(224*224*nb_train),green_mean.sum()/(224*224*nb_train),red_mean.sum()/(224*224*nb_train)]
 
    #im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
def get_image_features(image_path):
    # normalize images
    im = cv2.resize(cv2.imread(image_path), (224, 224)).astype(np.float32)
    [b,g,r] = get_mean()
    im[:,:,0] -= b
    im[:,:,1] -= g
    im[:,:,2] -= r
    im = im.transpose((2,0,1)) # change from BGR to RGB
    im = np.expand_dims(im, axis=0) # change to 1 3 224 224
    return im
def prep_data_X():
    nb_train = 17
    X_train = np.zeros((nb_train,3,224,224))
    path = "./dataset/"
    i=0
    for image_file_name in os.listdir("./dataset"):
        if image_file_name.endswith(".jpg") or image_file_name.endswith(".jpeg"):
            image_file_name = path+image_file_name
            im = get_image_features(image_file_name)
            X_train[i,:,:,:]=im
            i=i+1
    return X_train
def prep_data_Y():
    #to be modified
    Y = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1]
    return np_utils.to_categorical(Y)

def train_model():
    model = VGG_16()
    Y_train = prep_data_Y()
    print Y_train
    X_train = prep_data_X()
    model.fit(X_train,Y_train,epochs = 3)
    model.save_weights('weight.h5')
    return model
def load_model():
    weight_path = 'weight.h5'
    model = VGG_16(weight_path)
    #out = model.predict(im)
    #print np.argmax(out)
    return model
if __name__ == "__main__":
    model = train_model()
    #model = load_model()
    Y_predict=model.predict(get_image_features('test.jpg'))
    print Y_predict
