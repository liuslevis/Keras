# https://www.kaggle.com/jeffd23/catdognet-keras-convnet-starter
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# vgg16_weights.h5 https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
# https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

# pip3 install h5py

import os, random
import numpy as np
import PIL

from scipy import ndimage

import keras
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation, Conv2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import applications

train_data_dir = './input/dogcat/train_small'
valid_data_dir = './input/dogcat/valid_small'

top_model_weights_path = 'bottleneck_fc_model.h5'
bottleneck_features_train_path = 'bottleneck_features_train.npy'
bottleneck_features_valid_path = 'bottleneck_features_validation.npy'

# dimensions of our images.
img_width, img_height = 150, 150

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open(bottleneck_features_train_path, 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        valid_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open(bottleneck_features_valid_path, 'w'),
            bottleneck_features_validation)

def train_top_model():
    train_data = np.load(open(bottleneck_features_train_path))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open(bottleneck_features_valid_path))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
        loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

save_bottlebeck_features()
train_top_model()

# train_data = np.load(open(bottleneck_features_train_path, 'w'))
# the features were saved in order, so recreating the labels is easy
# train_labels = np.array([0] * 1000 + [1] * 1000)



