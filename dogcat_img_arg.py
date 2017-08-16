# https://www.kaggle.com/jeffd23/catdognet-keras-convnet-starter
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

import os, random
import numpy as np
import PIL

from scipy import ndimage

import keras
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation, Conv2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

train_dir = './input/dogcat/train_small'
valid_dir = './input/dogcat/valid_small'
test_dir  = './input/dogcat/test/'

num_train_samples = 2000
num_valid_samples = 1600

image_width = 150
image_height = 150
image_channels = 3
image_size = (image_width, image_height)
input_shape = (image_width, image_height, image_channels)

batch_size = 16
epochs = 50

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    train_dir,  # this is the target directory
    target_size=image_size,  # all images will be resized to 150x150
    batch_size=batch_size,
    class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    valid_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), border_mode='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(
    loss=keras.losses.binary_crossentropy, # categorical_crossentropy
    optimizer=keras.optimizers.Adadelta(lr=1e-1), # RMSprop()
    metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=num_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_valid_samples // batch_size,
    )

model.save_weights('./output/first_try.h5')

# model.summary()
