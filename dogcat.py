# https://www.kaggle.com/jeffd23/catdognet-keras-convnet-starter
import os, random
import numpy as np
import PIL

from scipy import ndimage

import keras
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation, Conv2D
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

TRAIN_DIR = './input/dogcat/train/'
TEST_DIR = './input/dogcat/test/'

ROWS = 64
COLS = 64
CHANNELS = 3
# INPUT_SHAPE = (CHANNELS, ROWS, COLS) # Theano
INPUT_SHAPE = (ROWS, COLS, CHANNELS) # Tensorflow

LABEL_NUM = 2
LABEL_DOG = 1
LABEL_CAT = 0

test_paths = [TEST_DIR + i for i in os.listdir(TEST_DIR) if 'jpg' in i]

dog_paths = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'dog' in i and 'jpg' in i]
cat_paths = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'cat' in i and 'jpg' in i]

split_index = int(len(dog_paths) * 0.8)
train_paths = dog_paths[:split_index]  + cat_paths[:split_index]
valid_paths = dog_paths[-split_index:] + cat_paths[-split_index:]

random.shuffle(train_paths)
random.shuffle(valid_paths)

def read_image(path):
    # import cv2
    # image = cv2.imread(path, cv2.IMREAD_COLOR)
    # image = cv2.resize(image, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    with PIL.Image.open(path) as image:
        image = image.resize((ROWS, COLS), resample=PIL.Image.BICUBIC)
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
    return im_arr

def prep_data(image_paths):
    count = len(image_paths)
    data = np.ndarray((count, *INPUT_SHAPE), dtype=np.uint8)

    for i, image_path in enumerate(image_paths):
        image = read_image(image_path)
        data[i] = image if image.shape == INPUT_SHAPE else image.T
        if i % 250 == 0: print('Loading image {} of {}'.format(i, count))
    return data

def read_labels(image_paths):
    labels = []
    for i in image_paths:
        if 'dog' in i:
            labels.append(LABEL_DOG)
        else:
            labels.append(LABEL_CAT)
    return labels

x_train = prep_data(train_paths)
x_valid = prep_data(valid_paths)
x_test  = prep_data(test_paths)

print("Train shape: {}".format(x_train.shape))
print("Valid shape: {}".format(x_valid.shape))
print("Test  shape: {}".format(x_test.shape))

y_train = read_labels(train_paths)
y_valid = read_labels(valid_paths)

y_train = keras.utils.to_categorical(y_train, LABEL_NUM)
y_valid = keras.utils.to_categorical(y_valid , LABEL_NUM)

batch_size = 50
epochs = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), border_mode='same', activation='relu', input_shape=INPUT_SHAPE))
model.add(Conv2D(32, kernel_size=(3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), border_mode='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), border_mode='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), border_mode='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(LABEL_NUM))
model.add(Activation('sigmoid'))

model.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_valid, y_valid))
score = model.evaluate(x_valid, y_valid, verbose=0)
print('valid loss:', score[0])
print('valid accuracy:', score[1])

pred = model.predict(x_test, batch_size=batch_size, verbose=1)