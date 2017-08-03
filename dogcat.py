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
INPUT_SHAPE = (CHANNELS, ROWS, COLS)

LABEL_NUM = 2
LABEL_DOG = 1
LABEL_CAT = 0

# train_paths = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'jpg' in i]
# test_paths  = [TEST_DIR + i  for i in os.listdir(TEST_DIR)  if 'jpg' in i]

dog_paths = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'dog' in i and 'jpg' in i]
cat_paths = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'cat' in i and 'jpg' in i]


train_paths = dog_paths[:100]  + cat_paths[:100]
test_paths  = dog_paths[-100:] + cat_paths[-100:]

random.shuffle(train_paths)
test_paths = test_paths[:25]

# def read_image_cv(path):
#     import cv2
#     image = cv2.imread(path, cv2.IMREAD_COLOR)
#     return cv2.resize(image, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def read_image(path):
    with PIL.Image.open(path) as image:
        image = image.resize((ROWS, COLS), resample=PIL.Image.BICUBIC)
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
    return im_arr

def prep_data(image_paths):
    count = len(image_paths)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_path in enumerate(image_paths):
        image = read_image(image_path)
        data[i] = image.T
        if i % 250 == 0: print('Processed {} of {}'.format(i, count))
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
x_test  = prep_data(test_paths)

print("Train shape: {}".format(x_train.shape))
print("Test shape: {}".format(x_test.shape))

y_train = read_labels(train_paths)
y_test  = read_labels(test_paths)

y_train = keras.utils.to_categorical(y_train, LABEL_NUM)
y_test  = keras.utils.to_categorical(y_test , LABEL_NUM)

batch_size = 10
epochs = 12

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', input_shape=INPUT_SHAPE))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

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
    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
