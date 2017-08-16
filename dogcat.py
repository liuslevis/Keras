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

train_dir = './input/dogcat/train_small/'
valid_dir = './input/dogcat/valid_small/'
test_dir = './input/dogcat/test/'

image_width = 150
image_height = 150
image_channels = 3
image_size = (image_width, image_height)
input_shape = (image_width, image_height, image_channels)

LABEL_NUM = 2
LABEL_DOG = 1
LABEL_CAT = 0

def read_paths(train_dir):
    return [train_dir + 'dogs/' + i for i in os.listdir(train_dir + 'dogs/') if 'jpg' in i] + [train_dir + 'cats/' + i for i in os.listdir(train_dir + 'cats/') if 'jpg' in i]

def read_image(path, image_size):
    # import cv2
    # image = cv2.imread(path, cv2.IMREAD_COLOR)
    # image = cv2.resize(image, image_size, interpolation=cv2.INTER_CUBIC)
    with PIL.Image.open(path) as image:
        image = image.resize(image_size, resample=PIL.Image.BICUBIC)
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
    return im_arr

def prep_data(image_paths, input_shape):
    count = len(image_paths)
    data = np.ndarray((count, *input_shape), dtype=np.uint8)

    for i, image_path in enumerate(image_paths):
        image_width, image_height, _ = input_shape
        image = read_image(image_path, (image_width, image_height))
        data[i] = image if image.shape == input_shape else image.T
        if i % 250 == 0: print('Loading image {} of {}'.format(i, count))
    return data

def read_labels(image_paths):
    labels = []
    for i in image_paths:
        if '/dogs/' in i:
            labels.append(LABEL_DOG)
        else:
            labels.append(LABEL_CAT)
    return labels

def read_labels_as_categorical(image_paths):
    return keras.utils.to_categorical(read_labels(image_paths), LABEL_NUM)

train_paths = read_paths(train_dir)
valid_paths = read_paths(valid_dir)

random.shuffle(train_paths)
random.shuffle(valid_paths)

x_train = prep_data(train_paths, input_shape)
x_valid = prep_data(valid_paths, input_shape)

print("Train shape: {}".format(x_train.shape))
print("Valid shape: {}".format(x_valid.shape))

y_train = read_labels(train_paths)
y_valid = read_labels(valid_paths)

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

batch_size = 16
epochs = 100
optimizer = keras.optimizers.Adadelta(lr=1e-1) # RMSprop()
loss = keras.losses.binary_crossentropy # categorical_crossentropy

model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_valid, y_valid),
    # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')],
    )
score = model.evaluate(x_valid, y_valid, verbose=0)
print('valid loss:', score[0])
print('valid accuracy:', score[1])
