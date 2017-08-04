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

dog_paths = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'dog.' in i and 'jpg' in i][:1400]
cat_paths = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'cat.' in i and 'jpg' in i][:1400]

split_index = int(len(dog_paths) * 0.71)
train_paths = dog_paths[:split_index] + cat_paths[:split_index]
valid_paths = dog_paths[split_index:] + cat_paths[split_index:]
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
        if 'dog.' in i:
            labels.append(LABEL_DOG)
        else:
            labels.append(LABEL_CAT)
    return labels

def read_labels_as_categorical(image_paths):
    return keras.utils.to_categorical(read_labels(image_paths), LABEL_NUM)


x_train = prep_data(train_paths)
x_valid = prep_data(valid_paths)

print("Train shape: {}".format(x_train.shape))
print("Valid shape: {}".format(x_valid.shape))


y_train = read_labels(train_paths)
y_valid = read_labels(valid_paths)


###############
# batch_size = 16
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# valid_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     './input/dogcat/train_small',
#     target_size=(150,150),
#     batch_size=batch_size,
#     class_mode='binary')

# valid_generator = valid_datagen.flow_from_directory(
#     './input/dogcat/valid_small',
#     target_size=(150,150),
#     batch_size=batch_size,
#     class_mode='binary'
#     )
# model.fit_generator(
#         train_generator,
#         steps_per_epoch=2000 // batch_size,
#         epochs=50,
#         validation_data=valid_generator,
#         validation_steps=800 // batch_size)
# model.save_weights('first_try.h5')  # always save your weights after training or during training

###########

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), border_mode='same', activation='relu', input_shape=INPUT_SHAPE))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

batch_size = 50
epochs = 100
optimizer = keras.optimizers.RMSprop(lr=1e-1)
# optimizer = keras.optimizers.Adadelta(lr=1e-1)
loss = keras.losses.binary_crossentropy
# loss = keras.losses.categorical_crossentropy

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

# test_paths = [TEST_DIR + i for i in os.listdir(TEST_DIR) if 'jpg' in i]
# x_test  = prep_data(test_paths)
# print("Test  shape: {}".format(x_test.shape))
# pred = model.predict(x_test, batch_size=batch_size, verbose=1)