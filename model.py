from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import numpy as np

def prepare_data():
    image_list = []
    logfile = pd.read_csv('./data/driving_log.csv', sep=',')
    for image_path in logfile['center']:
        image_list.append(cv2.imread('./data/' + image_path))
    labels = logfile['steering']
    image_data = np.asarray(image_list, dtype=np.float64)
    print(image_data.shape)
    print(image_data)
    print(labels.shape)
    return image_data, labels

def the_model():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation(activation='relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation(activation='relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation(activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation(activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Activation(activation='relu'))
    model.add(Dense(1164))
    model.add(Activation(activation='relu'))
    model.add(Dense(100))
    model.add(Activation(activation='relu'))
    model.add(Dense(50))
    model.add(Dropout(0.4))
    model.add(Activation(activation='relu'))
    model.add(Dense(10))
    model.add(Activation(activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model

def train_model(X_train, y_train, X_valid, y_valid, batches, epochs):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        zca_whitening=True
    )

    datagen.fit(X_train)
    model = the_model()
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batches),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=epochs,
                        validation_data=(X_valid, y_valid))

    save_parameters(model)

def save_parameters(the_model):
    the_model.save_weights('model.h5')
    json_file = open('model.json', mode='rw')
    json_file.write(the_model.to_json())


batches = 100
epochs = 5

X_train, y_train = prepare_data()
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25)
train_model(X_train, y_train, X_valid, y_valid, batches, epochs)

