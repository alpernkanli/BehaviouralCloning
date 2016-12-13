from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
import pandas as pd

train = pd.read_csv('driving_log.csv')
print(train[0])

def the_model():
    model = Sequential()

    model.add(BatchNormalization())
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