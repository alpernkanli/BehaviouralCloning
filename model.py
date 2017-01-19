import cv2
import pandas as pd
import numpy as np
import json


from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.models import Sequential
from keras.optimizers import Adam

from keras.utils.visualize_util import plot



log = pd.read_csv('./data/driving_log.csv')


rows, cols, ch = 64, 64, 3
batch_size = 100
split_size = 0.1
samples_per_epoch = 20000
angle_offset = 0.27
validation_samples = 2000
epoch_count = 4

log = log.sample(frac=1).reset_index(drop=True)

training_data = log.loc[0:(log.shape[0]*(1.0-split_size)) - 1]
validation_data = log.loc[log.shape[0]*(1.0-split_size):]


# For learning roads with different brightness
def random_V(image, angle):
    HSV_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_v = 0.25 + np.random.uniform()
    HSV_image[:,:,2] = HSV_image[:,:,2]*random_v
    image = cv2.cvtColor(HSV_image, cv2.COLOR_HSV2RGB)
    return image, angle

# For learning roads with different main color
def random_H(image, angle):
    HSV_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_h = 0.2 + np.random.uniform()
    HSV_image[:,:,0] = HSV_image[:,:,0]*random_h
    image = cv2.cvtColor(HSV_image, cv2.COLOR_HSV2RGB)
    return image, angle

# Because for a turn there could be more than one possible angle
def angle_jitter(image, angle):
    angle = angle + 0.05*(np.random.uniform() - 0.5)
    return image, angle

# To generate more data
def random_flip(image, angle):
    if np.random.random() > 0.4:
        image = cv2.flip(image, 1)
        angle = angle*(-1.0)
    return image, angle

# 2 more image for every image.
def augment_and_process(row):
    angle = row['steering']
    camera = np.random.choice(['center', 'left', 'right'])
    
    if camera == 'right':
        angle -= angle_offset
    elif camera == 'left':
        angle += angle_offset
    
    path = row[camera]
    datapath = './data/' + path
    datapath = datapath.replace(" ", "")
    
    image = cv2.imread(datapath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image, angle = random_V(image, angle)
    image, angle = random_H(image, angle)
    image, angle = angle_jitter(image, angle)
    image, angle = random_flip(image, angle)
    
    image = image[55:135, 0:320]
    image = cv2.resize(image, (cols, rows))
    image = image.astype(np.float32)
    return image, angle

def batch_generator(data):
    batch_count = data.shape[0] // batch_size
    i = 0
    while 1:
        batch_features = np.zeros((batch_size, rows, cols, ch), dtype=np.float32)
        batch_labels = np.zeros((batch_size,), dtype=np.float32)
        
        j = 0
        for _, row in data.loc[i*batch_size: (i+1)*batch_size - 1].iterrows():
            batch_features[j], batch_labels[j] = augment_and_process(row)
            j += 1
        
        i += 1
        if i == batch_count - 1:
            i = 0
        yield batch_features, batch_labels


def the_model():
    model = Sequential()
    
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(rows, cols, ch)))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same'))
    model.add(ELU())
    
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))
    
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(1024, name='Dense0'))
    model.add(Dropout(0.3))
    model.add(ELU())
    
    model.add(Dense(512, name='Dense1'))
    model.add(ELU())
    
    model.add(Dense(128, name='Dense2'))
    model.add(ELU())
    
    model.add(Dense(1, name='Out'))
    
    model.compile(optimizer="adam", loss="mse")
    return model


def save_parameters(m):
    m.save_weights('model.h5')
    json_file = open('model.json', mode='w')
    json.dump(m.to_json(), json_file)


model = the_model()
plot(model, to_file='model.png', show_shapes=True)


model.fit_generator(batch_generator(training_data), 
                    samples_per_epoch= samples_per_epoch,
                    nb_epoch=epoch_count,
                    verbose=1,
                    validation_data=batch_generator(validation_data),
                    nb_val_samples=validation_samples)


save_parameters(model)




