import csv
import cv2
import numpy as np
from scipy import ndimage
from matplotlib.pyplot import imread

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Activation, Dense, Dropout, Flatten 
from keras.layers import Convolution2D

from keras.layers.pooling import MaxPooling2D 
import math
from random import shuffle


samples = []
with open ('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skips the header line
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            measurements = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = batch_sample[3]
                #image_flipped = np.fliplr(images)
                #images.append(image_flipped)
               
                images.append(center_image)
                angles.append(center_angle)
                

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x/127.5) - 1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu")) 
model.add(Dropout(0.09))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.09))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.09))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(0.09))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(100))
#model.add(Dropout(0.5))
model.add(Dense(50))
#model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=7, verbose=1)

######################################
'''
from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''
#########################################
model.save('model.h5')

from IPython.display import display, Image, SVG
from keras.models import load_model

def loadModel(modelPath):
    """
    Loads the model `modelPath`.
    """
    model = load_model(modelPath)
    return model

loadModel('model.h5').summary()
exit()