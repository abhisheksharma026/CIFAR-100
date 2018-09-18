# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 22:28:52 2018

@author: Abhishek 
"""
# https://www.cs.toronto.edu/~kriz/cifar.html

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

meta = unpickle('cifar-100-python/meta')
# type(meta)
meta.keys()

fine_label_names = [label.decode('utf8') for label in meta[b'fine_label_names']]
print(fine_label_names) # all 100 names

train = unpickle('cifar-100-python/train')
train.keys()

filenames = [t.decode('utf8') for t in train[b'filenames']]
fine_labels = train[b'fine_labels']
train_data = train[b'data']
batch_label = train[b'batch_label']
coarse_labels = train[b'coarse_labels']


X = train_data.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

plt.imshow(X[100])

plt.imshow(X[1])

test = unpickle('cifar-100-python/test')
test.keys()

test_filenames = [t.decode('utf8') for t in test[b'filenames']]
test_fine_labels = test[b'fine_labels']
test_data = test[b'data']
test_batch_label = test[b'batch_label']
test_coarse_labels = test[b'coarse_labels']

from sklearn.preprocessing import LabelBinarizer 

lb = LabelBinarizer() 
fine_labels = lb.fit_transform(fine_labels)

lb_test = LabelBinarizer() 
test_fine_labels = lb_test.fit_transform(test_fine_labels)


X_train = train_data.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")/255

X_test = test_data.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")/255

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras import optimizers

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (5, 5), input_shape = (32, 32, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

#  Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dropout(rate = 0.5))

classifier.add(Dense(units = 100, activation = 'softmax'))

# Compiling the CNN

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, fine_labels,
          epochs=100,
          batch_size= 32)




score = classifier.evaluate(X_test, test_fine_labels, batch_size = 32)

print(score)







