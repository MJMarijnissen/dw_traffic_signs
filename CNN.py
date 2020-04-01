# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:22:10 2020

@author: Kubus
"""
#%%
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from data_exploration import read_data

#%%
X_train, y_train, X_test, y_test, dict_signs = read_data()

if y_train.ndim == 1: y_train = to_categorical(y_train)
if y_test.ndim == 1: y_test = to_categorical(y_test)

#%%
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]

model = Sequential([
    Conv2D(filters = 64, kernel_size=(3,3), activation = 'relu', input_shape=input_shape),
    Flatten(),
    Dense(num_classes, activation = 'softmax'),
    ])

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics = ['accuracy'])
model.fit(X_train, y_train)

#%%