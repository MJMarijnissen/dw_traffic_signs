# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:22:10 2020

@author: Kubus
"""
#%%
import pandas as pd
import numpy as np
import os
import datetime

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

def get_cnn_v1(input_shape, num_classes):
    model = Sequential([
        Conv2D(filters = 64, kernel_size=(3,3), activation = 'relu', input_shape=input_shape),
        Flatten(),
        Dense(num_classes, activation = 'softmax'),
        ])
    return model

def train_model(model, X_train, y_train, params_fit={}):
    model.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics = ['accuracy'])
    
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq = 1)
    
    model.fit(
        X_train, 
        y_train,
        batch_size = params_fit.get('batch_size', 128),
        epochs = params_fit.get('epochs', 5),
        verbose = params_fit.get('verbose', 1),
        validation_data = params_fit.get('validation_data', (X_train, y_train)),
        callbacks = [tensorboard_callback]
    )
    return logdir

#%%
model = get_cnn_v1(input_shape, num_classes)
logdir = train_model(model, X_train, y_train)