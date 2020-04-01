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

