# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:27:43 2020

@author: Kubus
"""

import pandas as pd

train = pd.read_pickle('data/train.p')

X_train, y_train = train['features'], train['labels']

pd.read_csv('data/signnames.csv').sample(10)