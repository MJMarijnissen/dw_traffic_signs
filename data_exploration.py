# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:27:43 2020

@author: Kubus
"""

import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_pickle('data/train.p')

X_train, y_train = train['features'], train['labels']

plt.imshow(X_train[10000])

signs = pd.read_csv('data/signnames.csv')

dict_signs = signs.to_dict()["b"]

for id_sign in dict_signs.keys():
    given_signs = X_train[y_train == id_sign]
    plt.figure(figsize=(15,5))
    for i in range(9):
        plt.subplot('19{0}'.format(i+1))
        plt.imshow(given_signs[i])
        plt.axis("off")
        
plt.tight_layout()
plt.show()