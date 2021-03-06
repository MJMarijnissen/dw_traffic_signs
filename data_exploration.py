# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:27:43 2020

@author: Kubus
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

#%%

def read_data():
    """
    function reads data, returns 

    Returns
    -------
    X_train : numpy.ndarray
        train set input.
    y_train : numpy.ndarray
        train set outpu.
    X_test : numpy.ndarray
        test set input.
    y_test : numpy.ndarray
        test set output.
    dict_signs : DICT
        description of signs.

    """
    train = pd.read_pickle('data/train.p')
    test = pd.read_pickle('data/test.p')
    
    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], train['labels']
    signs = pd.read_csv('data/signnames.csv')
    dict_signs = signs.to_dict()["b"]
    return X_train, y_train, X_test, y_test, dict_signs

X_train, y_train, X_test, y_test, dict_signs = read_data()

#%%
def plot_all_signs():
    """
    Plots all signs in dataset

    Returns
    -------
    None.

    """
    plt.imshow(X_train[10000])
    
    for id_sign in dict_signs.keys():
        given_signs = X_train[y_train == id_sign]
        plt.figure(figsize=(15,5))
        for i in range(9):
            plt.subplot('19{0}'.format(i+1))
            plt.imshow(given_signs[i])
            plt.axis("off")
            
    plt.tight_layout()
    plt.show()
    pass
#%%%

def plot_labels_count():
    """
    plots bar chart of the ammout each sign is in dataset

    Returns
    -------
    None.

    """
    cnt = Counter(y_train).most_common()
    id_labels, cnt_labels = zip(*cnt)
    ids = range(len(id_labels))
    
    plt.figure(figsize=(15,5))
    plt.bar(ids, cnt_labels)
    plt.xlabel('Znaki')
    labels = [dict_signs[id_labels[idki]] for idki in id_labels]
    plt.xticks(ids, labels, rotation = 'vertical')
    plt.title('Znaki drogowe - liczba wystąpień')
    plt.show()
    pass

if __name__ == "__main__":
    read_data()
    plot_all_signs()
    plot_labels_count()
