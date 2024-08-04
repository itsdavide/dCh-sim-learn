#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Euclidean and cosine similarity code for the paper:
    
C. Marsala, D. Petturiti, and B. Vantaggi.
Adding Semantics to Fuzzy Similarity Measures Through the d-Choquet Integral.
In: Z. Bouraoui and S. Vesic (Eds.), Symbolic and Quantitative Approaches to 
Reasoning with Uncertainty. ECSQARU 2023, Volume 14294 of Lecture Notes 
in Computer Science, Springer, pp. 386–399, 2023.    
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# Fuzzy version of Iris dataset
data = pd.read_csv('datasets/fuzzy-iris.csv')


# ASSUMPTION: The fuzzy dataset is assumed to have N columns indexed as 
# '0', '1', ..., 'N-1', where the columns '0', '1', ..., 'N-2' form the feature
# vector X, while the columns 'N-1' is the class variable

# Divide X variables from y variable
X = data[['0', '1', '2', '3']]
y = data['4']

# LOO objective function using the Euclidean similaritty measure
def f_LOO_E(X, y):
    indices = X.index.values
    tot = 0
    for i in indices:
        max_val = -np.Infinity
        max_index = -1 
        for j in indices:
            if i != j:
                # Extract the fuzzy sets
                Z_i = np.array(X.loc[i])
                Z_j = np.array(X.loc[j])
                sim = 1 - ((Z_i - Z_j)**2).mean()
                max_val = np.maximum(max_val, sim)
                max_index = ((sim >= max_val) * j + (sim < max_val) * max_index).astype(int)
        test = (y.loc[i] == y.loc[max_index])
        tot += test # Boolean values are automatically converted in binary
    return tot

# LOO objective function using the Cosine similaritty measure
def f_LOO_Cos(X, y):
    indices = X.index.values
    tot = 0
    for i in indices:
        max_val = -np.Infinity
        max_index = -1 
        for j in indices:
            if i != j:
                # Extract the fuzzy sets
                Z_i = np.array(X.loc[i])
                Z_j = np.array(X.loc[j])
                sim = (Z_i * Z_j).sum() / (np.sqrt((Z_i**2).sum()) * np.sqrt((Z_j**2).sum()))
                max_val = np.maximum(max_val, sim)
                max_index = ((sim >= max_val) * j + (sim < max_val) * max_index).astype(int)
        test = (y.loc[i] == y.loc[max_index])
        tot += test # Boolean values are automatically converted in binary
    return tot

# Stratified 4-fold cross validation
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# Compute the LOO objective with the Euclidean similarity measure in the 4 folds
cnt = 1
acc = []
tot = []
for train_index, test_index in kf.split(X, y):
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]
    
    # Euclidean
    acc.append(f_LOO_E(X=X_test, y=y_test))
    tot.append(len(X_test))
    cnt += 1
    
# Print the mean accuracy obtained with the Euclidean Similarity measure
acc_E = (np.array(acc) / np.array(tot)).mean()
print('Euclidean similarity mean accuracy in the 4 folds:', acc_E)

# Compute the LOO objective with the Cosine similarity measure in the 4 folds
cnt = 1
acc = []
tot = []
for train_index, test_index in kf.split(X, y):
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]
    
    # Euclidean
    acc.append(f_LOO_Cos(X=X_test, y=y_test))
    tot.append(len(X_test))
    cnt += 1
    
# Print the mean accuracy obtained with the Cosine Similarity measure
acc_Cos = (np.array(acc) / np.array(tot)).mean()
print('Cosine similarity mean accuracy in the 4 folds:', acc_Cos)