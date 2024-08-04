#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Similarity measures parameterized by the uniform probability measure code for the paper:
    
C. Marsala, D. Petturiti, and B. Vantaggi.
Adding Semantics to Fuzzy Similarity Measures Through the d-Choquet Integral.
In: Z. Bouraoui and S. Vesic (Eds.), Symbolic and Quantitative Approaches to 
Reasoning with Uncertainty. ECSQARU 2023, Volume 14294 of Lecture Notes 
in Computer Science, Springer, pp. 386–399, 2023.    
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Generic restricted dissimilarity function
delta = None

# Generic similarity measure
S = None


# Restricted dissimilarity functions
def delta_pp(x, y, p):
    return (np.abs(x**p - y**p))**(1 / p)

def delta_p1(x, y, p):
    return (np.abs(x**p - y**p))

def delta_1p(x, y, p):
    return (np.abs(x - y))**(1 / p)

# Array of sets of indices (1-additive case)
sets = np.array([{0}, {1}, {2}, {3}])

# Compute the d-Choquet integral
def d_Choquet(X, sets, m, p):
    # Sort the array X and find the permutation sigma
    sorted_X = np.sort(X)
    sigma = np.ones(len(X)) * (-1)
    for i in range(len(sorted_X)):
        indices = np.where(X == sorted_X[i])[0]
        for j in indices:
            if not np.any(sigma == j):
                sigma[i] = j
                break
    
    # Compute the Choquet integral
    integral = 0
    for i in range(len(X)):
        # Compute the nu(E_i)
        E_i = set()
        for j in range(i, len(X)):
            E_i.add(int(sigma[j]))
        nu_i = np.zeros(m.shape[0])
        for k in range(len(sets)):
            current_set = sets[k]
            if current_set.issubset(E_i):
                nu_i += m[:, k]
        x_i = X[int(sigma[i])]
        x_i_1 = X[int(sigma[i - 1])] if i != 0 else 0
        integral += delta(x_i, x_i_1, p) * nu_i
    
    return integral

# Fuzzy set-theoretic operations
def Inter(X, Y):
    return np.minimum(X, Y)

def Union(X, Y):
    return np.maximum(X, Y)

def Diff(X, Y):
    return np.minimum(X, 1 - Y)

def Delta(X, Y):
    return np.maximum(Diff(X, Y), Diff(Y, X))

# Fuzzy similarity measures
def S1(X, Y, sets, m, p):
    return d_Choquet(Inter(X, Y), sets, m, p) / (d_Choquet(Diff(X, Y), sets, m, p) +
                                                 d_Choquet(Diff(Y, X), sets, m, p) +
                                                 d_Choquet(Inter(X, Y), sets, m, p))
def S2(X, Y, sets, m, p):
    return d_Choquet(Inter(X, Y), sets, m, p) / (d_Choquet(Delta(X, Y), sets, m, p) +
                                                 d_Choquet(Inter(X, Y), sets, m, p))
def S3(X, Y, sets, m, p):
    return d_Choquet(Inter(X, Y), sets, m, p) / d_Choquet(Union(X, Y), sets, m, p)

# Fuzzy version of Iris dataset
data = pd.read_csv('datasets/fuzzy-iris.csv')
data

# Divide X variables from y variable
from sklearn.model_selection import train_test_split
X = data[['0', '1', '2', '3']]
y = data['4']

# Uniform 1-addtive Mobius
m_u = np.ones(4) / 4

# LOO objective function
def f_loo(m, sets, X, y, p, S):
    indices = X.index.values
    tot = np.zeros(m.shape[0])
    for i in indices:
        max_val = np.ones(m.shape[0]) * (-np.Infinity)
        max_index = np.ones(m.shape[0]) * (-1) 
        for j in indices:
            if i != j:
                # Extract the fuzzy sets
                Z_i = np.array(X.loc[i])
                Z_j = np.array(X.loc[j])
                sim = S(Z_i, Z_j, sets, m, p)
                max_val = np.maximum(max_val, sim)
                max_index = ((sim >= max_val) * j + (sim < max_val) * max_index).astype(int)
        test = (y.loc[i] == y.loc[max_index]).to_numpy()
        tot += test # Boolean values are automatically converted in binary
    return tot

deltas = [delta_pp, delta_p1, delta_1p]
Ss = [S1, S2, S3]


for delta in deltas:
    print('\n*** Computing with delta:', delta)
    for S in Ss:
        print('*** Computing with S:', S)
        
        # 4-fold cross validation
        kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        
        ps = np.arange(0.5, 5, 0.5)
        acc_p = []
        tot_p = []
        
        for p in ps:
            cnt = 1
            acc = []
            tot = []
            ms_opt = []
            for train_index, test_index in kf.split(X, y):
                X_train = X.iloc[train_index]
                y_train = y.iloc[train_index]
                X_test = X.iloc[test_index]
                y_test = y.iloc[test_index]
            
                # Test the m_opt
                acc.append(f_loo(np.array([m_u]), sets=sets, X=X_test, y=y_test, p=p, S=S)[0])
                tot.append(len(X_test))
                cnt += 1
            acc_p.append(acc)
            tot_p.append(tot)
        
        A = np.array(acc_p)
        T = np.array(tot_p)
        
        print('Accuracy: ')
        print(A / T)
        for i in range(len(ps)):
            print('p = ', ps[i])
            print('Mean accuracy: ', (A / T)[i, :].mean())
            print('Std accuracy: ', (A / T)[i, :].std())