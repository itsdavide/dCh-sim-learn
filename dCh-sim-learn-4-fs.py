# -*- coding: utf-8 -*-

""" 
Fuzzy similarity learning code for the paper:
    
C. Marsala, D. Petturiti, and B. Vantaggi.
Adding Semantics to Fuzzy Similarity Measures Through the d-Choquet Integral.
In: Z. Bouraoui and S. Vesic (Eds.), Symbolic and Quantitative Approaches to 
Reasoning with Uncertainty. ECSQARU 2023, Volume 14294 of Lecture Notes 
in Computer Science, Springer, pp. 386–399, 2023.    
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from pyswarms.single import GlobalBestPSO

import time

# ----------------------------------
# To have a command line execution
import argparse

parserArgs = argparse.ArgumentParser()

parserArgs.add_argument("-f", "--filename", type=str, help="dataset to use", required=True)
parserArgs.add_argument("-k", "--additivity", type=int, help="value of k for additivity", default="1")
parserArgs.add_argument("-s", "--similarity", type=int, help="chosen similarity (1, 2 or 3)", default="1")
parserArgs.add_argument("-d", "--dissimilarity", type=str, help="chose dissimilarity (pp, 1p, p1)", default="p1")
parserArgs.add_argument("-i", "--iterations", type=int, help="number of iterations (pyswarms)", default="20")


args= parserArgs.parse_args()

dataset = args.filename           # dataset to use (e.g. datasets/fuzzy-iris.csv)
k_add = args.additivity           # value of k for k-additivity
similarity_num = args.similarity  # similarity 1 (for S1), 2 (for S2), 3 (for S3)
case_delta = args.dissimilarity   # dissimilairty: pp, p1, 1p
nb_iterations = args.iterations   # number of iterations for pyswarms


# -----------------------------------------------
# CONFIGURATION :
# Restricted dissimilarity functions

def delta_pp(x, y, p):
    return (np.abs(x**p - y**p))**(1 / p)

def delta_p1(x, y, p):
    return (np.abs(x**p - y**p))

def delta_1p(x, y, p):
    return (np.abs(x - y))**(1 / p)


# Select the restricted dissimilarity function
delta = None
if case_delta == 'pp':
    delta = delta_pp
if case_delta == 'p1':
    delta = delta_p1
if case_delta == '1p':
    delta = delta_1p
    
# -------------------------

# Training set
data = pd.read_csv(dataset)

# ASSUMPTION: The fuzzy dataset is assumed to have N columns indexed as 
# '0', '1', ..., 'N-1', where the columns '0', '1', ..., 'N-2' form the feature
# vector X, while the columns 'N-1' is the class variable

# Divide X variables from y variable
N = data.shape[1]
X = data[[str(i) for i in range(N-1)]]
y = data[str(N-1)]

# Case 1 additive: Array of sets of indices (1-additive case)
nb_dim = X.shape[1]  # nbr of variables (description)

# -----------------------------------------------
print("**********\nInformations:")
mess_info = "case "+case_delta+": "
mess_info += str(k_add)+"-additive"
mess_info += "\tsimilarity: S"+str(similarity_num)
mess_info += "\tdata: "+dataset
print(mess_info)
print("Nb of iterations: ",nb_iterations)
print("**********\n")

# -----------------------------------------------
# Building all the sets
def build_sets(nb_cols,k=2):
    """ build a set
        nb_cols (int): number of variables
        output: np.array(set[int]) : array of sets of indices (2-additive case)
        k: additivity: k=1 for 1-additive, etc.
        rem: start counting from 0
    """
    d_sets = []
    for i in range(0,nb_cols):
        d_sets.append({i})
    if k>1:
        for i in range(0,nb_cols):
            for j in range(i+1,nb_cols):
                d_sets.append({i,j})
    return np.array(d_sets)

# Case 1 additive:
# Array of sets of indices (1-additive case)
sets = build_sets(nb_dim, k=k_add)
########################################################

# -----------------------------------------------
def d_Choquet(X, sets, m, p):
    # New version : debugged 2023-05-28
    # Sort the array X and find the permutation sigma
    sigma = np.argsort(X)  # Sort increasing and give indices (don't modify X)
    #print('sigma2:', sigma)
    # Compute the Choquet integral
    integral = 0
    for i in range(len(X)):
        # Compute the nu(E_i)
        E_i = {sigma[j] for j in range(i,len(X))}
        nu_i = np.zeros(m.shape[0])
        for k in range(len(sets)):
            current_set = sets[k]
            if current_set.issubset(E_i):
                nu_i += m[:, k]
        x_i = X[sigma[i]]
        x_i_1 = X[sigma[i-1]] if i != 0 else 0
        integral += delta(x_i, x_i_1, p) * nu_i
    return integral

# -----------------------------------------------
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

# This is the function to optimize through PSO:
# the optimization variable is the Mobius inverse m,
# while the others are fixed parameters
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

# Defines a new function to have a minimization problem
def g_loo(m, sets, X, y, p, S):
    return -f_loo(m, sets, X, y, p, S)


# -----------------------------------------------------
# Initialization

# Initialization for k_add = 1:
m_init = np.array([np.maximum((np.ones(4)/4) + ((-1)**i) * np.random.uniform(0, 0.25, 4), 0) for i in range(20)])
m_init[0] = (np.ones(4)/4)

# Initialization for k_add = 2:
if k_add == 2:
    m_init = np.array([np.append(np.maximum((np.ones(4)/4) + ((-1)**i) * np.random.uniform(0, 0.25, 4), 0),  np.zeros(6)) for i in range(20)])
    m_init[0] = np.append((np.ones(4)/4), np.zeros(6))

# Instantiate the optimizer
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# We can add bounds to make the game monotone and additive
m_max = np.ones(len(sets))
m_min = np.zeros(len(sets))
bounds = (m_min, m_max)

similarity = None
if similarity_num == 1:
    similarity = S1
if similarity_num == 2:
    similarity = S2
if similarity_num == 3:
    similarity = S3
# -----------------------------------------------------

# Stratified 4-fold cross validation
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

ps = np.arange(0.5, 5, 0.5)  #: ORIGINAL

acc_p = []
tot_p = []
ms_opt_p = []

tic = time.time()
for p in ps:
    cnt = 1
    acc = []
    tot = []
    ms_opt = []
    k_cv = 0 # to count cross-validation
    print("***** Case p =",p)
    for train_index, test_index in kf.split(X, y):
        k_cv += 1

        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        # Optimization with PSO
        tic2 = time.time()
        optimizer = GlobalBestPSO(n_particles=20, dimensions=len(sets), options=options, bounds=bounds, init_pos=m_init)
        opt, m_opt = optimizer.optimize(g_loo, iters=nb_iterations, sets=sets, X=X_train, y=y_train, p=p, S=similarity)

        toc2 = time.time()
        print("\t---> Running: ", toc2-tic2, "seconds for k_cv =", k_cv)
        # Test the m_opt
        acc.append(f_loo(np.array([m_opt]), sets=sets, X=X_test, y=y_test, p=p, S=similarity)[0])
        tot.append(len(X_test))
        ms_opt.append(m_opt)
        cnt += 1
    acc_p.append(acc)
    tot_p.append(tot)
    ms_opt_p.append(ms_opt)

toc = time.time()
print("***** Ending: ",toc-tic,"seconds")

A = np.array(acc_p)
T = np.array(tot_p)
Ms = np.array(ms_opt_p)

print('Accuracy: ')
print(A / T)
for i in range(len(ps)):
    print('p = ', ps[i])
    print('Mean accuracy: ', (A / T)[i, :].mean())
    print('Std accuracy: ', (A / T)[i, :].std())

# -----------------------------------------------------
