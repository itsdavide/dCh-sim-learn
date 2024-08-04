# dCh-sim-learn
Fuzzy similarity learning code for the paper:
    
C. Marsala, D. Petturiti, and B. Vantaggi.
_Adding Semantics to Fuzzy Similarity Measures Through the d-Choquet Integral_.
In: Z. Bouraoui and S. Vesic (Eds.), **Symbolic and Quantitative Approaches to 
Reasoning with Uncertainty. ECSQARU 2023**, Volume 14294 of **Lecture Notes 
in Computer Science**, Springer, pp. 386–399, 2023.


# Requirements
The learning task is carried out through the Particle Swarm Optimization (PSO) technique.
The code requires the PySwarms library available at: https://pypi.org/project/pyswarm/.

# Datasets
The file **fuzzy-iris.csv** is a fuzzified version of the classical Iris dataset.

# File inventory
**dCh-sim-learn-4-fs.py**: Performs the similarity learning with a stratified 4-fold cross validation through PSO for different values of _p_, by optimizing the Leave One Out (LOO) objective function, for a fixed choice of restricted dissimilarity function in _{pp, p1, 1p}_ and similarity measure in _{S1, S2, S3}_. The code must be exectude at the command line by writing

_$ python3 dCh-sim-learn-4-fs.py [-h] -f FILENAME [-k ADDITIVITY] [-s SIMILARITY] [-d DISSIMILARITY] [-i ITERATIONS]_

**iris-4-fs-euc-cos-sim.py**: Computes the Leave One Out (LOO) objective function with a stratified 4-fold cross validation for different values of _p_, using the Euclidean and the Cosine similarity measure on the **fuzzy-iris.csv** dataset.

**iris-4-fs-unif.py**: Computes the Leave One Out (LOO) objective function with a stratified 4-fold cross validation for different values of _p_, using each similarity measure in _{S1, S2, S3}_ with the uniform probability measure on the **fuzzy-iris.csv** dataset.

**figures-AVG-pp.py**, **figures-AVG-p1.py**, **figures-AVG-1p.py**, **figures-unif.py**: Plot the graphs of mean accuracy in the 4 folds as a function of _p_.


