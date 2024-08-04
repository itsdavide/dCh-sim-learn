# dCh-sim-learn
Similarity learning code for the paper:
    
C. Marsala, D. Petturiti, and B. Vantaggi.
_Adding Semantics to Fuzzy Similarity Measures Through the d-Choquet Integral_.
In: Z. Bouraoui and S. Vesic (Eds.), **Symbolic and Quantitative Approaches to 
Reasoning with Uncertainty. ECSQARU 2023**, Volume 14294 of **Lecture Notes 
in Computer Science**, Springer, pp. 386–399, 2023.


# Requirements
The learning task is carried out through the Particle Swarm Optimization (PSO) technique.
The code requires the PySwarms library available at:[https://pypi.org/project/pyswarm/](https://pypi.org/project/pyswarms/).

# Datasets
The file fuzzy-iris.csv is a fuzzified version of the classical Iris dataset.

# File inventory
**dCh-sim-learn-4-fs.py**: Performs the similarity learning through PSO by optimizing the Leave One Out (LOO) objective function, for a choice of restricted dissimilarity function in {pp, p1, 1p} and similarity measure in {S1, S2, S3}. The file must be exectude at the command line by writing

_dCh-sim-learn-4-fs.py [-h] -f FILENAME [-k ADDITIVITY] [-s SIMILARITY] [-d DISSIMILARITY] [-i ITERATIONS]_


