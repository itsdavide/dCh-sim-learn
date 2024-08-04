#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphs creation for the paper:

C. Marsala, D. Petturiti, and B. Vantaggi.
Adding Semantics to Fuzzy Similarity Measures Through the d-Choquet Integral.
In: Z. Bouraoui and S. Vesic (Eds.), Symbolic and Quantitative Approaches to 
Reasoning with Uncertainty. ECSQARU 2023, Volume 14294 of Lecture Notes 
in Computer Science, Springer, pp. 386–399, 2023.  
"""

import numpy as np
import matplotlib.pyplot as plt

ps = np.arange(0.5, 5, 0.5)
E_Iris = 0.9400782361308677 * 100
Cos_Iris = 0.7928520625889046 * 100

# pp Iris S1 1-additive
A_pp_1_1 = np.array([[0.68421053, 0.68421053, 0.64864865, 0.72972973],
       [0.72368421, 0.67105264, 0.67567568, 0.70945946],
       [0.71052632, 0.69078948, 0.70945946, 0.77027027],
       [0.6381579 , 0.70394737, 0.72972973, 0.73648649],
       [0.625     , 0.73684211, 0.72297297, 0.72297297],
       [0.61184211, 0.73026316, 0.70945946, 0.72972973],
       [0.625     , 0.71710527, 0.71621621, 0.73648649],
       [0.57236842, 0.73684211, 0.70945946, 0.77702702],
       [0.59868421, 0.73684211, 0.68243243, 0.74324324]])

# pp Iris S2 1-additive
A_pp_2_1 = np.array([[0.77631578, 0.78947368, 0.89189189, 0.86486486],
       [0.73684211, 0.76315789, 0.86486486, 0.85135135],
       [0.88815789, 0.88815789, 0.92567568, 0.89864865],
       [0.92763158, 0.86842105, 0.93243243, 0.82432432],
       [0.94736842, 0.875     , 0.92567568, 0.82432432],
       [0.94736842, 0.86842105, 0.91891892, 0.81756757],
       [0.94078947, 0.88157895, 0.91216216, 0.83108108],
       [0.91447368, 0.86842105, 0.87162162, 0.83108108],
       [0.92763158, 0.89473684, 0.87837838, 0.79054054]])

# pp Iris S3 1-additive
A_pp_3_1 = np.array([[0.86842105, 0.73026316, 0.91891892, 0.83108108],
       [0.96052632, 0.91447368, 0.93918919, 0.91891892],
       [0.98684211, 0.85526315, 0.9527027 , 0.93918919],
       [0.99342105, 0.84210526, 0.93243244, 0.93243244],
       [1.        , 0.81578947, 0.93243244, 0.91216216],
       [0.97368421, 0.84210526, 0.92567568, 0.94594595],
       [0.94736842, 0.86842105, 0.91891892, 0.91216216],
       [0.97368421, 0.84868421, 0.93243243, 0.94594595],
       [0.96710526, 0.8618421 , 0.91216216, 0.94594595]])

# pp Iris S1 2-additive
A_pp_1_2 = np.array([[0.68421053, 0.68421053, 0.64189189, 0.73648649],
       [0.68421053, 0.69078948, 0.68243243, 0.70945946],
       [0.69736843, 0.69736843, 0.68918919, 0.72297298],
       [0.67105264, 0.68421053, 0.68243244, 0.72972973],
       [0.65131579, 0.69078948, 0.69594595, 0.73648649],
       [0.65789474, 0.69736843, 0.66891892, 0.73648649],
       [0.65131579, 0.69736843, 0.66216216, 0.73648649],
       [0.65789474, 0.69736843, 0.64864865, 0.72972973],
       [0.68421053, 0.71052632, 0.66216216, 0.74324324]])

# pp Iris S2 2-additive
A_pp_2_2 = np.array([[0.77631578, 0.78947368, 0.89189189, 0.8445946 ],
       [0.75      , 0.76315789, 0.85135135, 0.86486486],
       [0.80921052, 0.81578947, 0.92567568, 0.85810811],
       [0.84210526, 0.85526316, 0.93243243, 0.81756757],
       [0.84210526, 0.8618421 , 0.89864865, 0.83783783],
       [0.88815789, 0.84210526, 0.93918919, 0.88513513],
       [0.86842105, 0.85526316, 0.91216216, 0.87162162],
       [0.90131579, 0.85526316, 0.91891892, 0.85135135],
       [0.90789473, 0.84868421, 0.93243243, 0.85810811]])

# pp Iris S3 2-additive
A_pp_3_2 = np.array([[0.82236842, 0.73026316, 0.91891892, 0.81081081],
       [0.98026316, 0.91447368, 0.93918919, 0.9527027 ],
       [0.96710526, 0.86842105, 0.93918919, 0.93918919],
       [0.97368421, 0.84210526, 0.94594595, 0.93918919],
       [0.95394737, 0.81578947, 0.9527027 , 0.93243243],
       [0.94736842, 0.84210526, 0.94594595, 0.91891892],
       [0.92105263, 0.86842105, 0.91891892, 0.93918919],
       [0.93421053, 0.86842105, 0.91891892, 0.93918919],
       [0.96052632, 0.875     , 0.91891892, 0.93918919]])

# Compute the mean accuracy
M_pp_1_1 = np.array([A_pp_1_1[i, :].mean() for i in range(len(ps))]) * 100
M_pp_2_1 = np.array([A_pp_2_1[i, :].mean() for i in range(len(ps))]) * 100
M_pp_3_1 = np.array([A_pp_3_1[i, :].mean() for i in range(len(ps))]) * 100
M_pp_1_2 = np.array([A_pp_1_2[i, :].mean() for i in range(len(ps))]) * 100
M_pp_2_2 = np.array([A_pp_2_2[i, :].mean() for i in range(len(ps))]) * 100
M_pp_3_2 = np.array([A_pp_3_2[i, :].mean() for i in range(len(ps))]) * 100

plt.figure(figsize=(6, 4))
plt.title(r'Iris dataset: PSO, $\delta_{p,p}$')
plt.plot(ps, M_pp_1_1, marker='o', color='green')
plt.plot(ps, M_pp_1_2, marker='o', color='green', linestyle='dashed')
plt.plot(ps, M_pp_2_1, marker='o', color='red')
plt.plot(ps, M_pp_2_2, marker='o', color='red', linestyle='dashed')
plt.plot(ps, M_pp_3_1, marker='o', color='blue')
plt.plot(ps, M_pp_3_2, marker='o', color='blue', linestyle='dashed')
plt.plot(ps, np.ones(len(ps)) * E_Iris, color='magenta', linestyle='dashed')
plt.plot(ps, np.ones(len(ps)) * Cos_Iris, color='darkorange', linestyle='dashed')
plt.xlabel('$p$')
plt.ylabel('Mean accuracy (%)')
plt.savefig('AVG-iris-pp-PSO.png', dpi=300)