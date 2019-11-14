# Proximal Gradient Method

import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Solution vector size and lambda
n = 100
p = 100
lambd = np.sqrt(2*n*np.log(p))

# Solution vector initialize
x = np.zeros(p)
A = np.zeros((n,p))
b = np.zeros(n)

for j in range(20):
    index = np.random.randint(0,p)
    x[index] = np.random.normal(0,10)

for i in range(len(A)):
    for j in range(len(A[0])):
        A[i,j] = np.random.normal(0,1)

b = np.dot(A, x)

# ObjectFunction
def ObjectFunction(X, A, b, lambd):
    return (1/2)*np.linalg.norm(np.dot(A, X)-b)**2 + lambd*np.linalg.norm(X, ord=1)

# SoftThresholdOpretor(To solve Prox(g(uk)))
def SoftThresholdOpretor(X, T):
    S = np.copy(X)
    for i in range(len(X)):
        if X[i] > T:
            S[i] = X[i] - T
        elif X[i] < -T:
            S[i] = X[i] + T
        else:
            S[i] = 0
    return S

# Iteration 
k=0
K=300

X = np.zeros(p)
PGD_Plot=[]

tau = 1/np.linalg.norm(np.dot(np.transpose(A), A))
# Stopping condition will be added soon
while k<K:
    # Objectvalue record
    PGD_Plot.append(ObjectFunction(X, A, b, lambd))
    X = SoftThresholdOpretor(X - tau*np.dot(np.transpose(A),(np.dot(A, X)-b)),tau*lambd)
    k=k+1

plt.plot(range(0,len(PGD_Plot)), PGD_Plot)