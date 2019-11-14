# Accelerated Proximal Gradient Method

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

X_last= np.zeros(p)
X = np.zeros(p)
APGD_Plot=[]

tau = 1/np.linalg.norm(np.dot(np.transpose(A), A))

t_last = 1
t = 1
error=1000
APGD_Error=[]
# Stopping condition will be added soon
while k < K - 1 and error > 1e-4:
    # Objectvalue record
    APGD_Plot.append(ObjectFunction(X, A, b, lambd))
    v = X + ((t_last -1)/t)*(X-X_last)
    X_new = SoftThresholdOpretor(v - tau*np.dot(np.transpose(A),(np.dot(A, v)-b)),tau*lambd)
    X_last = X
    X = X_new

    t_new = (1+np.sqrt(1+4*t**2))/2
    t_last = t
    t = t_new
    error =np.linalg.norm((1/tau)*(X_last-X)-(np.dot(np.transpose(A),(np.dot(A, X_last)-b))-np.dot(np.transpose(A),(np.dot(A, X)-b))))
    APGD_Error.append(error)
    k = k + 1

plt.plot(range(0,len(APGD_Plot)), APGD_Plot)
plt.figure()
plt.plot(range(0,len(APGD_Error)), APGD_Error)
print(APGD_Error[len(APGD_Error)-1])