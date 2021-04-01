import math
import numpy as np

def cost_func(t, y, w, l):
    K = y.shape[0]
    N = y.shape[1]
    output = 0
    for n in range(0,N+1):
        for k in range(0,K+1):
            #TODO cost 
            output += t[n][k] * math.log(y[n][k]) - l/2
    return output

def h1_act(a):
    return math.log(1+math.exp(a))

def h2_act(a):
    return (math.exp(a) - math.exp(-a)) / (math.exp(a) + math.exp(-a))

def h3_act(a):
    return math.cos(a)

def calc_ynk(w, zn):
    s = 0
    K = w.shape[1]
    for j in range(0, K+1):
        s += math.exp(w[:,j])
    return s


# N training examples, D characteristics
# M weight vectors per characteristic
# K classes

N = 1000
D = 50
M = 100
K = 10

x = np.ones(shape=(N,D+1))
x[:,0] = 1
y = np.ones(shape=(N,K))

y_out = np.ones(shape=(N,K))

w1 = np.ones(shape=(M, D+1))

z = np.ones(shape=(N, M+1))

w2 = np.ones(shape=(K, M+1))

for n in range(0,D+2):
    z[n][0] = 1
    for j in range(1,M+1):
        #TODO choose corect h1
        z[n][j] = h1_act(np.transpose(w1[j])*x[n])

    for k in range(0,K+1):
        y_out[n][k] = calc_ynk(w2, z[n])  
