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

def h1_act_derivative(a):
    return math.exp(a)/(1+math.exp(a))

def h2_act(a):
    return (math.exp(a) - math.exp(-a)) / (math.exp(a) + math.exp(-a))

def h2_act_derivative(a):
    return 1 - h2_act(a)**2

def h3_act(a):
    return math.cos(a)

def h3_act_derivative(a):
    return - math.cos(a)

def mult_derivative(x,a):
    return a

def add_derivative(x,c):
    return 1

def softmax(k, w, zn):
    s = 0
    K = w.shape[0]
    for j in range(0, K+1):
        s += math.exp(np.transpose(w[j])*zn)
    numerator = math.exp(np.transpose(w[k])*zn)
    if s!=0:
        return numerator/s
    return 0


# N training examples, D characteristics
# M weight vectors per characteristic
# K classes
# Nb minibatch 
# l learning rate, l>=0

N = 1000
D = 50
M = 100
K = 10
Nb = 100
l = 0.01

x = np.ones(shape=(N,D+1))
x[:,0] = 1
y = np.ones(shape=(N,K))

t = np.ones(shape=(Nb,K))

w1 = np.ones(shape=(M, D+1))

z = np.ones(shape=(Nb, M+1))

w2 = np.ones(shape=(K, M+1))

for num1 in range(0,N/Nb+1):
    # each batch
    for num2 in range(0, Nb+1):
        n = num1 * num2 + num2
        z[num2][0] = 1
        for j in range(1,M+1):
            # TODO choose correct h1
            z[num2][j] = h1_act(np.transpose(w1[j])*x[n])

        for k in range(0,K+1):
            t[num2][k] = softmax(k, w2, z[num2])
    w2 = np.transpose(t - y)*z - l*w2
    
