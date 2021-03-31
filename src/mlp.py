import math
import numpy as np

def cost_func(t, y, w, l){
    K = y.shape[0]
    N = y.shape[1]
    output = 0
    for n in range(0,N+1):
        for k in range(0,K+1):
            output += t[n][k] * math.log(y[n][k]) - l/2*
    return output
}

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
        sum += math.exp(w[:,j])

x = np.ones(shape=(100,40))
y = np.ones(shape=(100,1))

w1 = np.ones(shape=())