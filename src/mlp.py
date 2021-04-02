import math
import numpy as np
import read as r

def cost_gradient(t, y, w1, w2, l):
    K = y.shape[0]
    N = y.shape[1]
    cost1 = 0
    for k in range(0,y.shape[1]):
        for n in range(0,y.shape[0]):
            cost1 += t[n][k]*np.log(y[n][k]) - l/2*np.linalg.norm(w2)**2
    cost = np.sum(np.dot(np.transpose(t),np.log(y)) - l/2*np.linalg.norm(w2)**2)
    gradient_w1 = np.dot(np.transpose(t - y),z) - l*w2
    # TODO calc gradient for W2
    gradient_w2 = 0
    return cost, gradient_w1, gradient_w2

def h1_act(a):
    h = math.log(1+math.exp(a))
    h = 100 + math.log(math.exp(-100)+ math.exp(a-100))
    return h

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
    for j in range(0, K):
        s += np.exp(np.dot(np.transpose(w[j]),zn))
    numerator = np.exp(np.dot(np.transpose(w[k]),zn))
    if s!=0:
        return numerator/s
    return 0

def choose_func():
    ch = 0 
    while ch<1 or ch>3:
        print("Which one activation function do you want to choose?")
        print("1. log(1+exp(a))")
        print("2. (exp(a)-exp(-a))/(exp(a)+exp(-a))")
        print("3. cos(a)")
        ch = int(input("Give the apriopriate number: "))
    if ch == 1:
        h = h1_act
    elif ch == 2:
        h = h2_act
    else:
        h = h3_act
    return h


# N training examples, D characteristics
# M weight vectors per characteristic
# K classes
# Nb minibatch 
# l learning rate, l>=0

h = choose_func()

import os 
path_to_data = os.path.dirname(os.path.realpath(__file__))+'\\..\\data\\mnist\\'
x_train, y_train, x_validation, y_validation, x_test, y_test = r.read(path_to_data)

N = x_train.shape[0]
D = x_train.shape[1]
print(N, D)
M = 100
K = 10
Nb = 100
l = 0.01

print("N =",N," D =",D," M =",M," K =",K," Nb =",Nb," l =",l)

x_train = np.insert(x_train, 0, 1, axis=1)
print("x training set shape =", x_train.shape)
print("y training set shape =", y_train.shape)

y = np.empty(shape=(Nb,K))

w1 = 0.01*np.random.rand(M, D+1) + 0.001

z = np.empty(shape=(Nb, M+1))

w2 = 0.01*np.random.rand(K, M+1) + 0.001

for num1 in range(0, int(N/Nb)+1, Nb):
    # each batch
    for num2 in range(0, Nb):
        n = num1 + num2
        z[num2][0] = 1
        for j in range(1,M+1):
            mult = np.dot(np.transpose(w1[j-1]), x_train[n])
            z[num2][j] = h(mult)

        for k in range(0,K):
            y[num2][k] = softmax(k, w2, z[num2])

    cost, gradient_w1, gradient_w2 = cost_gradient(y_train[num1:num1+Nb],y,w1,w2,l)
    w2 = w2 - l*gradient_w2
    w1 = w1 - l*gradient_w1

