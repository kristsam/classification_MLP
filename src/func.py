import numpy as np
import math

'''softplus'''
def softplus(a):
    h = np.log(1+np.exp(a))
    # h = 100 + math.log(math.exp(-100)+ math.exp(a-100))
    return h

def softplus_derivative(a):
    return np.exp(a)/(1+np.exp(a))

'''tanh'''
def tanh(a):
    return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))

def tanh_derivative(a):
    return 1 - tanh(a)**2

'''cos'''
def cos(a):
    return np.cos(a)

def cos_derivative(a):
    return -np.sin(a)

def mult_derivative(x,a):
    return a

def add_derivative(x,c):
    return 1

def softmax(x, ax=1):
    m = np.max( x, axis=ax, keepdims=True )#max per row
    p = np.exp( x - m )
    return ( p / np.sum(p,axis=ax,keepdims=True) )

def softmax_derivative(x):
    return softmax(x)(1-softmax(x))

def h_act_to_string(h):
    return h.__name__