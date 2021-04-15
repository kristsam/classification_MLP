import numpy as np
import math

'''softplus'''
def h1_act(a):
    h = np.log(1+np.exp(a))
    # h = 100 + math.log(math.exp(-100)+ math.exp(a-100))
    return h

def h1_act_derivative(a):
    return np.exp(a)/(1+np.exp(a))

'''tanh'''
def h2_act(a):
    return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))

def h2_act_derivative(a):
    return 1 - h2_act(a)**2

'''cos'''
def h3_act(a):
    return np.cos(a)

def h3_act_derivative(a):
    return -np.sin(a)

def mult_derivative(x,a):
    return a

def add_derivative(x,c):
    return 1

def softmax(x, ax=1):
    m = np.max( x, axis=ax, keepdims=True )#max per row
    p = np.exp( x - m )
    return ( p / np.sum(p,axis=ax,keepdims=True) )

def h_act_to_string(h):
    t = ''
    if h == h1_act:
        t = 'softplus'
    elif h == h2_act:
        t = 'tanh'
    elif h == h3_act:
        t = 'cos'
    return t