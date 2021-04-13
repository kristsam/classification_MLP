import numpy as np
import math

def h1_act(a):
    h = math.log(1+math.exp(a))
    # h = 100 + math.log(math.exp(-100)+ math.exp(a-100))
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

def softmax(x, ax=1):
    m = np.max( x, axis=ax, keepdims=True )#max per row
    p = np.exp( x - m )
    return ( p / np.sum(p,axis=ax,keepdims=True) )