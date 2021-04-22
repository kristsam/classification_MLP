import func as fc
import numpy as np
import math 

class HiddenLayer:

    VALID_FUNCTIONS = ['softplus','tanh','cos']

    def __choose_func(self, activation):
        if activation in HiddenLayer.VALID_FUNCTIONS:
            h = eval('fc.'+activation )
            h_der = eval('fc.'+activation+"_derivative")
        else: 
            if activation not in self.VALID_FUNCTIONS:
                raise ValueError("results: activation function must be one of %r." % self.VALID_FUNCTIONS)
        return h, h_der
    
    def __init__(self, units, activation, add_column=False):
        '''        
        activation (String): {'softplus', 'tanh', 'cos'} There are 3 activation functions:
        1. log(1+exp(a))
        2. (exp(a)-exp(-a))/(exp(a)+exp(-a))
        3. cos(a)
        '''
        self.units = units
        self.activation, self.activation_derivative =  self.__choose_func(activation)
        self.add_column = add_column

    def adjust(self, old_layer_units, initializer='glorot'):
        init = choose_initializer(initializer)
        self.w = init(self.units,old_layer_units)

    
    def forward_propagation(self, x):
        self.input = x
        o = self.activation(np.dot(self.input,self.w.T))
        if self.add_column:
            o = np.insert(o, 0, 1, axis=1)
        return o
    
    def back_propagation(self, x):
        index = 0 
        if self.add_column:
            index = 1
        return np.dot(np.transpose(np.multiply(x[:,index:], self.activation_derivative(np.dot(self.input,self.w.T)))), self.input)

    def activation_to_string(self):
        return fc.h_act_to_string(self.activation)

    def crop(self, x):
        if self.add_column:
            return x[:,1:]

    def layer_units(self):
        if self.add_column:
            return self.units + 1
        else:
            return self.units


class OutputLayer:

    def __init__(self, output_features, activation='softmax'):
        self.output_features = output_features
        self.activation = fc.softmax
        self.activation_derivative = fc.softmax_derivative

    def adjust(self, old_layer_units, initializer='glorot'):
        init = choose_initializer(initializer)
        self.w = init(self.output_features,old_layer_units)

    def forward_propagation(self, x):
        self.input = x
        y = self.activation(np.dot(self.input,self.w.T))
        return y

    def back_propagation(self,x):
        return np.dot(x.T, self.input)


def glorot(fin, fout):
    x = math.sqrt(6/(fin+fout))
    w = np.empty(shape=(fin,fout))
    for i in range(0,w.shape[0]):
        for j in range(0,w.shape[1]):
            w[i][j] = np.random.uniform(-x,x)
    return w

def choose_initializer(initializer):
    ini = glorot
    if initializer=='glorot':
        ini = glorot
    return ini
