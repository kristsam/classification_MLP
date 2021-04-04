import math
import numpy as np

class MultiLayerPerceptron:
    def __init__(self):
        self.M = []
        self.h = []

    def add(self, m, activation):
        self.M.append(m)
        self.h.append(activation)

    def compile(self, learning_rate=0.01, l=0.01):
        self.hita = learning_rate
        self.l = 0.01

    def __split(self, x, validation_split):
        split_rows = int(x.shape[0]* (1-validation_split))
        return x[:split_rows, :], x[split_rows:, :]

    def fit(self, x, y, batch_size, validation_split=0.2):
        self.x_train, self.x_validation = self.__split(x, validation_split)
        self.y_train, self.y_validation = self.__split(y, validation_split)
        self.N = self.x_train.shape[0]
        self.D = self.x_train.shape[1]
        self.Nb = batch_size
        self.x_train = np.insert(self.x_train, 0, 1, axis=1)
        print(self.N, self.D)
        # one inside layer
        self.M = self.M[0]
        self.K = y.shape[1]

        y_out = np.empty(shape=(self.Nb,self.K))

        w1 = 0.01*np.random.rand(self.M, self.D+1) + 0.001

        z = np.empty(shape=(self.Nb, self.M+1))

        w2 = 0.01*np.random.rand(self.K, self.M+1) + 0.001

        for num1 in range(0, int(self.N/self.Nb)+1, self.Nb):
            # each batch
            for num2 in range(0, self.Nb):
                n = num1 + num2
                z[num2][0] = 1
                for j in range(1,self.M+1):
                    mult = np.dot(np.transpose(w1[j-1]), self.x_train[n])
                    z[num2][j] = self.h[0](mult)

                for k in range(0,self.K):
                    y_out[num2][k] = softmax(k, w2, z[num2])

            cost, gradient_w1, gradient_w2 = cost_gradient(self.y_train[num1:num1+self.Nb],y_out,w1,w2,z,self.l)
            w2 = w2 - self.hita*gradient_w2
            w1 = w1 - self.hita*gradient_w1

    # TODO predict

    def summary(self):
        print("N =",self.N," D =",self.D," M =",self.M," K =",self.K," Nb =",self.Nb," l =",self.l)
        print("x training set shape =", self.x_train.shape)
        print("y training set shape =", self.y_train.shape)




def cost_gradient(t, y, w1, w2, z, l):
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
