import math
import numpy as np
import sys
import matplotlib.pyplot as plt

class MultiLayerPerceptron:

    VALID_FUNCTIONS = ['h1','h2','h3']

    def __init__(self):
        self.M = []
        self.h = []
        self.learning_rate = []

    def __choose_func(self, activation):
        if activation == self.VALID_FUNCTIONS[0]:
            h = h1_act
        elif activation == self.VALID_FUNCTIONS[1]:
            h = h2_act
        elif activation== self.VALID_FUNCTIONS[2]: 
            h = h3_act
        else: 
            if activation not in self.VALID_FUNCTIONS:
                raise ValueError("results: activation function must be one of %r." % self.VALID_FUNCTIONS)
        return h

    def add(self, m, activation):
        '''
        Parameters
        ----------
        m (int):

        activation (String): {'h1', 'h2', 'h3'} There are 3 activation functions:
        1. log(1+exp(a))
        2. (exp(a)-exp(-a))/(exp(a)+exp(-a))
        3. cos(a)
        '''
        self.M.append(m)
        self.h.append(self.__choose_func(activation))

    def compile(self, learning_rate=[0.01], l=[0.01]):
        if isinstance(learning_rate, list):
              self.learning_rate = learning_rate
        elif isinstance(learning_rate, int):
              self.learning_rate = [learning_rate]
        if isinstance(l, list):
             self.lam = l
        elif isinstance(l, int):
             self.lam = [l]

    def __split(self, x, validation_split):
        split_rows = int(x.shape[0]* (1-validation_split))
        return x[:split_rows, :], x[split_rows:, :]

    def save(self, error_in_epochs_list, learning_rate, lam, error_best, epoch_best, learning_rate_best, lam_best, report):
        # TODO learning values and lam value legend
        fig = plt.figure()
        
        fig.suptitle("Multilayer Perceptron")

        for i in range(0, len(learning_rate)):
            for j in range(0,len(lam)):
                plt.subplot(len(learning_rate), len(lam), i*len(lam) + j+1)
                plt.plot(error_in_epochs_list[i, j, :, 0]+1,error_in_epochs_list[i, j, :, 1], label="training data")
                plt.plot(error_in_epochs_list[i, j, :, 0]+1,error_in_epochs_list[i, j, :, 2], label="validation data")
                plt.ylabel('error')
                plt.xlabel('epochs')
                plt.axis([1, error_in_epochs_list[i, j, -1, 0]+1, 0, 1])

        plt.legend()
        fig.savefig(report)
        return

    def predict(self, x, w1=None, w2=None, h=None, fixed=False):
        if not fixed:
            x_new = np.insert(x, 0, 1, axis=1)
            w1 = self.w1
            w2 = self.w2
            h = self.h
        else:
            x_new = np.copy(x)
        y = np.zeros(shape=(x_new.shape[0], self.K))
        z = np.zeros(shape=(self.M+1,1))
        for i in range(0,x_new.shape[0]):
            z[0][0] = 1
            for j in range(1,self.M+1):
                z[j][0] = h(np.dot(np.transpose(w1[j-1]), x_new[i]))
            for k in range(0,self.K):
                y[i][k] = softmax(k, w2, z)
        return y

    def score(self, y, t):
        count = 0
        for i in range(0,t.shape[0]):
            index_max = 0
            for j in range(0,t.shape[1]):
                if y[i][index_max] < y[i][j]:
                    index_max = j
            if t[i][index_max] == 1:
                count += 1
        return 1-count/t.shape[0]

    def keep_the_best_fit(self, M, N, Nb, learning_rate, lam, epochs, h, report=None):
        D = self.D
        K = self.K
        
        error_best = sys.maxsize
        epoch_best = -1
        learning_rate_best = -1
        lam_best = -1
        w1_best = 0.01*np.random.rand(M, D+1) + 0.001
        w2_best = 0.01*np.random.rand(K, M+1) + 0.001

        y_out = np.empty(shape=(Nb,K))
        z = np.empty(shape=(Nb, M+1))

        error_in_epochs_list = np.empty(shape=(len(learning_rate), len(lam), epochs, 3))
        for lr in range(0,len(learning_rate)):
            for l in range(0,len(lam)):
                w1 = 0.01*np.random.rand(M, D+1) + 0.001
                w2 = 0.01*np.random.rand(K, M+1) + 0.001
                for epoch in range(0,epochs):
                    for num1 in range(0, int(N/Nb)+1, Nb):
                        # each batch
                        for num2 in range(0, Nb):
                            n = num1 + num2
                            z[num2][0] = 1
                            for j in range(1,M+1):
                                mult = np.dot(np.transpose(w1[j-1]), self.x_train[n])
                                z[num2][j] = h(mult)

                            for k in range(0,self.K):
                                y_out[num2][k] = softmax(k, w2, z[num2])

                        cost, gradient_w1, gradient_w2 = cost_gradient(self.y_train[num1:num1+Nb],y_out,w1,w2,z,lam[l])
                        w2 = w2 - learning_rate[lr]*gradient_w2
                        w1 = w1 #- learning_rate[lr]*gradient_w1

                    # predict and return error
                    predictions_training = self.predict(self.x_train, w1, w2, h, fixed=True)
                    predictions_validation = self.predict(self.x_validation, w1, w2, h, fixed=True)
                    error_training = self.score(predictions_training, self.y_train)
                    error_validation = self.score(predictions_validation, self.y_validation)
                    error_in_epochs_list[lr][l][epoch][0] = epoch
                    error_in_epochs_list[lr][l][epoch][1] = error_training
                    error_in_epochs_list[lr][l][epoch][2] = error_validation

                    if error_best > error_validation:
                        error_best = error_validation
                        w1_best = np.copy(w1)
                        w2_best = np.copy(w2)
                        epoch_best = -1
                        learning_rate_best = learning_rate[lr]
                        lam_best = lam[l]

        if report is not None:
            self.save(error_in_epochs_list, learning_rate, lam, error_best, epoch_best, learning_rate_best, lam_best, report)
        return w1_best, w2_best, learning_rate_best, lam_best

    def fit(self, x, y, batch_size, epochs, validation_split=0.2, report=None):
        self.x_train, self.x_validation = self.__split(x, validation_split)
        self.y_train, self.y_validation = self.__split(y, validation_split)
        self.N = self.x_train.shape[0]
        self.D = self.x_train.shape[1]
        self.Nb = batch_size
        self.x_train = np.insert(self.x_train, 0, 1, axis=1)
        self.x_validation = np.insert(self.x_validation, 0, 1, axis=1)
        self.K = y.shape[1]
        # NOTE one inside layer
        self.M = self.M[0]
        self.h = self.h[0]
        self.w1, self.w2, self.learning_rate_chosen, self.lam_chosen = self.keep_the_best_fit(self.M, self.N, self.Nb, self.learning_rate, self.lam, epochs, self.h, report)

    def summary(self):
        print("N =",self.N," D =",self.D," M =",self.M," K =",self.K," Nb =",self.Nb," l =", self.lam)
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
    gradient_w2 = np.dot(np.transpose(t - y),z) - l*w2
    # TODO calc gradient for W1
    gradient_w1 = 0
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


# N training examples, D characteristics
# M weight vectors per characteristic
# K classes
# Nb minibatch 
# l learning rate, l>=0
