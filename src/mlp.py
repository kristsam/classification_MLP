import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import func as fc

class MultiLayerPerceptron:

    VALID_FUNCTIONS = ['h1','h2','h3']

    def __init__(self):
        self.M = []
        self.h = []
        self.learning_rate = []

    def __choose_func(self, activation):
        if activation == self.VALID_FUNCTIONS[0]:
            h = fc.h1_act
        elif activation == self.VALID_FUNCTIONS[1]:
            h = fc.h2_act
        elif activation== self.VALID_FUNCTIONS[2]: 
            h = fc.h3_act
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

    def compile(self, learning_rate=[0.01], lam=[0.01], initializer='glorot'):
        if isinstance(learning_rate, list):
              self.learning_rate = learning_rate
        elif isinstance(learning_rate, int):
              self.learning_rate = [learning_rate]
        if isinstance(lam, list):
             self.lam = lam
        elif isinstance(lam, int):
             self.lam = [lam]
        self.initializer = choose_initializer(initializer)

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
        z = np.zeros(shape=(x_new.shape[0], self.M+1))
        z[:,0] = 1
        z[:,1:] = h(np.dot(x, w1.T))
        y = fc.softmax(np.dot(z,w2.T), ax=1)
        return y

    def score(self, y, t):
        # index of max values array
        index_y = np.argmax(y, axis=1)
        index_t = np.argmax(t, axis=1)
        corrects = np.sum(index_y == index_t)
        return 1-corrects/t.shape[0]

    def keep_the_best_fit(self, M, N, Nb, learning_rate, lam, epochs, h, report=None):
        D = self.D
        K = self.K
        
        error_best = sys.maxsize
        epoch_best = -1
        learning_rate_best = -1
        lam_best = -1
        w1_best = self.initializer(M,D+1)
        w2_best = self.initializer(K,M+1)

        y_out = np.empty(shape=(Nb,K))
        z = np.empty(shape=(Nb, M+1))

        error_in_epochs_list = np.empty(shape=(len(learning_rate), len(lam), epochs, 3))
        for lr in range(0,len(learning_rate)):
            for l in range(0,len(lam)):
                w1 = self.initializer(M,D+1)
                w2 = self.initializer(K,M+1)
                for epoch in range(0,epochs):
                    for num1 in range(0, N, Nb):
                        # avoiding batch iteration throw linear algebra
                        z[:,0] = 1
                        z[:,1:] = h(np.dot(self.x_train[num1:num1+Nb], w1.T))
                        y_out = fc.softmax(np.dot(z,w2.T))

                        cost, gradient_w1, gradient_w2 = cost_gradient(self.y_train[num1:num1+Nb],y_out,w1,w2,z,self.x_train[num1:num1+Nb],lam[l])
                        w2 += learning_rate[lr]*gradient_w2
                        # w1 += learning_rate[lr]*gradient_w1

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

                    print("Epoch "+str(epoch+1)+" out of "+str(epochs)+" for learning rate="+str(learning_rate[lr])+", lam="+str(lam[l])+" - cost="+str(cost))

        if report is not None:
            self.save(error_in_epochs_list, learning_rate, lam, error_best, epoch_best, learning_rate_best, lam_best, report)
        return w1_best, w2_best, learning_rate_best, lam_best

    def fit(self, x, y, batch_size, epochs, validation_split=0.2, report=None):
        self.x_train, self.x_validation = self.__split(x/255, validation_split) #normalize x by dividing with 255
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


def cost_gradient(t, y, w1, w2, z, x, l):

    cost = np.sum(np.multiply(t, np.log(y))) - l/2*np.sum(np.square(w2))

    temp_w1 = np.insert(w1, 0, 1, axis=0)
    gradient_w2 = np.dot(np.transpose(t - y),z) - l*w2
    # TODO calc gradient for W1
    gradient_w1 = np.dot((t-y),w2)
    # h3 FALSE
    gradient_w1 = np.dot(np.dot(np.dot((t-y),w2),-np.sin(np.dot(temp_w1,x.T))), x)
    return cost, gradient_w1, gradient_w2

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


# N training examples, D characteristics
# M weight vectors per characteristic
# K classes
# Nb minibatch 
# l learning rate, l>=0
