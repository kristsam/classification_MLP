import math
import numpy as np
import sys, copy
import matplotlib.pyplot as plt
import func as fc
import layers

class MultiLayerPerceptron:

    diagram_count = 0

    def __init__(self):
        # h_l = HiddenLayer
        self.h_l = []
        self.learning_rate = []


    def add(self, layer):
        '''
        Parameters
        ----------
        layer (HiddenLayer):
        '''
        self.h_l.append(layer)

    def compile(self, learning_rate=[0.01], lam=[0.01], initializer='glorot'):
        if isinstance(learning_rate, list):
              self.learning_rate = learning_rate
        elif isinstance(learning_rate, int):
              self.learning_rate = [learning_rate]
        if isinstance(lam, list):
             self.lam = lam
        elif isinstance(lam, int):
             self.lam = [lam]
        self.initializer = initializer

    def __split(self, x, validation_split):
        split_rows = int(x.shape[0]* (1-validation_split))
        return x[:split_rows, :], x[split_rows:, :]

    def save(self, error_in_epochs_list, learning_rate, lam, report):
        fig = plt.figure()
        
        lot_m = ''
        lot_h = ''
        for k in range(0,len(self.h_l)):
            lot_m += 'M'+str(k+1)+'='+str(self.h_l[k].units)+', '
            lot_h += 'h'+str(k+1)+'='+self.h_l[k].activation_to_string()+', '
        fig.suptitle('Multilayer Perceptron \n'+lot_m+'\n'+lot_h+'\n'+'Nb='+str(self.Nb)+', learning rate='+str(learning_rate)+', lambda='+str(lam))

        plt.subplot(2, 1, 1)
        plt.plot(error_in_epochs_list[:, 0]+1,error_in_epochs_list[:, 1], label="training data")
        plt.plot(error_in_epochs_list[:, 0]+1,error_in_epochs_list[:, 2], label="validation data")
        plt.ylabel('error')
        plt.xlabel('epochs')
        plt.axis([1, error_in_epochs_list[-1, 0]+1, 0, 1])

        plt.subplot(2, 1, 2)
        plt.plot(error_in_epochs_list[:, 0]+1,error_in_epochs_list[:, 3], label="training data")
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.axis([1, error_in_epochs_list[-1, 0]+1, np.min(error_in_epochs_list[:, 3]), 0])

        plt.legend()
        fig.savefig(report+'diagram'+str(MultiLayerPerceptron.diagram_count)+'.png')
        print('Diagram saved as '+'"diagram'+str(MultiLayerPerceptron.diagram_count)+'.png" !'+ '\n')
        MultiLayerPerceptron.diagram_count +=1
        return

    def predict(self, x, hidden_layers=None, output_layer=None, fixed=False):
        if not fixed:
            x_new = np.insert(x/255, 0, 1, axis=1)
            h_l = self.h_l_best
            o_l = self.o_l_best
        else:
            x_new = np.copy(x)
            h_l = hidden_layers
            o_l = output_layer
        for k in range(0, len(h_l)):
            if k == 0:
                o = h_l[k].forward_propagation(x_new)
            else:
                o = h_l[k].forward_propagation(o)
        y = o_l.forward_propagation(o) 

        return y

    def score(self, y, t):
        # index of max values array
        index_y = np.argmax(y, axis=1)
        index_t = np.argmax(t, axis=1)
        corrects = np.sum(index_y == index_t)
        return 1-corrects/t.shape[0]

    def fit(self, x, y, batch_size, epochs, validation_split=0.2, report=None):
        
        self.x_train, self.x_validation = self.__split(x/255, validation_split) #normalize x by dividing with 255
        self.y_train, self.y_validation = self.__split(y, validation_split)
        self.N = self.x_train.shape[0]
        self.D = self.x_train.shape[1]
        self.K = y.shape[1]
        self.Nb = batch_size
        self.x_train = np.insert(self.x_train, 0, 1, axis=1)
        self.x_validation = np.insert(self.x_validation, 0, 1, axis=1)
        error_best = sys.maxsize


        for lr in range(0,len(self.learning_rate)):
            for l in range(0,len(self.lam)):
                error_in_epochs_list = np.empty(shape=(epochs, 4))
                error_in_epochs_list[:,3] = 0
                for k in range(0, len(self.h_l)):
                    old_units = 0
                    if k == 0:
                        old_units = self.x_train.shape[1]
                    else:
                        old_units = self.h_l[k-1].layer_units()
                    self.h_l[k].adjust(old_layer_units=old_units, initializer=self.initializer )
                self.o_l = layers.OutputLayer(y.shape[1])
                self.o_l.adjust(self.h_l[-1].layer_units(), initializer=self.initializer)
                for epoch in range(0,epochs):
                    for num1 in range(0, self.N, self.Nb):
                        y_out = self.predict(self.x_train[num1:num1+self.Nb], self.h_l, self.o_l, fixed=True)

                        cost, h_l_gradients, o_l_gradient = cost_gradient(self.y_train[num1:num1+self.Nb],y_out,self.x_train[num1:num1+self.Nb],self.h_l,self.o_l,self.lam[l])
                        self.o_l.w += self.learning_rate[lr]*o_l_gradient
                        for k in range(0,len(self.h_l)):
                            self.h_l[k].w += self.learning_rate[lr]*h_l_gradients[k]
                        
                        error_in_epochs_list[epoch][3] += self.Nb/self.N*cost 

                    # predict and return error
                    predictions_training = self.predict(self.x_train, self.h_l, self.o_l, fixed=True)
                    predictions_validation = self.predict(self.x_validation, self.h_l, self.o_l, fixed=True)
                    error_training = self.score(predictions_training, self.y_train)
                    error_validation = self.score(predictions_validation, self.y_validation)
                    error_in_epochs_list[epoch][0] = epoch
                    error_in_epochs_list[epoch][1] = error_training
                    error_in_epochs_list[epoch][2] = error_validation

                    if error_best > error_validation:
                        error_best = error_validation
                        h_l_best = copy.deepcopy(self.h_l)
                        o_l_best = copy.deepcopy(self.o_l)
                        epoch_best = epoch
                        learning_rate_best = self.learning_rate[lr]
                        lam_best = self.lam[l]

                    print("Epoch "+str(epoch+1)+" out of "+str(epochs)+" for learning rate="+str(self.learning_rate[lr])+", lam="+str(self.lam[l])+" - cost="+str(error_in_epochs_list[epoch][3]))
                print("")

                if report is not None:
                    self.save(error_in_epochs_list, self.learning_rate[lr], self.lam[l], report)
        print("The least error was "+str(error_best)+" found for learning rate="+str(learning_rate_best)+", lam="+str(lam_best)+" and epoch="+str(epoch_best+1)+".\n")
        self.h_l_best = h_l_best
        self.o_l_best = o_l_best
        self.learning_rate_best = learning_rate_best
        self.lam_best = lam_best
        return


def cost_func(t, y, weights, l):
    cost = np.sum(np.multiply(t, np.log(y)))
    for k in range(0, len(weights)):
        cost -= l/2*np.sum(np.square(weights[k]))
    return cost


def check_gradient(t, x, h_l, o_l, l, g_h_l, g_o_l):
    epsilon = 1e-5
    # check the gradient

    temp_o_l = copy.deepcopy(o_l)

    # hidden layers gradient check
    g_h_l_estimate = []
    for j in range(0, len(h_l)):
        g_h_l_estimate.append(np.empty(shape=(2, h_l[j].w.shape[0], h_l[j].w.shape[1])))
        for m in range(0, g_h_l_estimate[j].shape[1]):
            for k in range(0, g_h_l_estimate[j].shape[2]):
                for i in range(0,g_h_l_estimate[j].shape[0]):
                    temp_h_l = copy.deepcopy(h_l)
                    temp_h_l[j].w[m][k] = temp_h_l[j].w[m][k] + ((-1)**i)*epsilon
                    y_out = MultiLayerPerceptron.predict(MultiLayerPerceptron, x=x, hidden_layers=temp_h_l, output_layer=temp_o_l, fixed=True)
                    weights = [temp_h_l[g].w for g in range(0, len(temp_h_l))]
                    weights.append(temp_o_l.w)
                    g_h_l_estimate[j][i][m][k] = cost_func(t, y_out, weights, l)
        
    estimation_g_h_l = []
    for k in range (0, len(g_h_l_estimate)):
        estimation_g_h_l.append((g_h_l_estimate[k][0] - g_h_l_estimate[k][1]) / (2*epsilon))
    
    # output layer gradient check
    temp_h_l = copy.deepcopy(h_l)

    g_o_l_estimate = np.empty(shape=(2, o_l.w.shape[0], o_l.w.shape[1]))
    for m in range(0, g_o_l_estimate.shape[1]):
        for k in range(0, g_o_l_estimate.shape[2]):
            for i in range(0,g_o_l_estimate.shape[0]):
                temp_o_l = copy.deepcopy(o_l)
                temp_o_l.w[m][k] =  temp_o_l.w[m][k] + ((-1)**i)*epsilon
                y_out = MultiLayerPerceptron.predict(MultiLayerPerceptron, x=x, hidden_layers=temp_h_l, output_layer=temp_o_l, fixed=True)
                weights = [temp_h_l[g].w for g in range(0, len(temp_h_l))]
                weights.append(temp_o_l.w)
                g_o_l_estimate[i][m][k] = cost_func(t, y_out, weights, l)

    estimation_g_o_l = (g_o_l_estimate[0] - g_o_l_estimate[1]) / (2*epsilon)

    if np.max(np.abs(estimation_g_o_l - g_o_l)) > epsilon*2:
        print("WARNING. Inaccurate derivative calculated.")  

    for k in range(0,len(h_l)):
        if np.max(np.abs(estimation_g_h_l[k] - g_h_l[k])) >  epsilon*2:
            print("WARNING. Inaccurate derivative calculated.")   
    return

def cost_gradient(t, y, x, h_l, o_l, l):

    weights = [h_l[k].w for k in range(0, len(h_l))]
    weights.append(o_l.w)
    cost = cost_func(t, y, weights, l)

    gradient_hidden_layers = []

    back = t-y
    gradient_output_layer = o_l.back_propagation(back) - l*o_l.w
    back = np.dot(back, o_l.w)
    gradient_hidden_layers.append(h_l[-1].back_propagation(back) - l*h_l[-1].w)
    for k in range(len(h_l)-2,-1,-1):
        back = h_l[k+1].crop(back)
        back = np.dot(back, h_l[k+1].w)
        gradient_hidden_layers.append(h_l[k].back_propagation(back) - l*h_l[k].w)

    # reoreder from end to start
    reorder = [k for k in range(len(h_l)-1,-1,-1)]
    gradient_hidden_layers = [gradient_hidden_layers[i] for i in reorder]
    
    check_gradient(t,x, h_l, o_l, l, gradient_hidden_layers, gradient_output_layer)
    return cost, gradient_hidden_layers, gradient_output_layer
