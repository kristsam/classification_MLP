import pandas as pd
import numpy as np

def choose(path,text):
    x = []
    y = []
    for i in range(0,10):
        df = pd.read_csv(path+text+str(i)+".txt", sep=" ", header=None)
        x += df.values.tolist()
        temp = [i for j in range(0,len(df.index))]
        y += temp
    y = one_hot_vector(y)
    return x,y

def one_hot_vector(y):
    y_out = np.zeros((len(y),10))
    for i in range(0,len(y)):
        y_out[i][y[i]] = 1
    return y_out

def read(path):
    x_train_temp, y_train_temp = choose(path,"train")
    x_test, y_test = choose(path,"test")

    x_train = np.array(x_train_temp[:int(len(x_train_temp)*0.8)])
    y_train = np.array(y_train_temp[:int(len(y_train_temp)*0.8)])
    x_validation = np.array(x_train_temp[int(len(x_train_temp)*0.8):])
    y_validation = np.array(y_train_temp[int(len(y_train_temp)*0.8):])
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_validation, y_validation, x_test, y_test

# TODO refactor