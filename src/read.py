import pandas as pd
import numpy as np
import random

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

def rearrange(x, y):
    x1, x2, x3 = [], [], []
    y1, y2, y3 = [], [], []
    for i in range(0,len(x)):
        r = random.random()
        if r < 0.33:
            x1.append(x[i])
            y1.append(y[i])
        elif r < 0.66:
            x2.append(x[i])
            y2.append(y[i])
        else:
            x3.append(x[i])
            y3.append(y[i])
    return x1+x2+x3, y1+y2+y3

def read(path):
    x_train, y_train = choose(path,"train")
    x_train, y_train = rearrange(x_train, y_train)

    x_test, y_test = choose(path,"test")

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test
