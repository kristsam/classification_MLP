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
    indices = np.random.choice(x.shape[0],size =(x.shape[0]), replace=False)
    return x[indices], y[indices]

def read(path):
    x_train, y_train = choose(path,"train")
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train, y_train = rearrange(x_train, y_train)

    x_test, y_test = choose(path,"test")
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test, y_test = rearrange(x_test, y_test)

    print('MNIST dataset read and ready for use.\n')

    return x_train, y_train, x_test, y_test


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_cifar_10(path):
    FILE_NAME_TRAIN = 'data_batch_'
    TEST_BATCH = 'test_batch'
    x_train = []
    y_train = []
    for i in range(1,6):
        batch = unpickle(path+FILE_NAME_TRAIN+str(i))
        if i == 1:
            x_train =  batch[b'data']
            y_train = batch[b'labels']
        else:
            x_train = np.concatenate((x_train, batch[b'data']))
            y_train = np.concatenate((y_train, batch[b'labels']))

    x_test = unpickle(path+TEST_BATCH)[b'data']
    y_test = np.array(unpickle(path+TEST_BATCH)[b'labels'])

    y_train = one_hot_vector(y_train)
    y_test = one_hot_vector(y_test)

    print('CIFAR_10 dataset read and ready for use.\n')
    return x_train, y_train, x_test, y_test
