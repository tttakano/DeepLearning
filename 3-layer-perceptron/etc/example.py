import chainer
import numpy as np
from chainer import cuda,Function,gradient_check,Variable,optimizers,serializers,utils
from chainer import Links,Chain,ChainList
import chainer.functions as F
import chainer.links as L
import csv
import pandas as pd

datagram=pd.read_csv('train.py')
dataset=dataframe.values
dataset=dataset.astype('float32')

def create_dataset(dataset):
    X, Y = [], []
    for i in range(len(dataset)):
        X.append(dataset[i][:5])
        if i<=9000:
            Y.append(dataset[i][5:6])
        elif i>9000:
            Y.append(dataset[i][6:7])
    X = np.array(X).reshape(len(dataset),5)
    Y = np.array(Y).reshape(len(dataset),1)
    return X, Y

def split_data(x, y, test_size=0.1):
    pos = round(len(x) * (1 - test_size))
    trainX, trainY = x[:pos], y[:pos]
    testX, testY   = x[pos:], y[pos:]
    return trainX, trainY, testX, testY

X, Y = create_dataset(dataset)
trainX, trainY, testX, testY = split_data(X, Y, 0.1)
