import chainer
import numpy as np
from chainer import cuda,Function,gradient_check,Variable,optimizers,serializers,utils
from chainer import Link,Chain,ChainList
import chainer.functions as F
import chainer.links as L
import csv
import pandas as pd

dataframe = pd.read_csv('module.csv')
dataset = dataframe.values
dataset = dataset.astype('float32')

def create_dataset(dataset):
    X, Y = [], []
    for i in range(len(dataset)):
        X.append(dataset[i][:5])
        if i<=9000:
            Y.append(dataset[i][5:6])
        elif i>9000:
            Y.append(dataset[i][6:7])
    X = np.array(X).reshape(len(dataset),5).astype(np.float32)
    Y = np.array(Y).reshape(len(dataset),1).astype(np.float32)
    return X, Y

def split_data(x, y, test_size=0.1):
    pos = round(len(x) * (1 - test_size))
    trainX, trainY = x[:pos-1], y[:pos-1]
    testX, testY   = x[pos-1:], y[pos-1:]
    return trainX, trainY, testX, testY

X, Y = create_dataset(dataset)
trainX, trainY, testX, testY = split_data(X, Y, 0.1)

class MyModel (Chain):
    def __init__(self):
        super(MyModel, self).__init__(
            l1=L.Linear(5,3),
            l2=L.Linear(3,1),
        )
        
    def __call__(self,x,y):
        return F.mean_squared_error(self.fwd(x), y)

    def fwd(self,x):
         h1 = F.relu(self.l1(x))
         h2 = self.l2(h1)
         return h2
        
model = MyModel()
optimizer = optimizers.Adam()
optimizer.setup(model)

for i in range(10000000):
    x = Variable(trainX)
    y = Variable(trainY)
    model.zerograds()
    loss = model(x,y)
    loss.backward()
    print("i={} loss={}".format(i,loss.data))
    optimizer.update()
    
train_loss=loss
test_loss=0
xt=Variable(testX)
yt=model.fwd(xt)
ans=yt.data
test_loss=F.mean_squared_error(ans, testY)
print("train_loss={},test_loss={}".format(train_loss.data,test_loss.data))
