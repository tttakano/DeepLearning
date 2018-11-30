import numpy as np
 
import chainer
import chainer.functions as F
import chainer.links as L
 
 
class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(None, 32, 3, pad=1),
            bn1=L.BatchNormalization(32),
            conv2=L.Convolution2D(None, 64, 3, pad=1),
            bn2=L.BatchNormalization(64),
            conv3=L.Convolution2D(None, 96, 3, pad=1),
            fc6=L.Linear(None, 1000),
            fc7=L.Linear(None, 1000),
            fc8=L.Linear(None, 10),
        )
 
    def __call__(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)
 
        h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), 3, pad=1)
        h = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h))), 3, pad=1)
        h = F.max_pooling_2d(F.relu(self.conv3(h)), 3, pad=1)
        h = F.dropout(F.relu(self.fc6(h)), train=train)
        h = F.dropout(F.relu(self.fc7(h)), train=train)
        h = self.fc8(h)
        
        return F.softmax_cross_entropy(h,t), F.accuracy(h,t)