#! /usr/bin/env python
 
import argparse
import sys
 
import numpy as np
import six
import matplotlib as plt
import pandas as pd
from scipy import ndimage
 
import chainer
from chainer import training, cuda
from chainer.training import extensions
from chainer import serializers
 
from sklearn.datasets import fetch_mldata
import time
 
 
import CNN
 
 
parser = argparse.ArgumentParser(description='Trainig Program')
parser.add_argument('--gpu','-g', default=-1,type=int,help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
 
insize = 28
ch = 1
batchsize = 100
n_epoch = 100
 
rot = 0
 
train_num = 1000
test_num = 1000
 
sys.stdout.write('Loading dataset...')
 
mnist = fetch_mldata('MNIST original', data_home=".")
 
x_train = []
x_test = []
y_train = []
y_test = []
 
cnt = [0 for i in range(10)]
 
p = np.random.random_integers(0, len(mnist.data), train_num+test_num)
 
for index, (data, label) in enumerate(np.array(zip(mnist.data, mnist.target))[p]):
    image_rotated = np.random.rand()*rot*2-rot
    if(len(x_train)) < train_num:
        x_train.append(ndimage.rotate(data.reshape(insize, insize), image_rotated, reshape=False))
        y_train.append(label)
        cnt[int(label)] += 1
    else:
        if len(x_test) < test_num:
            x_test.append(ndimage.rotate(data.reshape(insize, insize), image_rotated, reshape=False))
            y_test.append(label)
 
 
x_train = np.array(x_train).astype(np.float32).reshape((len(x_train), ch, insize, insize)) / 255.0
y_train = np.array(y_train).astype(np.int32)
x_test = np.array(x_test).astype(np.float32).reshape((len(x_test), ch, insize, insize)) / 255.0
y_test = np.array(y_test).astype(np.int32)
N = len(y_train)
N_test = len(y_test)
 
sys.stdout.write('done')
print('')
for i in range(10):
    sys.stdout.write(str(cnt[i])+':')
print('')
 
result_train_acc = []
result_train_loss = []
result_test_acc = []
result_test_loss = []
 
def main():
 
    model = CNN.CNN()
    if args.gpu >= 0:
        cuda.init(args.gpu)
        model.to_gpu()
 
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    print('')
 
    t_loss = 100
    bst_epoch = 0
 
    for epoch in six.moves.range(1, n_epoch + 1):
        print('epoch', epoch)
        t = time.clock()
 
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        #Train
        for i in six.moves.range(0, N, batchsize):
            #print str(i)
            x_batch = x_train[perm[i:i+batchsize]]
            y_batch = y_train[perm[i:i+batchsize]]
 
            if args.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)
 
            optimizer.zero_grads()
            loss, acc = model(x_batch, y_batch)
            loss.backward()
            optimizer.update()
            sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
            sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)
        print('train mean loss={}, accuracy={}'.format(sum_loss/N, sum_accuracy/N))
        result_train_acc.append(sum_accuracy/N)
        result_train_loss.append(sum_loss/N)
 
        #Test
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, N_test, batchsize):
            x_batch = x_test[i:i+batchsize]
            y_batch = y_test[i:i+batchsize]
            if args.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)
            
            loss, acc = model(x_batch, y_batch, train=False)
            sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
            sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)
        
        print('test mean loss={}, accuracy={}'.format(sum_loss/N_test, sum_accuracy/N_test))
        result_test_acc.append(sum_accuracy/N_test)
        result_test_loss.append(sum_loss/N_test)
 
        t = time.clock() - t
        print(str(t))
 
        if t_loss > sum_loss/N_test:
            #serializers.save_npz(str(rot)+'my.model', model)
            t_loss = sum_loss/N_test
            bst_epoch = epoch
 
    df = pd.DataFrame({'train_acc': result_train_acc, 'train_loss': result_train_loss,
                       'test_acc': result_test_acc, 'test_loss': result_test_loss})
    df.to_csv('./result/'+str(rot)+'_100n.csv')
    
    print(str(bst_epoch)+"is the best test loss:"+str(t_loss))
 
if __name__ == '__main__':
    main()
























