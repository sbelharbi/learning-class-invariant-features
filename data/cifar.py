import keras
from keras.datasets import cifar10
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import cPickle as pkl


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = y_test.reshape(y_test.size)
y_train = y_train.reshape(y_train.size)
cl = np.unique(y_train)
vl_ind = []
for i in cl:
    ind = np.argwhere(y_train == i)
    ind = ind.reshape(ind.size)
    for k in range(1000):
        np.random.shuffle(ind)
    vl_ind.extend(ind[:int(len(ind)/10)])
    # debug
#    path = "./CIFAR10/" + str(i) + '/'
#    if not os.path.exists(path):
#        os.makedirs(path)
#    for k in ind:
#        fig = plt.figure()
#        plt.imshow(x_train[k])
#        fig.savefig(path + str(k) + ".png")

x_vl = x_train[vl_ind]
y_vl = y_train[vl_ind]
ind_tr = []
for i in range(x_train.shape[0]):
    if i not in vl_ind:
        ind_tr.append(i)
for i in range(10000):
    np.random.shuffle(ind_tr)

new_x_train = x_train[ind_tr]
new_y_train = y_train[ind_tr]

stuff = [(x_train, y_train), (x_vl, y_vl), (x_test, y_test)]
for e in stuff:
    print e[0].shape, e[1].shape

with open("cifar10.pkl", "w") as f:
    pkl.dump(stuff, f)
