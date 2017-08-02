import keras
from keras.datasets import cifar10
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import cPickle as pkl
from tools import add_noise
from tools import add_cifar_10
from scipy import ndimage


def repeat_it(x, y, nbr):
    out_x, out_y = None, None
    for i in range(nbr):
        gen = add_noise(x)
        if out_x is None:
            out_x = gen
            out_y = y
        else:
            out_x = np.vstack((out_x, gen))
            out_y = np.hstack((out_y, y))
    return out_x, out_y


def repeat_it_cifar(x, y, nbr, x_cifar):
    out_x, out_y = None, None
    for i in range(nbr):
        gen = add_cifar_10(x, x_cifar)
        if out_x is None:
            out_x = gen
            out_y = y
        else:
            out_x = np.vstack((out_x, gen))
            out_y = np.hstack((out_y, y))
    return out_x, out_y

# MNIST + noise
#path_data = "./data/mnist.pkl"
#f = open(path_data, 'r')
#train, valid, test = pkl.load(f)
#trainx, trainy = train[0], train[1]
#validx, validy = valid[0], valid[1]
#testx, testy = test[0], test[1]
#
## random noise
#times_tr, times_vl, times_ts = 2, 2, 5
#
#trainx_noise, trainy_new = repeat_it(trainx, trainy, times_tr)
#validx_noise, validy_new = repeat_it(validx, validy, times_vl)
#testx_noise, testy_new = repeat_it(testx, testy, times_ts)
#
#stuff = [(trainx_noise, trainy_new), (validx_noise, validy_new),
#         (testx_noise, testy_new)]
#with open("./data/mnist_noise.pkl", "w") as f:
#    pkl.dump(stuff, f)
#path = "./data/mnist_noise/"
#if not os.path.exists(path):
#    os.makedirs(path)
#for k in range(trainx_noise.shape[0]):
#    fig = plt.figure()
#    plt.imshow(trainx_noise[k].reshape(28, 28), cmap='gray')
#    fig.savefig(path + str(k) + ".png")
#    # blurred
#    if k == 10:
#        break


# MNIST + cifar 10.
path_data = "./data/mnist.pkl"
f = open(path_data, 'r')
train, valid, test = pkl.load(f)
trainx, trainy = train[0], train[1]
validx, validy = valid[0], valid[1]
testx, testy = test[0], test[1]
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()

# random noise
times_tr, times_vl, times_ts = 2, 2, 5

trainx_noise, trainy_new = repeat_it_cifar(trainx, trainy, times_tr, x_train_cifar[:40000])
validx_noise, validy_new = repeat_it_cifar(validx, validy, times_vl, x_train_cifar[40000:])
testx_noise, testy_new = repeat_it_cifar(testx, testy, times_ts, x_test_cifar)

stuff = [(trainx_noise, trainy_new), (validx_noise, validy_new),
         (testx_noise, testy_new)]
with open("./data/mnist_img.pkl", "w") as f:
    pkl.dump(stuff, f)
path = "./data/mnist_img/"
if not os.path.exists(path):
    os.makedirs(path)
for k in range(trainx_noise.shape[0]):
    fig = plt.figure()
    plt.imshow(trainx_noise[k].reshape(28, 28), cmap='gray')
    fig.savefig(path + str(k) + ".png")
    # blurred
    if k == 10:
        break