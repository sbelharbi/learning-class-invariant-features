import matplotlib.pyplot as plt
import numpy as np
import cPickle as pkl
from keras.datasets import cifar10
from tools import add_noise
from tools import add_cifar_10
import copy


path_data = "./data/out.pkl"
f = open(path_data, 'r')
train, valid, test = pkl.load(f)
trainx, trainy = train[0], train[1]
print trainy
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()
# with open("./data/cifar10_data.pkl", 'w') as fx:
#    pkl.dump(((x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar)), fx)
#with open("./data/cifar10_data.pkl", 'r') as fx:
#    (x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = pkl.load(fx)

trainx_noise = add_noise(copy.deepcopy(trainx))
ind = [0, 1, 6, 3, 7]
trainx_img = add_cifar_10(copy.deepcopy(trainx), x_train_cifar[ind], sh=False)
for k in range(trainx_img.shape[0]):
    fig = plt.figure()
    plt.imshow(trainx_img[k].reshape(28, 28), cmap='gray')
    fig.savefig("./data/"+ str(k) + ".png")
x = np.vstack((trainx, trainx_noise, trainx_img))
# Plot
fig, axes = plt.subplots(3, 5, figsize=(12, 6),
                         subplot_kw={'xticks': [], 'yticks': []})


i = 0
for ax in axes.flat:
    print x_train_cifar.shape
    # img = x_train_cifar[i, :, :, 0].reshape(32, 32)
    img = x[i, :].reshape(28, 28)
    ax.imshow(img, cmap='gray', interpolation="bilinear")
    ax.set_aspect("auto")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    i += 1

fig.subplots_adjust(hspace=0.01, wspace=0.01)
fig.savefig("./data/samples.eps", format="eps", dpi=300)
