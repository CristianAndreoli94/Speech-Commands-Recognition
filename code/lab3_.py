import numpy as np
import pvml
import matplotlib.pyplot as plt

words = open("data/classes.txt").read().split()
print(words)

data = np.load("data/train.npz")
xtrain = data["arr_0"]
ytrain = data["arr_1"]
print(xtrain.shape, ytrain.shape)
data = np.load("data/test.npz")
xtest = data["arr_0"]
ytest = data["arr_1"]
print(xtest.shape, ytest.shape)
spectrogram = xtrain[0, :].reshape(20, 80)

# plt.imshow(spectrogram)
# plt.colorbar()
# plt.show()

mu = xtrain.mean(0)
std = xtrain.std(0)

xtrain = (xtrain - mu) / std
xtest = (xtest - mu) / std

# plt.imshow(spectrogram)
# plt.colorbar()
# plt.show()


def show_weights(network):
    w = network.weights[0]
    maxval = np.abs(w).max()
    for klass in range(35):
        plt.subplot(5, 7, klass+1)
        plt.imshow(w[:, 0].reshape(20, 80), cmap="seismic", vmin=maxval, vmax=maxval)
        plt.title(words[klass])
    plt.show()


def make_confusion_matrix(predictions, labels):
    cmat = np.zeros((35, 35))
    for i in range(predictions.size):
        cmat[labels[i], predictions[i]] += 1
    return cmat


def display_confusion_matrix(cmat):
    plt.figure(figsize=(20,20))
    plt.imshow(cmat, cmap="Blues")
    for i in range(35):
        for j in range(35):
            val = int(cmat[i, j])
            plt.text(j, i, int(val))
    plt.title("Confusion matrix")

    plt.show()
    print(" " * 10, end="")
    for j in range(35):
        print(f"{words[j][:4]:4} ", end="")
    print()

'''
def display_confusion_matrix(cmat):
    print(" " * 10, end="")
    for j in range(35):
        print(f"{words[j][:4]:4} ", end="")
    print()
    for i in range(35):
        print(f"{words[i]:10}", end="")
        for j in range(35):
            val = int(cmat[i, j])
            print(f"{val:4d} ", end="")
        print()
'''

m = xtrain.shape[0]
network = pvml.MLP.load("mlp2.npz")
# show_weights(network)
predictions, logits = network.inference(xtest)
cmat = make_confusion_matrix(predictions, ytest)
display_confusion_matrix(cmat)
