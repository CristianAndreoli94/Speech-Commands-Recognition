# provare ad applicare una normalizzazione per casa, whitein è lunga

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


def l2_normalization(X):
    q = np.sqrt((X ** 2).sum(1, keepdims=True))
    q = np.maximum(q, 1e-15)  # 1e -15 avoids division by zero
    X = X / q
    return X


def l1_normalization(X):
    q = np . abs(X).sum(1, keepdims=True)
    q = np . maximum(q, 1e-15)  # 1e -15 avoids division by zero
    X = X / q
    return X


def maxabs_normalization(xtrain, xtest):
    amax = np.abs(xtrain).max(0)
    xtrain = xtrain / amax
    xtest = xtest / amax
    return xtrain, xtest


def meanvar_normalization(xtrain, xtest):
    mu = xtrain.mean(0)
    sigma = xtrain.std(0)
    xtrain = (xtrain - mu) / sigma
    xtest = (xtest - mu) / sigma
    return xtrain, xtest


# xtrain, xtest = meanvar_normalization(xtrain, xtest)
xtrain = l2_normalization(xtrain)
xtest = l2_normalization(xtest)

# plt.imshow(spectrogram)
# plt.colorbar()
# plt.show()

m = xtrain.shape[0]
network = pvml.MLP([1600, 140, 70, 35])
for epoch in range(40):
    network.train(xtrain, ytrain, lr=1e-3, steps=m//100, batch=100)
    # network.train(xtrain, ytrain, lr=1e-3, steps=1)
    predictions, logits = network.inference(xtrain)
    training_acc = (predictions == ytrain).mean()
    predictions, logits = network.inference(xtest)
    test_acc = (predictions == ytest).mean()
    print("Epoch: " + str(epoch) + " Train :", training_acc, "Test: ", test_acc)
network.save("mlp.npz")
