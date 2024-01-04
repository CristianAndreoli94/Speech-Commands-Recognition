import numpy as np
import pvml
import matplotlib.pyplot as plt


# LOAD THE LIST OF CLASSES
words = open("speech-comands/classes.txt").read().split()
print(words)

# LOAD THE TRAINING AND TEST DATA
data = np.load("speech-comands/train.npz")
Xtrain = data["arr_0"]
Ytrain = data["arr_1"]
print(Xtrain.shape, Ytrain.shape)
data = np.load("speech-comands/test.npz")
Xtest = data["arr_0"]
Ytest = data["arr_1"]
print(Xtest.shape, Ytest.shape)

'''
# SHOW A SAMPLE IMAGE
image = Xtrain[8, :].reshape(20, 80)
# plt.figure(figsize=(20, 20))  # enlarge the figure
plt.imshow(image)
plt.colorbar()
plt.show()
counters = np.bincount(Ytrain)
for i in range(35):
    print(words[i], "\t", counters[i])
'''

# MEAN/VARIANCE NORMALIZATION
# mu = Xtrain.mean(0)
# std = Xtrain.std(0)
'''
image = mu.reshape(20, 80)
# plt.figure(figsize=(20, 20))  # enlarge the figure
plt.imshow(image)
plt.colorbar()
plt.title("Mean")
plt.figure()
image = std.reshape(20, 80)
plt.imshow(image)
plt.colorbar()
plt.title("Stddev")
plt.show()
'''
# Xtrain = (Xtrain - mu) / std
# Xtest = (Xtest - mu) / std

def l2_normalization(X):
    q = np.sqrt((X ** 2).sum(1, keepdims=True))
    q = np.maximum(q, 1e-15)  # 1e -15 avoids division by zero
    X = X / q
    return X


def maxabs_normalization(Xtrain, Xtest):
    amax = np.abs(Xtrain).max(0)
    Xtrain = Xtrain / amax
    Xtest = Xtest / amax
    return Xtrain, Xtest


def whitening(Xtrain, Xtest):
    mu = Xtrain.mean(0)
    sigma = np.cov(Xtrain.T)
    evals, evecs = np.linalg.eigh(sigma)
    w = evecs / np.sqrt(evals)
    Xtrain = (Xtrain - mu) @ w
    Xtest = (Xtest - mu) @ w
    return Xtrain, Xtest


# Xtrain, Xtest = whitening(Xtrain, Xtest)
# Xtrain, Xtest = maxabs_normalization(Xtrain, Xtest)
Xtrain = l2_normalization(Xtrain)
Xtest = l2_normalization(Xtest)


# COMPUTE THE ACCURACY
def accuracy(net, X, Y):
    labels, probs = net.inference(X)
    acc = (labels == Y).mean()
    return acc * 100


# CREATE AND TRAIN THE MULTI-LAYER PERCEPTRON
net = pvml.MLP([1600, 35])
m = Ytrain.size
plt.ion()
train_accs = []
test_accs = []
epochs = []
batch_size = 100
for epoch in range(100):
    net.train(Xtrain, Ytrain, 1e-3, steps=m // batch_size,
              batch=batch_size)
    # if epoch % 5 == 0:
    train_acc = accuracy(net, Xtrain, Ytrain)
    test_acc = accuracy(net, Xtest, Ytest)
    print(epoch, train_acc, test_acc)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    epochs.append(epoch)
    plt.clf()
    plt.plot(epochs, train_accs)
    plt.plot(epochs, test_accs)
    plt.xlabel("epochs")
    plt.ylabel("accuracies (%)")
    plt.legend(["train", "test"])
    plt.pause(0.01)
plt.ioff()
plt.show()

# SAVE THE MODEL TO DISK
net.save("mlp3.npz")

