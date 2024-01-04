import numpy as np
import pvml
import matplotlib.pyplot as plt


# LOAD THE LIST OF CLASSES
words = open("speech-comands/classes.txt").read().split()
print(words)


# LOAD TRAINING AND TEST DATA
data = np.load("speech-comands/train.npz")
Xtrain = data["arr_0"]
Ytrain = data["arr_1"]
print(Xtrain.shape, Ytrain.shape)
data = np.load("speech-comands/test.npz")
Xtest = data["arr_0"]
Ytest = data["arr_1"]
print(Xtest.shape, Ytest.shape)


def l2_normalization(X):
    q = np.sqrt((X ** 2).sum(1, keepdims=True))
    q = np.maximum(q, 1e-15)  # 1e -15 avoids division by zero
    X = X / q
    return X


Xtrain = l2_normalization(Xtrain)
Xtest = l2_normalization(Xtest)

'''
# MEAN/VARIANCE NORMALIZAZION
mu = Xtrain.mean(0)
std = Xtrain.std(0)
Xtrain = (Xtrain - mu) / std
Xtest = (Xtest - mu) / std
'''

# MEASURE THE ACCURACY ON A GIVEN SET
def accuracy(net, X, Y):
    labels, probs = net.inference(X)
    acc = (labels == Y).mean()
    return acc * 100


# LOAD THE WEIGHTS
net = pvml.MLP.load("mlp3.npz")


# SHOW THE WEIGHTS FOR A GIVEN CLASS 
w = net.weights[0]
image = w[:, 25].reshape(20, 80)
plt.imshow(image, cmap="seismic", vmin=-0.6, vmax=0.6)
plt.colorbar()
plt.show()


# DISPLAY THE CONFUSION MATRIX
def show_confusion_matrix(Y, predictions):
    classes = Y.max() + 1
    cm = np.empty((classes, classes))
    for klass in range(classes):
        sel = (Y == klass).nonzero()
        counts = np.bincount(predictions[sel], minlength=classes)
        cm[klass, :] = 100 * counts / max(1, counts.sum())
    plt.figure(3,figsize=(20, 20))
    plt.clf()
    plt.imshow(cm, vmin=0, vmax=100, cmap=plt.cm.Blues)
    for i in range(classes):
        for j in range(classes):
            txt = "{:.1f}".format(cm[i, j], ha="center", va="center")
            col = ("black" if cm[i, j] < 75 else "white")
            plt.text(j - 0.25, i, txt, color=col)
    plt.title("Confusion matrix")


predictions, probs = net.inference(Xtest)
show_confusion_matrix(Ytest, predictions)
plt.show()


# COMPUTE AND PRINT THE CONFISION MATRIX
nclasses = len(words)
cm = np.zeros((nclasses, nclasses))
for i in range(Ytest.size):
    cm[Ytest[i], predictions[i]] += 1
cm = cm / cm.sum(1, keepdims=True)
for i in range(nclasses):
    for j in range(nclasses):
        print(f"{cm[i, j]:.2f}", end=" ")
    print()
