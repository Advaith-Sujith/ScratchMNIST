#Imports
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import datasets

test = datasets.MNIST("mnist784", train=False, download=True)

test_x = test.data.numpy().reshape(10000, 784).T / 255.0
test_y = test.targets.numpy()



with np.load("mnist_train.npz") as model:
    W1 = model["W1"]
    B1 = model["B1"]
    W2 = model["W2"]
    B2 = model["B2"]
    W3 = model["W3"]
    B3 = model["B3"]

#Rectified Linear Unit activation function
def ReLU(Z):
    return np.maximum(0,Z)

#Softmax activation function
def Softmax(Z):
    e = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)

def forwardProp(X, W1, B1, W2, B2, W3, B3):
    Z1 = np.dot(W1, X) + B1
    A1 = ReLU(Z1)

    Z2 = np.dot(W2, A1) + B2
    A2 = ReLU(Z2)

    Z3 = np.dot(W3, A2) + B3
    A3 = Softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3


X = test_x[:, 4:5]
Y = test_y[4]
Z1, A1, Z2, A2, Z3, A3 = forwardProp(X, W1, B1, W2, B2, W3, B3)
pred = np.argmax(A3, axis=0)[0]
plt.imshow(X.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.title(f"Prediction: {pred}, Actual: {Y}")
plt.show()