#Imports
import numpy as np
from torchvision import datasets


train = datasets.MNIST("mnist784", train=True, download=True)

train_x = train.data.numpy().reshape(60000,784).T / 255.0
train_y = train.targets.numpy()


#Initialising Weights and Biases
def init_parameters():
    W1 = np.random.randn(64, 784) * np.sqrt(2/784)
    B1 = np.zeros((64,1))

    W2 = np.random.randn(32, 64) * np.sqrt(2/64)
    B2 = np.zeros((32,1))

    W3 = np.random.randn(10, 32) * np.sqrt(2/32)
    B3 = np.zeros((10,1))

    return W1, B1, W2, B2, W3, B3

#Rectified Linear Unit activation function
def ReLU(Z):
    return np.maximum(0,Z)

#Softmax activation function
def Softmax(Z):
    e = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)

#One Hot Encoding Function of Labels
def OneHotEncode(Y):
    return np.eye(10)[Y].T

#Cost Function
def Cost(A3, Y):
    m = Y.shape[0]
    Y = OneHotEncode(Y)
    loss = -Y*np.log(A3 + 1e-8)
    cost = np.sum(loss)/m
    return cost

def Accuracy(A3, Y):
    predictions = np.argmax(A3, axis=0)
    return np.mean(predictions == Y)

def forwardProp(X, W1, B1, W2, B2, W3, B3):
    Z1 = np.dot(W1, X) + B1
    A1 = ReLU(Z1)

    Z2 = np.dot(W2, A1) + B2
    A2 = ReLU(Z2)

    Z3 = np.dot(W3, A2) + B3
    A3 = Softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3

def ReLU_Derivative(Z):
    return (Z>0).astype(float)
#This bottom part is full calculus
def backProp(X, Y, W1, B1, W2, B2, W3, B3, Z1, A1, Z2, A2, Z3, A3):

    m = X.shape[1]

    Y = OneHotEncode(Y)

    dZ3 = A3 - Y #dL3/dZ3
    dW3 = (1/m) * np.dot(dZ3, A2.T)
    dB3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * ReLU_Derivative(Z2)
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    dB2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * ReLU_Derivative(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    dB1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW3, dB3, dW2, dB2, dW1, dB1

def neuron_Learn(dW3, dB3, dW2, dB2, dW1, dB1, W1, B1, W2, B2, W3, B3):
    lr=0.2

    W1 -= lr * dW1
    B1 -= lr * dB1

    W2 -= lr * dW2
    B2 -= lr * dB2

    W3 -= lr * dW3
    B3 -= lr * dB3

    return W1, B1, W2, B2, W3, B3


W1, B1, W2, B2, W3, B3 = init_parameters()

for epoch in range(1000):
        X = train_x
        Y = train_y

        Z1, A1, Z2, A2, Z3, A3 = forwardProp(X, W1, B1, W2, B2, W3, B3)

        dW3, dB3, dW2, dB2, dW1, dB1 = backProp(X, Y, W1, B1, W2, B2, W3, B3, Z1, A1, Z2, A2, Z3, A3)

        W1, B1, W2, B2, W3, B3 = neuron_Learn(dW3, dB3, dW2, dB2, dW1, dB1, W1, B1, W2, B2, W3, B3)

        _, _, _, _, _, A3_full = forwardProp(
            train_x,
            W1, B1, W2, B2, W3, B3
        )

        print(
            f"Loss: {Cost(A3_full, train_y):.4f}, "
            f"Acc: {Accuracy(A3_full, train_y):.4f}"
        )

np.savez("mnist_train1.npz", W1=W1, B1=B1, W2=W2, B2=B2, W3=W3, B3=B3, Accuracy=Accuracy(A3, train_y))