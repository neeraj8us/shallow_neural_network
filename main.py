# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt


def get_input(nm):
    x1 = np.random.rand(5, nm)
    c1 = x1[0, 0]
    # x1 = np.floor(x1)
    # x1[1] = np.power(x1[0], 2)

    x2 = np.random.rand(5, nm)
    # x2[1] = np.power(x2[0], 4)

    X = np.concatenate((x1, x2), axis=1)
    X[2] = (X[0]) * (X[0])
    X[3] = (X[1]) * (X[1])
    X[4] = X[0] * X[1]
    Y1 = 2 * X[1] * X[1] - 2 * X[0] * X[0] < 0.2
    Y2 = (2 * X[1] * X[1] - 2 * X[0] * X[0]) > 0.8
    Y1 = Y1.reshape((1, 2 * nm))
    Y2 = Y2.reshape((1, 2 * nm))
    Y = np.maximum(Y1,Y2)
    Y = Y.reshape((1, 2 * nm))
    print(str(X.shape), str(Y))
    plt.scatter(X[2, :], X[3, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    return X, Y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initalize(nx, n_h):
    W1 = np.random.randn(n_h, nx) * 0.01;
    b1 = np.random.randn(n_h, 1) * 0.01
    W2 = np.random.randn(1, n_h) * 0.01
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2


def forward(W, b1, W2, b2, X):
    A1 = np.tanh(np.dot(W, X) + b1)
    A2 = sigmoid(np.dot(W2, A1) + b2)
    return A1, A2


def grad(X, A1, A2, W2, Y):
    # print(X.shape)
    # print(A.shape)
    # print(Y.shape)
    m = X.shape[1]
    dz2 = A2 - Y
    dw2 = (1 / m) * np.dot(dz2, A1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(W2.T, dz2) * (1 - np.power(A1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
    # print("db = ", str(db), "dW = " , dw)
    return dw1, db1, dw2, db2


def getCost(A, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(A + 0.0000000001) + (1 - Y) * np.log(1 - A + 0.0000000000001))
    cost = np.squeeze(cost)
    return cost


def updateGrad(W1, b1, dW1, db1, W2, b2, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2


def nn_model(X, Y, num_neurons=5, iter=10000, learning_rate=0.8):
    W1, b1, W2, b2 = initalize(X.shape[0], num_neurons)
    # print("W =" , W)
    i = 0
    A1 = None
    A2 = None
    while i < iter:
        i += 1
        A1, A2 = forward(W1, b1, W2, b2, X)
        cost = getCost(A2, Y)
        if i % 1000 == 0:
            print("Cost after ", i, "iterations = ", cost, learning_rate)
        dW1, db1, dW2, db2 = grad(X, A1, A2, W2, Y)
        W1, b1, W2, b2 = updateGrad(W1, b1, dW1, db1, W2, b2, dW2, db2, learning_rate)
        learning_rate = learning_rate * 0.99999

    B = A2
    B[B > 0.5] = 1
    B[B <= 0.5] = 0

    print("predictions", B.shape, str(A2), B)
    return W1, b1, W2, b2


def predict(W1, b1, W2, b2, X):
    A1, A2 = forward(W2, b2, W2, b2, X)
    return A2


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.random.seed(1)
    X, Y = get_input(1000)
    # plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    # plt.show()

    W1, b1, W2, b2 = nn_model(X, Y, 6, 50000, 0.015)
    # X, Y = get_input(100)
    A1, A2 = forward(W1, b1, W2, b2, X)
    # print("Values ", A)
    A2[A2 > 0.5] = 1
    A2[A2 <= 0.5] = 0;
    print(A2.shape)
    print("numberOf1s", np.sum(A2))
    print("numberOf1s", np.sum(Y))
    print(W1)
    print(W2)
    accuracy = np.sum(np.abs((A2 - Y)))
    print("Error count = ", accuracy * 100 / 2000, "%")

    plt.scatter(X[0, :], X[1, :], c=A2, s=40, cmap=plt.cm.Spectral)
    plt.show()
