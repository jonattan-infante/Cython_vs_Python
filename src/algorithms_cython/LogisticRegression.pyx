import numpy as np


def sig(x):
    return 1 / (1 + np.exp(-x))


def cost_function(y, y_hat):
    cost_acumulator = 0
    for i in range(len(y)):
        cost_acumulator += y[i] * np.log(y_hat[i]) + (1 - y[i]) * np.log(1 - y_hat[i])
    return -(1 / len(y)) * cost_acumulator


def predict(X, W, b):
    predictions = []
    for x in X:
        z = sum([x[i] * W[i] for i in range(len(x))]) + b
        y_i = sig(z)
        predictions.append(y_i)
    return predictions


def get_gradients(X_train, y_train, y_hat):
    dw, db = np.zeros(np.shape(X_train)), np.zeros(np.shape(y_train))
    number_row = np.shape(X_train)[0]
    number_column = np.shape(X_train)[1]
    for i in range(number_row):
        for j in range(number_column):
            dw[i][j] = (y_train[i] - y_hat[i]) * X_train[i][j]
        db[i] += (y_train[i] - y_hat[i])
    return -1 * np.average(dw, axis=0), -1 * np.average(db)


def update_params(W, dW, b, dB, learning_rate=0.1):
    new_w = [W[i] - learning_rate * dW[i] for i in range(len(W))]
    new_b = b - learning_rate * dB
    return new_w, new_b


def training_model():
    X = np.random.randint(10, size=(1000, 3))
    y = np.random.randint(1, size=1000)

    W = [np.random.normal() for x in range(np.shape(X)[1])]
    b = np.random.normal()
    niter = 1000
    learning_rate = 0.1
    for i in range(niter):
        y_hat = predict(X, W, b)
        cost = cost_function(y, y_hat)
        dW, dB = get_gradients(X, y, y_hat)
        W, b = update_params(W, dW, b, dB, learning_rate)

