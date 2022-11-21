#cython: language_level=3
import numpy as np
cimport numpy as cnp

cdef sig(float x):
    return 1 / (1 + np.exp(-x))


cdef cost_function(y, y_hat):
    cost_acumulator = 0
    for i in range(len(y)):
        cost_acumulator += y[i] * np.log(y_hat[i]) + (1 - y[i]) * np.log(1 - y_hat[i])
    return -(1 / len(y)) * cost_acumulator


cdef predict(X, W, b):
    predictions = []
    for x in X:
        z = sum([x[i] * W[i] for i in range(len(x))]) + b
        y_i = sig(z)
        predictions.append(y_i)
    return predictions


cdef get_gradients(X_train, y_train, y_hat):
    cdef cnp.ndarray[cnp.double_t,ndim=2] dw
    cdef cnp.ndarray[cnp.double_t, ndim=1] db
    dw = np.zeros(np.shape(X_train))
    db = np.zeros(np.shape(y_train))
    cdef int number_row = np.shape(X_train)[0]
    cdef int number_column = np.shape(X_train)[1]
    cdef int i, j
    for i in range(number_row):
        for j in range(number_column):
            dw[i][j] = (y_train[i] - y_hat[i]) * X_train[i][j]
        db[i] += (y_train[i] - y_hat[i])
    return -1 * np.average(dw, axis=0), -1 * np.average(db)


cdef update_params(W, dW, b, dB, learning_rate=0.1):
    cdef int i
    cdef list new_w = [W[i] - learning_rate * dW[i] for i in range(len(W))]
    cdef float new_b = b - learning_rate * dB
    return new_w, new_b


def training_model():
    cdef cnp.ndarray[cnp.int_t, ndim=2] X
    cdef cnp.ndarray[cnp.int_t, ndim=1] y

    X = np.random.randint(10, size=(1000, 3))
    y = np.random.randint(1, size=1000)

    cdef int x
    cdef list W = [np.random.normal() for x in range(np.shape(X)[1])]
    cdef float b = np.random.normal()
    cdef int niter = 1000
    cdef float learning_rate = 0.1
    cdef list y_hat

    for i in range(niter):
        y_hat = predict(X, W, b)
        cost = cost_function(y, y_hat)
        dW, dB = get_gradients(X, y, y_hat)
        W, b = update_params(W, dW, b, dB, learning_rate)

