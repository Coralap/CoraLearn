import numpy as np


def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # actual loss
    loss = np.mean((y_true - y_pred) ** 2)

    # derivative of the loss function
    grad = (2 / y_true.shape[0]) * (y_pred - y_true)

    return loss, grad
