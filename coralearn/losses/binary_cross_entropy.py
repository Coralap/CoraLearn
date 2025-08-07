import numpy as np


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), eps, 1 - eps)  # clip to avoid log(0)
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce
