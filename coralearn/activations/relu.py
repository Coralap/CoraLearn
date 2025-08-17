import numpy as np


def relu(x):
    g = np.maximum(0, x)
    grad = np.where(x > 0, 1, 0)
    return g,grad
