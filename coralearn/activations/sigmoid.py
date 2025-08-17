import numpy as np


def sigmoid(x):
    g = 1 / (1 + np.exp(-x))
    grad = g * (1 - g)
    return g, grad
