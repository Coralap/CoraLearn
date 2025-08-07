import numpy as np


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability trick
    return exps / np.sum(exps, axis=1, keepdims=True)
