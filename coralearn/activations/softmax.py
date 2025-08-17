import numpy as np

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    out = exps / np.sum(exps, axis=1, keepdims=True)

    # compute gradient
    batch_size, n_classes = out.shape
    grad = np.zeros((batch_size, n_classes, n_classes))
    for i in range(batch_size):
        y = out[i].reshape(-1, 1)
        grad[i] = np.diagflat(y) - np.dot(y, y.T)

    return out, grad