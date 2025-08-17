import numpy as np


def sparse_categorical_cross_entropy(y_true, y_pred, eps=1e-15):
    y_true = np.asarray(y_true)
    y_pred = np.clip(y_pred, eps, 1 - eps)

    n_samples = len(y_true)
    correct_class_probs = y_pred[np.arange(n_samples), y_true]

    loss = -np.mean(np.log(correct_class_probs))

    # Gradient: y_pred - one_hot(y_true)
    grad = y_pred.copy()
    grad[np.arange(n_samples), y_true] -= 1
    grad /= n_samples

    return loss, grad
