import numpy as np


def sparse_categorical_cross_entropy(y_true, y_pred, eps=1e-15):
    y_true = np.asarray(y_true)

    y_pred = np.clip(np.asarray(y_pred), eps, 1 - eps)  # clip to avoid log(0)

    correct_class_probs = y_pred[np.arange(len(y_true)), y_true]
    losses = -np.log(correct_class_probs)
    loss = np.mean(losses)

    grad = np.zeros_like(y_pred)
    grad[np.arange(len(y_true)), y_true] = -1 / correct_class_probs
    grad /= len(y_true)

    return loss, grad
