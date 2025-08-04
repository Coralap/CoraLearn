import numpy as np
def zscore_normalize(X, mean=None, std=None):
    X = np.asarray(X)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0, ddof=0)
        std[std == 0] = 1.0  # Avoid division by zero
    return (X - mean) / std, mean, std
