import numpy as np


class RobustScaler:
    def __init__(self):
        self.median_ = None
        self.iqr_ = None

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.median_ = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        self.iqr_ = q3 - q1
        self.iqr_[self.iqr_ == 0] = 1.0

    def transform(self, X):
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("Must fit scaler before transforming data.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return (X - self.median_) / self.iqr_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("Must fit scaler before inverse transforming data.")

        return X_scaled * self.iqr_ + self.median_
