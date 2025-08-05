import numpy as np


class MinMaxScaler:
    def __init__(self):
        self.X_min = None
        self.X_max = None

    def fit(self,X):
        self.X_min = np.min(X, axis=0)
        self.X_max = np.max(X, axis=0)


    def transform(self, X):

        if self.X_min is None or self.X_max is None:
            raise RuntimeError("Must fit scaler before transforming data.")
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1

        return (X - self.X_min) / denom

    def fit_transform(self, X):

        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        if self.X_min is None or self.X_max is None:
            raise RuntimeError("Must fit scaler before inverse transforming data.")
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1

        return X_scaled * denom + self.X_min
