import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0

    def transform(self, X):

        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Must fit scaler before transforming data.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):

        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Must fit scaler before inverse transforming data.")
        return X_scaled * self.std_ + self.mean_
