import numpy as np
from coralearn.models.classical.basemodel import BaseModel
from coralearn.scalers.standard_scaler import StandardScaler

class PolynomialModel(BaseModel):
    def __init__(self, input_size, degree=2):
        super().__init__(input_size * degree)
        self.degree = degree
        self.og_input_size = input_size

        self.X_mean = None
        self.X_std = None

    def expand_X(self, X):
        m, n = X.shape
        poly_X = np.zeros((m, n * self.degree), dtype=np.float32)
        for i in range(m):
            for degree in range(1, self.degree + 1):
                for j in range(n):
                    col = (degree - 1) * n + j
                    poly_X[i][col] = X[i][j] ** degree
        return poly_X

    def predict(self, X):
        X_scaled, _, _ = StandardScaler(X, self.X_mean, self.X_std)
        poly_X = self.expand_X(X_scaled)
        return poly_X.dot(self.w) + self.b

    def compute_gradient(self, X, y):
        scaler = StandardScaler()
        if self.X_mean is None or self.X_std is None:
            # Fit scaler and save mean/std
            X_scaled, self.X_mean, self.X_std = scaler.fit_transform(X)
        else:
            # Use stored mean/std for scaling without fitting again
            scaler.mean_ = self.X_mean
            scaler.scale_ = self.X_std
            X_scaled = scaler.transform(X)

        X_expanded = self.expand_X(X_scaled)
        m = X_expanded.shape[0]

        f_wb = X_expanded.dot(self.w) + self.b
        error = f_wb - y

        dj_dw = X_expanded.T.dot(error) / m
        dj_db = np.sum(error) / m

        return dj_dw, dj_db


