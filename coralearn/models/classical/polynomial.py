import numpy as np

from coralearn.models.classical.basemodel import BaseModel

#most of this code is the same as the linear model, i suggest looking at it and then just look at the expand and scale x for further comments
class PolynomialModel(BaseModel):
    def __init__(self, input_size, degree=2):
        super().__init__(input_size)
        self.degree = degree
        self.og_input_size = input_size
        self.w = np.zeros(input_size * degree, dtype=np.float32)
        self.b = 0.0

        self.X_mean = None
        self.X_std = None


    #z-score normalization
    def scale_X(self, X):
        if self.X_mean is None or self.X_std is None:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            # Avoid division by zero
            self.X_std[self.X_std == 0] = 1.0
        return (X - self.X_mean) / self.X_std


    #power every number by 1 - degree to get x**1 x**2 x**3 until x**degree for poly regression
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
        X_scaled = self.scale_X(X)
        poly_X = self.expand_X(X_scaled)
        return poly_X.dot(self.w) + self.b

    def compute_gradient(self, X, y):
        X_scaled = self.scale_X(X)
        X_expanded = self.expand_X(X_scaled)
        m = X_expanded.shape[0]
        dj_dw = np.zeros_like(self.w)
        dj_db = 0

        for i in range(m):
            f_wb = self.w.dot(X_expanded[i]) + self.b
            dj_dw += (f_wb - y[i]) * X_expanded[i]
            dj_db += f_wb - y[i]

        dj_dw /= m
        dj_db /= m

        return dj_dw, dj_db

    def fit(self, X, y, epochs=1000, learning_rate=0.001):
        for _ in range(epochs):
            dj_dw, dj_db = self.compute_gradient(X, y)
            self.w -= learning_rate * dj_dw
            self.b -= learning_rate * dj_db
