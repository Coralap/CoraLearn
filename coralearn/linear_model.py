import numpy as np

class LinearModel:
    def __init__(self, input_size):
        self.w = np.zeros(input_size, dtype=np.float32)
        self.b = 0.0

    def predict(self, X):
        return X.dot(self.w) + self.b  # Vectorized dot product

    def compute_gradient(self,X, y):

        # Number of training examples
        m = X.shape[0]
        dj_dw = 0
        dj_db = 0

        for i in range(m):
            f_wb = self.w.dot(X[i]) + self.b
            dj_dw_i = (f_wb - y[i]) * X[i]
            dj_db_i = f_wb - y[i]
            dj_db += dj_db_i
            dj_dw += dj_dw_i
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db


    def fit(self, X, y, epochs, learning_rate=0.1):
        for i in range(epochs):
            dj_dw, dj_db = self.compute_gradient(X, y)

            # Update Parameters using equation (3) above
            self.b = self.b - learning_rate * dj_db
            self.w = self.w - learning_rate * dj_dw
