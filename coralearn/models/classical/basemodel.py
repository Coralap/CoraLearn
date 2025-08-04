import numpy as np

class BaseModel:
    def __init__(self, input_size):
        self.w = np.zeros(input_size, dtype=np.float32)
        self.b = 0.0

    def predict(self, X):
        raise NotImplementedError("Implement this for the particular model")

    def compute_gradient(self, X, y):
        raise NotImplementedError("Implement this for the particular model")

    def fit(self, X, y, epochs=100, learning_rate=0.1):
        for _ in range(epochs):
            #change the w and b based on the learning rate and derivatives given by the compute gradient function
            dj_dw, dj_db = self.compute_gradient(X, y)
            self.w -= learning_rate * dj_dw
            self.b -= learning_rate * dj_db
