import numpy as np

from coralearn.models.classical.basemodel import BaseModel
from coralearn.activations.sigmoid import sigmoid
from coralearn.losses.binary_cross_entropy import binary_cross_entropy


class LogisticRegressionModel(BaseModel):
    def predict(self, X, threshold=0.5):
        # x*w+b for a normal linear function
        return (sigmoid(X.dot(self.w) + self.b) >= threshold).astype(int)

    def predict_prob(self, X):
        return sigmoid(X.dot(self.w) + self.b)

    def compute_gradient(self, X, y):
        # take the number of inputs given
        m = X.shape[0]

        f_wb = sigmoid(X.dot(self.w) + self.b)
        error = f_wb - y
        dj_dw = X.T.dot(error) / m
        dj_db = np.sum(error) / m

        return dj_dw, dj_db
