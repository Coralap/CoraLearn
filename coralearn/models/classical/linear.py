import numpy as np

from coralearn.models.classical.basemodel import BaseModel


class LinearModel(BaseModel):
    def predict(self, X):
        # x*w+b for a normal linear function
        return X.dot(self.w) + self.b

    def compute_gradient(self, X, y):
        # take the number of inputs given
        m = X.shape[0]

        f_wb = X.dot(self.w) + self.b

        error = f_wb - y
        dj_dw = X.T.dot(error) / m
        dj_db = np.sum(error) / m

        return dj_dw, dj_db
