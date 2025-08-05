from coralearn.models.classical.basemodel import BaseModel
from coralearn.activations.sigmoid import sigmoid
from coralearn.losses.binary_cross_entropy import binary_cross_entropy

class LogisticRegressionModel(BaseModel):
    def predict(self, X,threshold =0.5):
        # x*w+b for a normal linear function
        return (sigmoid(X.dot(self.w) + self.b) >= threshold).astype(int)


    def predict_prob(self,X):
        return sigmoid(X.dot(self.w) + self.b)


    def compute_gradient(self, X, y):
        # take the number of inputs given
        m = X.shape[0]
        # place holder for the derivatives
        dj_dw = 0
        dj_db = 0

        # add the derivative of each input
        for i in range(m):
            f_wb = sigmoid(self.w.dot(X[i]) + self.b)
            dj_dw += (f_wb - y[i]) * X[i]
            dj_db += f_wb - y[i]
        # mean it
        dj_dw /= m
        dj_db /= m

        return dj_dw, dj_db
