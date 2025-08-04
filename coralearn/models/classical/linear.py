from coralearn.models.classical.basemodel import BaseModel


class LinearModel(BaseModel):
    def predict(self, X):
        # x*w+b for a normal linear function
        return X.dot(self.w) + self.b

    def compute_gradient(self, X, y):
        #take the number of inputs given
        m = X.shape[0]
        #place holder for the derivatives
        dj_dw = 0
        dj_db = 0

        #add the derivative of each input
        for i in range(m):
            f_wb = self.w.dot(X[i]) + self.b
            dj_dw += (f_wb - y[i]) * X[i]
            dj_db += f_wb - y[i]
        #mean it
        dj_dw /= m
        dj_db /= m

        return dj_dw, dj_db
