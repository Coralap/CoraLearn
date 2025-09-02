import numpy as np
from coralearn.optimizers import Optimizer


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = {}  # time step per layer

    def update(self, layer=None, layers=None, X=None, y=None, loss=None, layer_input=None):
        layer_id = id(layer)

        # initialize state if first time
        if layer_id not in self.m:
            self.m[layer_id] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]
            self.v[layer_id] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]
            self.t[layer_id] = 0

        mW, mb = self.m[layer_id]
        vW, vb = self.v[layer_id]
        self.t[layer_id] += 1
        t = self.t[layer_id]

        # first moment update
        mW = self.beta_1 * mW + (1 - self.beta_1) * layer.dW
        mb = self.beta_1 * mb + (1 - self.beta_1) * layer.db

        # second moment update
        vW = self.beta_2 * vW + (1 - self.beta_2) * np.power(layer.dW, 2)
        vb = self.beta_2 * vb + (1 - self.beta_2) * np.power(layer.db, 2)

        # bias correction
        mW_hat = mW / (1 - self.beta_1 ** t)
        mb_hat = mb / (1 - self.beta_1 ** t)
        vW_hat = vW / (1 - self.beta_2 ** t)
        vb_hat = vb / (1 - self.beta_2 ** t)

        # parameter update
        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.epsilon)

        # save back
        self.m[layer_id] = [mW, mb]
        self.v[layer_id] = [vW, vb]
