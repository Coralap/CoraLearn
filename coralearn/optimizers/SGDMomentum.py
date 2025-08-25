import numpy as np

from coralearn.optimizers import Optimizer


class SGDMomentum(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = []

    def update(self, params, grads):

        if not self.v or any(v.shape != p.shape for v, p in zip(self.v, params)):
            self.v = [np.zeros_like(p) for p in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.v[i] = self.beta * self.v[i] + grad

            param[...] -= self.lr * self.v[i]
