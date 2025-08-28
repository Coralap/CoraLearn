import numpy as np
from coralearn.optimizers import Optimizer


class SGDMomentum(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = {}  # store per-layer velocities

    def update(self, layer=None, layers=None, X=None, y=None, loss=None, layer_input=None):
        # Initialize velocities for this layer if not done yet
        if id(layer) not in self.v:
            self.v[id(layer)] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]

        vW, vb = self.v[id(layer)]

        # momentum update
        vW = self.beta * vW + layer.dW
        vb = self.beta * vb + layer.db

        # apply update
        layer.W -= self.lr * vW
        layer.b -= self.lr * vb

        # save back
        self.v[id(layer)] = [vW, vb]
