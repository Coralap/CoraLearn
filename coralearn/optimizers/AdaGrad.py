import numpy as np
from coralearn.optimizers import Optimizer


class AdaGrad(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr
        self.v = {}  # store per-layer velocities

    def update(self, layer=None, layers=None, X=None, y=None, loss=None, layer_input=None):
        epsilon = np.finfo(np.float64).eps
        # Initialize velocities for this layer if not done yet
        if id(layer) not in self.v:
            self.v[id(layer)] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]

        vW, vb = self.v[id(layer)]

        # momentum update
        vW += np.pow(layer.dW,2)
        vb += np.pow(layer.db,2)

        # apply update
        layer.W -= (self.lr/(epsilon+np.sqrt(vW)))*layer.dW
        layer.b -= (self.lr/(epsilon+np.sqrt(vb)))*layer.db

        # save back
        self.v[id(layer)] = [vW, vb]
