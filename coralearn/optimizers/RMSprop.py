import numpy as np
from coralearn.optimizers import Optimizer


class RMSprop(Optimizer):
    def __init__(self, lr=0.01, beta=0.9,epsilon = np.finfo(np.float64).eps):
        self.lr = lr
        self.beta = beta
        self.v = {}  # store per-layer velocities
        self.epsilon = epsilon
    def update(self, layer=None, layers=None, X=None, y=None, loss=None, layer_input=None):

        # Initialize velocities for this layer if not done yet
        if id(layer) not in self.v:
            self.v[id(layer)] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]

        vW, vb = self.v[id(layer)]

        # momentum update
        vW = self.beta * vW + (1-self.beta)*np.pow(layer.dW,2)
        vb = self.beta * vb + (1 - self.beta) * np.pow(layer.db, 2)
        # apply update
        layer.W -= (self.lr/(np.sqrt(vW+self.epsilon))) * layer.dW
        layer.b -= (self.lr/(np.sqrt(vb+self.epsilon))) * layer.db

        # save back
        self.v[id(layer)] = [vW, vb]
