import numpy as np
from coralearn.optimizers import Optimizer


class NAG(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = {}  # velocities per layer

    @property
    def network_level(self):
        return True

    def update(self, layers=None, X=None, y=None, loss=None, **kwargs):
        if layers is None or X is None or y is None or loss is None:
            raise ValueError("Network-level NAG requires `layers`, `X`, `y`, and `loss` arguments.")

        # Initialize velocities if not already
        for layer in layers:
            if id(layer) not in self.v:
                self.v[id(layer)] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]

        # Backup weights
        backup = [(layer.W.copy(), layer.b.copy()) for layer in layers]

        # Compute lookahead weights: W_lookahead = W - beta * v
        for layer in layers:
            vW, vb = self.v[id(layer)]
            layer.W -= self.beta * vW
            layer.b -= self.beta * vb

        # Forward pass lookahead
        out = X
        for layer in layers:
            out = layer.forward(out)

        # Compute loss at lookahead
        loss_val, loss_grad = loss(y, out)

        # Backward to get gradients
        grad = loss_grad
        for layer in reversed(layers):
            grad = layer.backward(grad)

        # Restore weights
        for layer, (W_orig, b_orig) in zip(layers, backup):
            layer.W[...] = W_orig
            layer.b[...] = b_orig

        # Update velocities and apply weight
        for layer in layers:
            vW, vb = self.v[id(layer)]
            # Standard NAG update
            vW_new = self.beta * vW + self.lr * layer.dW
            vb_new = self.beta * vb + self.lr * layer.db
            layer.W -= vW_new
            layer.b -= vb_new
            # Save updated velocities
            self.v[id(layer)] = [vW_new, vb_new]
