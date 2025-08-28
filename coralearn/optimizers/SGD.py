from coralearn.optimizers import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layer=None, layers=None, X=None, y=None, loss=None, layer_input=None):
        # Simple gradient descent
        layer.W -= self.lr * layer.dW
        layer.b -= self.lr * layer.db
