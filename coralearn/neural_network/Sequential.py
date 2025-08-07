class Sequential():
    def __init__(self, layers):
        self.loss = None
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compile(self, loss):
        self.loss = loss
