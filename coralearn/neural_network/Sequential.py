import numpy as np

from coralearn.optimizers.SGD import SGD


class Sequential():
    def __init__(self, layers):
        self.loss = None
        self.layers = layers
        self.optimizer = None

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compile(self, loss, optimizer=SGD):
        self.loss = loss
        self.optimizer = optimizer

    def backward(self, loss_grad):
        # go backwards through the layers
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)
        return loss_grad

    def train(self, X_in, y_in, epochs=100, batch_size=32):
        samples_length = len(X_in)
        for epoch in range(epochs):
            indices = np.random.permutation(samples_length)
            X_shuffled = X_in[indices]
            y_shuffled = y_in[indices]

            epoch_loss = 0
            n_batches = 0

            for i in range(0, samples_length, batch_size):
                end_index = min(i + batch_size, samples_length)
                X_batch = X_shuffled[i:end_index]
                y_batch = y_shuffled[i:end_index]

                # forward
                y_pred = self.forward(X_batch)

                # get loss and gradient
                loss_val, loss_grad = self.loss(y_batch, y_pred)

                epoch_loss += loss_val
                n_batches += 1

                # backward pass
                self.backward(loss_grad)

                if self.optimizer.network_level:
                    # forward/backward + update all layers at once
                    self.optimizer.update(layer=None, layers=self.layers, X=X_batch, y=y_batch, loss=self.loss)
                else:
                    for layer in self.layers:
                        self.optimizer.update(layer=layer)

            # print every 100 epochs the value
            if epoch % 100 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch}, Loss = {avg_loss:.4f}")
