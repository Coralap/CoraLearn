import numpy as np


class Dense():
    def __init__(self, input_size, output_size, activation):
        self.DA_out = None
        self.Z = None
        self.error_function = None
        self.A_out = None
        self.A_in = None
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.W = np.random.randn(input_size, output_size) * 0.1  # small random weights
        self.b = np.zeros((1, output_size))  # bias vector

    def forward(self, A_in):
        assert A_in.shape[1] == self.input_size, "Input size given does not match the layer's expected input size."
        self.A_in = A_in
        self.Z = np.matmul(A_in, self.W) + self.b
        self.A_out,self.DA_out = self.activation(self.Z)
        return self.A_out

    def compile(self, error_function):
        self.error_function = error_function

    def backward(self, dA, lr=0.1):
        dZ = dA * self.DA_out # sigmoid derivative
        dW = np.matmul(self.A_in.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.matmul(dZ, self.W.T)

        # update weights
        self.W -= lr * dW
        self.b -= lr * db
        return dA_prev

