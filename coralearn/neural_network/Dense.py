import numpy as np


class Dense():
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.W = np.random.randn(input_size, output_size) * 0.01  # small random weights
        self.b = np.zeros((1, output_size))  # bias vector


    def forward(self, A_in):
        assert A_in.shape[1] == self.input_size, "Input size given does not match the layer's expected input size."
        z = np.matmul(A_in, self.W) + self.b
        a_out = self.activation(z)
        return a_out
