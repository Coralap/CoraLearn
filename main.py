from coralearn.activations.sigmoid import sigmoid
from coralearn.models.classification.logistic_regression import LogisticRegressionModel
from coralearn.losses.binary_cross_entropy import binary_cross_entropy

import numpy as np
import matplotlib.pyplot as plt

from coralearn.neural_network.Dense import Dense
from coralearn.neural_network.Sequential import Sequential


# Example binary classification data
def check_logistic():
    X = np.array([[1], [2], [3], [4], [5], [6]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)  # binary labels

    model = LogisticRegressionModel(input_size=1)
    model.fit(X, y, epochs=10000, learning_rate=0.1)

    print("Weight:", model.w)
    print("Bias:", model.b)

    # Predict probabilities for new inputs
    x_test = np.array([[2.5], [4.5], [6.5]], dtype=np.float32)
    probs = model.predict(x_test)
    print("Predicted probabilities:", probs)

    # Predict classes using 0.5 threshold
    pred_classes = (probs >= 0.5).astype(int)
    print("Predicted classes:", pred_classes)

    # Compute loss on training data
    train_preds = model.predict(X)
    loss = binary_cross_entropy(y, train_preds)
    print(f"Training binary cross-entropy loss: {loss:.4f}")

    # Plot data and predicted probabilities
    plt.scatter(X, y, c='blue', label='True labels',marker ="x")
    plt.scatter(X, train_preds, c='red', label='Predicted probabilities')
    plt.xlabel('X')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

def check_neural():
    X = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    np.random.seed(42)

    # Initialize your model
    model = Sequential([
        Dense(input_size=3, output_size=4, activation=sigmoid),
        Dense(input_size=4, output_size=2, activation=sigmoid),
        Dense(input_size=2, output_size=1, activation=sigmoid)
    ])
    output = model.forward(X)
    print(output)

check_neural()