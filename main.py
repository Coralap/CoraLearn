from coralearn.models.classification.logistic_regression import LogisticRegressionModel
from coralearn.losses.binary_cross_entropy import binary_cross_entropy

import numpy as np
import matplotlib.pyplot as plt

# Example binary classification data
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
