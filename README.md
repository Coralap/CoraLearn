# Coralearn
An AI library written using only **NumPy**.

![PyPI](https://img.shields.io/pypi/v/coralearn)
![Python](https://img.shields.io/pypi/pyversions/coralearn)
![Downloads](https://img.shields.io/pypi/dm/coralearn)
![License](https://img.shields.io/github/license/Coralap/CoraLearn)

---

## Installation⬇️
```bash
pip install coralearn
```

## Quick Example

Here’s a simple neural network example using **CoraLearn**:

```python
import numpy as np
import pandas as pd

from coralearn.neural_network import Dense, Sequential
from coralearn.optimizers import SGDMomentum
from coralearn.activations import relu, linear_activation
from coralearn.losses import mean_squared_error
from coralearn.scalers import MinMaxScaler

# Alternatively, you can import everything directly from coralearn
# from coralearn import Dense, Sequential, relu, linear_activation, mean_squared_error, SGDMomentum, MinMaxScaler

# Load training data
train_data = pd.read_csv("Train.csv")

X = train_data[["col1", "col2", "col3"]].values   # your chosen input columns
y = train_data["target"].values                   # target column

# Scale the input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Build a sequential model
model = Sequential([
    Dense(input_size=32, output_size=16, activation=relu),
    Dense(input_size=16, output_size=8, activation=relu),
    Dense(input_size=8, output_size=1, activation=linear_activation),
])

# Compile with loss function and an optimizer
model.compile(loss=mean_squared_error, optimizer=SGDMomentum(lr=0.05))

# Train the model
model.train(X_scaled, y, epochs=20, batch_size=32)

# Make a prediction on new data
X_new = np.random.rand(5, 32)   # example new inputs
X_new_scaled = scaler.transform(X_new)

y_pred = model.forward(X_new_scaled)
print("Predictions:", y_pred)
```

## Current Features

### Scalers 🔢 
- MinMaxScaler
- RobustScaler
- StandardScaler

### Losses 📉
- binary_cross_entropy
- mean_squared_error
- sparse_categorical_cross_entropy

### Optimizers⚡ 
- SGD
- SGDMomentum

### Activations ⏰
- relu
- linear_activation
- softmax
- sigmoid

### Neural Network Components 🏗️ 
- Dense (fully connected layer)
- Sequential (model container)
- CNN coming soon
