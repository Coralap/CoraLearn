from coralearn.models.classical.linear import LinearModel
from coralearn.models.classical.polynomial import PolynomialModel
from coralearn.losses.mse import mean_squared_error
from coralearn.models.classification import k_nearest_neighbor

import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1], [2], [3], [4]], dtype=np.float32)
y = np.array([2, 8, 6, 8], dtype=np.float32)

model = PolynomialModel(input_size=1,degree=3)
model.fit(X, y, epochs=10000, learning_rate=0.01)

print("Weight:", model.w)
print("Bias:", model.b)
print("Prediction for 5:", model.predict(np.array([[5]], dtype=np.float32)))
print("The error is " + str(mean_squared_error(y,model.predict(X))))

plt.plot(X,y,"x")
plt.plot(X,model.predict(X))
plt.show()


