from coralearn.models.linear import LinearModel
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1], [2], [3], [4]], dtype=np.float32)
y = np.array([2, 8, 6, 8], dtype=np.float32)

model = LinearModel(input_size=1)
model.fit(X, y, epochs=1000, learning_rate=0.01)

print("Weight:", model.w)
print("Bias:", model.b)
print("Prediction for 5:", model.predict(np.array([[5]], dtype=np.float32)))

plt.plot(X,y,"x")
plt.plot(X,model.predict(X))
plt.show()