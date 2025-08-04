import numpy as np
from collections import Counter

class KNNModel:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, X):
        final_preds = []

        for x_test in X:
            distances = [(i, self.distance(x_test, x_train)) for i, x_train in enumerate(self.X_train)]

            top_k = sorted(distances, key=lambda pair: pair[1])[:self.k]

            top_k_labels = [self.y_train[i] for i, _ in top_k]

            most_common = Counter(top_k_labels).most_common(1)[0][0]
            final_preds.append(most_common)

        return final_preds
