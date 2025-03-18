#!/usr/bin/env python3
"""
gbtTutorial.py
--------------
This demo illustrates gradient boosting trees for regression on synthetic data.
We generate a nonlinear target, fit boosting models, and plot true versus predicted values.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

np.random.seed(1)
X = np.random.randn(100, 2)
y = np.sin(X[:,0]) + 0.5 * X[:,1] + 0.1 * np.random.randn(100)

M = 50  # Number of boosting iterations
eta = 0.1
F_pred = np.mean(y) * np.ones(100)
models = []
for m in range(M):
    residuals = y - F_pred
    tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=3)
    tree.fit(X, residuals)
    update = tree.predict(X)
    F_pred += eta * update
    models.append(tree)

plt.figure()
plt.scatter(y, F_pred, color='b')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Gradient Boosting Trees Regression")
plt.grid(True)
plt.show()
