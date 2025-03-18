#!/usr/bin/env python3
"""
flexibility_vs_interpretability_classification.py
---------------------------------------------------
Compares logistic regression (linear, interpretable) versus k-NN (flexible) on a circular decision boundary.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

def add_intercept(X):
    return np.hstack((np.ones((X.shape[0],1)), X))

def logistic_regression(X, y, lr, num_iters):
    n, d = X.shape
    w = np.zeros((d,1))
    losses = []
    for _ in range(num_iters):
        logits = X @ w
        preds = expit(logits)
        loss = -np.mean(y * np.log(preds+1e-8) + (1-y)*np.log(1-preds+1e-8))
        losses.append(loss)
        grad = (X.T @ (preds - y)) / n
        w = w - lr * grad
    return w, losses

def knn_predict(X_train, y_train, X_test, k):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

# 1. Generate synthetic data
np.random.seed(1)
N = 200
X = 2 * np.random.rand(N,2) - 1
y = (np.sum(X**2, axis=1) < 0.5**2).astype(int)

# 2. Fit logistic regression
X_lr = add_intercept(X)
w_lr, _ = logistic_regression(X_lr, y.reshape(-1,1), lr=0.1, num_iters=1000)

# 3. Prepare grid for decision boundaries
xx, yy = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_lr = add_intercept(grid)
preds_lr = expit(grid_lr @ w_lr)
preds_lr = (preds_lr >= 0.5).astype(int).reshape(xx.shape)

# k-NN predictions
preds_knn = knn_predict(X, y, grid, k=5).reshape(xx.shape)

# 4. Plot decision boundaries
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.contourf(xx, yy, preds_lr, levels=[-0.1,0.5,1.1], alpha=0.3, colors=['lightcoral','lightblue'])
plt.scatter(X[y==0,0], X[y==0,1], c='r', label='Class 0')
plt.scatter(X[y==1,0], X[y==1,1], c='b', label='Class 1')
plt.title('Logistic Regression (Linear)')
plt.xlabel('x1'); plt.ylabel('x2'); plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.contourf(xx, yy, preds_knn, levels=[-0.1,0.5,1.1], alpha=0.3, colors=['lightcoral','lightblue'])
plt.scatter(X[y==0,0], X[y==0,1], c='r', label='Class 0')
plt.scatter(X[y==1,0], X[y==1,1], c='b', label='Class 1')
plt.title('k-NN (k=5, Flexible)')
plt.xlabel('x1'); plt.ylabel('x2'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

print("Logistic regression coefficients (including intercept):")
print(w_lr)
