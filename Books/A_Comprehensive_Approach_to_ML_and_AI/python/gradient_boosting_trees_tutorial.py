#!/usr/bin/env python3
"""
gradientBoostingTreesTutorial.py
----------------------------------
This tutorial demonstrates a simplified gradient boosting trees algorithm for binary classification using logistic loss.
It:
  1. Generates synthetic 2D data:
       - Class 1: centered at (2,2)
       - Class 0: centered at (–2,–2)
  2. Trains an ensemble of regression stumps sequentially by fitting the negative gradient (y – p) of the logistic loss.
  3. Computes predictions via an additive model and converts them to probabilities using the sigmoid.
  4. Visualizes the decision boundary on a grid and reports training accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# --- Helper Functions ---
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def design_matrix(X, degree):
    # X: n x 1 vector; return n x (degree+1) matrix with columns [1, x, x^2, ...]
    return np.hstack([X**d for d in range(degree+1)])

def stump_reg_train(X, r):
    """
    Trains a regression stump to predict r.
    The stump splits on one feature at a threshold.
    Returns a dictionary with keys: 'feature', 'threshold', 'c1', 'c2'.
    """
    n, d = X.shape
    best_loss = np.inf
    best_feature = None
    best_threshold = None
    best_c1 = 0
    best_c2 = 0
    for j in range(d):
        thresholds = np.unique(X[:, j])
        for thresh in thresholds:
            left = X[:, j] < thresh
            right = ~left
            if left.sum() == 0 or right.sum() == 0:
                continue
            c1 = np.mean(r[left])
            c2 = np.mean(r[right])
            loss = np.sum((r[left] - c1)**2) + np.sum((r[right] - c2)**2)
            if loss < best_loss:
                best_loss = loss
                best_feature = j
                best_threshold = thresh
                best_c1 = c1
                best_c2 = c2
    stump = {'feature': best_feature, 'threshold': best_threshold, 'c1': best_c1, 'c2': best_c2}
    return stump

def stump_reg_predict(stump, X):
    n = X.shape[0]
    yhat = np.zeros(n)
    feat = stump['feature']
    thresh = stump['threshold']
    yhat[X[:, feat] < thresh] = stump['c1']
    yhat[X[:, feat] >= thresh] = stump['c2']
    return yhat

def gradient_boosting_train(X, y, T, eta):
    """
    Trains an ensemble of regression stumps via gradient boosting.
    X: n x d data matrix.
    y: n x 1 binary labels (0 or 1).
    T: number of boosting rounds.
    eta: learning rate.
    Returns a list of models, each model is a dict with keys 'stump' and 'coef'.
    """
    n = X.shape[0]
    F = np.zeros(n)  # Initial model F(x)=0
    models = []
    for t in range(T):
        p = sigmoid(F)
        # Negative gradient for logistic loss: r = y - p.
        r = y.flatten() - p
        # Fit a regression stump to residuals.
        stump = stump_reg_train(X, r)
        h = stump_reg_predict(stump, X)
        # Optimal multiplier via least-squares: alpha = <r,h>/<h,h>
        numerator = np.sum(r * h)
        denominator = np.sum(h * h) + 1e-12
        alpha = numerator / denominator
        # Update additive model.
        F = F + eta * alpha * h
        models.append({'stump': stump, 'coef': eta * alpha})
    return models

def gradient_boosting_predict(X, models):
    n = X.shape[0]
    F = np.zeros(n)
    for model in models:
        F += model['coef'] * stump_reg_predict(model['stump'], X)
    # Convert score to probability
    preds = sigmoid(F)
    return preds

# --- Main Script ---
# Generate synthetic data
N = 100
X1 = np.random.randn(N, 2) + 2
X0 = np.random.randn(N, 2) - 2
X = np.vstack((X1, X0))
y = np.vstack((np.ones((N,1)), np.zeros((N,1))))

# Train gradient boosting trees model
T = 50       # boosting rounds
eta = 0.1    # learning rate
models = gradient_boosting_train(X, y, T, eta)

# Compute predictions on training data
y_pred = gradient_boosting_predict(X, models)
y_pred_class = (y_pred >= 0.5).astype(int)
train_acc = np.mean(y_pred_class == y.flatten())
print(f'Gradient Boosting Trees Training Accuracy: {train_acc*100:.2f}%')

# --- Visualize Decision Boundary ---
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
gridPoints = np.c_[xx.ravel(), yy.ravel()]
preds_grid = gradient_boosting_predict(gridPoints, models)
preds_grid = preds_grid.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1], color='b', label='Class 1')
plt.scatter(X[y.flatten()==0, 0], X[y.flatten()==0, 1], color='r', label='Class 0')
plt.contourf(xx, yy, preds_grid, levels=[-0.5, 0.5, 1.5], alpha=0.3, colors=['lightcoral','lightblue'])
plt.title('Gradient Boosting Trees Classification')
plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
