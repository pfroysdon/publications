#!/usr/bin/env python3
"""
hyperparamTutorial.py
---------------------
This tutorial demonstrates hyperparameter tuning for ridge regression.
It generates synthetic data, performs k-fold cross-validation to select the best regularization parameter (lambda),
and then trains a final model. It also plots a validation curve and learning curves.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
# Generate synthetic data for regression
n = 200
X = np.linspace(0, 10, n).reshape(-1,1)
y = np.sin(X) + 0.5 * np.random.randn(n,1)

def holdout_split(X, y, test_ratio):
    n = X.shape[0]
    indices = np.random.permutation(n)
    n_test = int(np.round(test_ratio * n))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

X_train, y_train, X_test, y_test = holdout_split(X, y, 0.2)

def ridge_regression(X, y, lam):
    # Closed-form solution: theta = (X^T X + lam*I)^{-1} X^T y
    n, p = X.shape
    I = np.eye(p)
    theta = np.linalg.inv(X.T @ X + lam * I) @ (X.T @ y)
    return theta

def k_fold_cv(X, y, k, lam):
    n = X.shape[0]
    indices = np.random.permutation(n)
    fold_sizes = (n // k) * np.ones(k, dtype=int)
    fold_sizes[:n % k] += 1
    current = 0
    errors = []
    for fold_size in fold_sizes:
        start, end = current, current + fold_size
        val_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        theta = ridge_regression(X_train_fold, y_train_fold, lam)
        y_pred = X_val_fold @ theta
        errors.append(np.mean((y_val_fold - y_pred)**2))
        current = end
    return np.mean(errors)

# Hyperparameter grid for lambda
lambda_grid = np.logspace(-4, 2, 10)
cv_errors = np.zeros(len(lambda_grid))
k = 5
for i, lam in enumerate(lambda_grid):
    cv_errors[i] = k_fold_cv(X_train, y_train, k, lam)
best_idx = np.argmin(cv_errors)
best_lambda = lambda_grid[best_idx]
print(f"Best lambda from grid search: {best_lambda:.4f}")

# Train final model using best lambda
theta = ridge_regression(X_train, y_train, best_lambda)
y_pred_test = X_test @ theta
test_mse = np.mean((y_test - y_pred_test)**2)
print(f"Test MSE: {test_mse:.4f}")

# Plot validation curve
plt.figure()
plt.semilogx(lambda_grid, cv_errors, 'bo-', linewidth=2)
plt.xlabel("Lambda")
plt.ylabel("Cross-Validation MSE")
plt.title("Validation Curve")
plt.grid(True)
plt.show()

def learning_curves(X, y, k, lam):
    n = X.shape[0]
    sizes = np.linspace(int(0.1*n), n, 10, dtype=int)
    train_errors = []
    val_errors = []
    for size in sizes:
        idx = np.random.permutation(n)
        train_idx = idx[:size]
        val_idx = idx[size:]
        X_train_part = X[train_idx]
        y_train_part = y[train_idx]
        X_val_part = X[val_idx]
        y_val_part = y[val_idx]
        theta = ridge_regression(X_train_part, y_train_part, lam)
        train_err = np.mean((y_train_part - X_train_part @ theta)**2)
        val_err = np.mean((y_val_part - X_val_part @ theta)**2)
        train_errors.append(train_err)
        val_errors.append(val_err)
    return sizes, np.array(train_errors), np.array(val_errors)

sizes, train_errs, val_errs = learning_curves(X_train, y_train, k, best_lambda)
plt.figure()
plt.plot(sizes, train_errs, 'b-', linewidth=2, label="Training Error")
plt.plot(sizes, val_errs, 'r-', linewidth=2, label="Validation Error")
plt.xlabel("Training set size")
plt.ylabel("MSE")
plt.title("Learning Curves")
plt.legend()
plt.grid(True)
plt.show()
