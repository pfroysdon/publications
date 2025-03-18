#!/usr/bin/env python3
"""
bias_variance_decomposition.py
------------------------------
This script demonstrates the bias-variance trade-off by:
  1. Generating M independent training sets from a noisy sine function.
  2. Fitting a polynomial regression model (degree 3) to each training set.
  3. Evaluating predictions on a fine grid.
  4. Computing bias² and variance at each grid point.
"""

import numpy as np
import matplotlib.pyplot as plt

def true_function(x):
    return np.sin(2 * np.pi * x)

def polynomial_design_matrix(x, degree):
    # x is an (N,1) vector
    X_poly = np.hstack([x ** d for d in range(degree+1)])
    return X_poly

# Settings
M = 100
N = 30
degree = 3
noise_std = 0.2

x_fine = np.linspace(-1, 1, 200).reshape(-1, 1)
y_true = true_function(x_fine)

predictions = np.zeros((len(x_fine), M))
for m in range(M):
    x_train = np.random.uniform(-1, 1, (N, 1))
    y_train = true_function(x_train) + noise_std * np.random.randn(N, 1)
    X_train = polynomial_design_matrix(x_train, degree)
    # Solve normal equations
    w = np.linalg.pinv(X_train) @ y_train
    X_fine = polynomial_design_matrix(x_fine, degree)
    predictions[:, m] = (X_fine @ w).flatten()

avg_prediction = np.mean(predictions, axis=1)
bias_sq = (avg_prediction - y_true.flatten())**2
variance = np.var(predictions, axis=1)
noise_variance = noise_std**2
total_error = bias_sq + variance + noise_variance

# Plot results
plt.figure(figsize=(9,6))
plt.subplot(2,1,1)
plt.plot(x_fine, y_true, 'k--', linewidth=2, label="True Function")
plt.plot(x_fine, avg_prediction, 'r-', linewidth=2, label="Average Prediction")
for m in range(10):
    plt.plot(x_fine, predictions[:, m], 'b-', linewidth=1, alpha=0.5)
plt.xlabel("x"); plt.ylabel("y")
plt.title("True Function, Average Prediction, and Sample Fits")
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(x_fine, bias_sq, 'r-', linewidth=2, label="Bias^2")
plt.plot(x_fine, variance, 'b-', linewidth=2, label="Variance")
plt.plot(x_fine, total_error, 'k-', linewidth=2, label="Total Expected Error")
plt.xlabel("x"); plt.ylabel("Error")
plt.title("Error Decomposition Across x")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Average Bias^2: {np.mean(bias_sq):.4f}")
print(f"Average Variance: {np.mean(variance):.4f}")
print(f"Average Total Error (incl. noise): {np.mean(total_error):.4f}")
