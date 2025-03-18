#!/usr/bin/env python3
"""
bias_variance_tutorial.py
-------------------------
This tutorial demonstrates the bias-variance trade-off using polynomial regression.
We generate noisy data from a sine function, fit polynomials of varying degrees,
and compare training vs. test errors.
"""

import numpy as np
import matplotlib.pyplot as plt

def true_function(x):
    return np.sin(2 * np.pi * x)

def polynomial_design_matrix(x, degree):
    N = x.shape[0]
    X_poly = np.hstack([x**d for d in range(degree+1)])
    return X_poly

# 1. Generate synthetic data
np.random.seed(0)
N = 50
x = np.linspace(-1, 1, N).reshape(-1, 1)
noise_variance = 0.2
y = true_function(x) + noise_variance * np.random.randn(N, 1)

# Split into train (70%) and test (30%)
Ntrain = int(0.7 * N)
x_train = x[:Ntrain]
y_train = y[:Ntrain]
x_test = x[Ntrain:]
y_test = y[Ntrain:]

max_degree = 10
train_errors = np.zeros(max_degree)
test_errors = np.zeros(max_degree)

for d in range(1, max_degree+1):
    X_train = polynomial_design_matrix(x_train, d)
    w = np.linalg.pinv(X_train) @ y_train
    y_pred_train = X_train @ w
    train_errors[d-1] = np.mean((y_train - y_pred_train)**2)
    
    X_test = polynomial_design_matrix(x_test, d)
    y_pred_test = X_test @ w
    test_errors[d-1] = np.mean((y_test - y_pred_test)**2)

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.plot(np.arange(1, max_degree+1), train_errors, 'o--', linewidth=1.5, markersize=8, label="Training Error")
plt.plot(np.arange(1, max_degree+1), test_errors, 'o--', linewidth=1.5, markersize=8, label="Test Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Trade-off")
plt.legend()
plt.grid(True)

# Visual comparison for degree 5
d_select = 5
X_train_select = polynomial_design_matrix(x_train, d_select)
w_select = np.linalg.pinv(X_train_select) @ y_train
x_fine = np.linspace(-1, 1, 200).reshape(-1, 1)
y_true_fine = true_function(x_fine)
X_fine_select = polynomial_design_matrix(x_fine, d_select)
y_pred_fine = X_fine_select @ w_select

plt.subplot(1,2,2)
plt.plot(x_train, y_train, 'bo', label="Training Data")
plt.plot(x_fine, y_true_fine, 'k-', linewidth=1.5, label="True Function")
plt.plot(x_fine, y_pred_fine, 'r-', linewidth=1.5, label=f"Degree {d_select} Fit")
plt.xlabel("x"); plt.ylabel("y")
plt.title(f"Polynomial Fit (Degree = {d_select})")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
