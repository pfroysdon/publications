#!/usr/bin/env python3
"""
flexibility_vs_interpretability.py
------------------------------------
Demonstrates the trade-off between flexibility and interpretability.
We fit a linear model and a 5th-degree polynomial to noisy sine data.
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data
np.random.seed(1)
N = 50
x = np.linspace(0, 1, N).reshape(-1, 1)
true_function = lambda x: np.sin(2 * np.pi * x)
noise_std = 0.1
y = true_function(x) + noise_std * np.random.randn(N, 1)

# 2. Fit linear regression (degree 1)
def design_matrix(x, degree):
    return np.hstack([x**d for d in range(degree+1)])

X_linear = design_matrix(x, 1)
w_linear = np.linalg.pinv(X_linear) @ y

# 3. Fit polynomial regression (degree 5)
X_poly = design_matrix(x, 5)
w_poly = np.linalg.pinv(X_poly) @ y

# 4. Predictions on fine grid
x_fine = np.linspace(0, 1, 200).reshape(-1, 1)
y_true = true_function(x_fine)
y_pred_linear = design_matrix(x_fine, 1) @ w_linear
y_pred_poly = design_matrix(x_fine, 5) @ w_poly

# 5. Plot results
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(x, y, color='b', label='Data')
plt.plot(x_fine, y_true, 'k--', linewidth=2, label='True Function')
plt.plot(x_fine, y_pred_linear, 'r-', linewidth=2, label='Linear Fit')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Linear Regression (Interpretable)')
plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(x, y, color='b', label='Data')
plt.plot(x_fine, y_true, 'k--', linewidth=2, label='True Function')
plt.plot(x_fine, y_pred_poly, 'r-', linewidth=2, label='5th Degree Fit')
plt.xlabel('x'); plt.ylabel('y'); plt.title('5th Degree Polynomial Regression (Flexible)')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

print("Linear regression coefficients:")
print(w_linear)
print("5th degree polynomial coefficients:")
print(w_poly)
