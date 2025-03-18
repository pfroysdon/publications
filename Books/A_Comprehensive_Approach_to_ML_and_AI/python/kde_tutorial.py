#!/usr/bin/env python3
"""
kdeTutorial.py
--------------
This tutorial implements Kernel Density Estimation (KDE) from scratch.
It estimates the density of data in vector X using a Gaussian kernel with a given bandwidth.
"""

import numpy as np
import matplotlib.pyplot as plt

def myKDE(X, h, num_points=100):
    # Create evaluation grid
    x_min = np.min(X) - 3*h
    x_max = np.max(X) + 3*h
    x_grid = np.linspace(x_min, x_max, num_points)
    
    # Gaussian kernel function
    K = lambda u: (1/np.sqrt(2*np.pi)) * np.exp(-0.5*u**2)
    
    n = len(X)
    f_hat = np.zeros_like(x_grid)
    for i, x_val in enumerate(x_grid):
        u = (x_val - X) / h
        f_hat[i] = np.sum(K(u))
    f_hat = f_hat / (n * h)
    return x_grid, f_hat

# Generate synthetic data: mixture of two Gaussians
np.random.seed(1)
X = np.concatenate((np.random.randn(100), np.random.randn(100)+3))
h = 0.5
num_points = 200
x_grid, f_hat = myKDE(X, h, num_points)

plt.figure()
plt.plot(x_grid, f_hat, linewidth=2)
plt.title("Kernel Density Estimate for a Mixture of Gaussians")
plt.xlabel("x"); plt.ylabel("Estimated Density")
plt.grid(True)
plt.show()
