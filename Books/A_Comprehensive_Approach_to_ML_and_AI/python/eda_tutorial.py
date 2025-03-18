#!/usr/bin/env python3
"""
edaTutorial.py
--------------
This tutorial demonstrates an exploratory data analysis (EDA) pipeline.
A synthetic dataset with 5 features is generated, and PCA is applied to reduce the dimensionality to 2.
The projected data are then visualized.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
X = np.random.randn(200, 5)  # 200 samples, 5 features

def simple_pca(X, k):
    # Center data
    X_centered = X - np.mean(X, axis=0)
    # Covariance matrix
    S = np.cov(X_centered, rowvar=False)
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(S)
    # Sort in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Select top k eigenvectors
    W = eigvecs[:, :k]
    Z = X_centered @ W
    return Z, W

Z, W = simple_pca(X, 2)

plt.figure()
plt.scatter(Z[:,0], Z[:,1], c='b', edgecolors='k')
plt.title("PCA: First Two Principal Components")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
