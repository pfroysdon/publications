#!/usr/bin/env python3
"""
feature_engineering_tutorial.py
-------------------------------
This tutorial demonstrates a complete feature engineering pipeline for binary classification.
Steps:
  1. Data Cleaning: Mean imputation for missing values and removal of outliers.
  2. Feature Transformation: z-score normalization.
  3. Feature Extraction: PCA.
  4. Feature Derivation: Creating polynomial features (degree 2).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(1)

# 1. Generate Synthetic Dataset
N = 200
# Class 0: centered at (1,1)
X_class0 = np.random.randn(2, N//2) * 0.5 + np.array([[1], [1]])
# Class 1: centered at (3,3)
X_class1 = np.random.randn(2, N//2) * 0.5 + np.array([[3], [3]])
X = np.hstack((X_class0, X_class1))  # 2 x N
y = np.concatenate((np.zeros(N//2), np.ones(N//2)))

# Introduce missing values: set 5% of entries to np.nan
num_missing = int(0.05 * X.size)
missing_indices = np.random.choice(X.size, num_missing, replace=False)
X_flat = X.flatten()
X_flat[missing_indices] = np.nan
X = X_flat.reshape(X.shape)

# Introduce anomalies: multiply feature 1 of 5 random samples by 10
anomaly_indices = np.random.choice(N, 5, replace=False)
X[0, anomaly_indices] = X[0, anomaly_indices] * 10

# 2. Data Cleaning
def impute_missing(X):
    X_imputed = X.copy()
    for j in range(X.shape[0]):
        col = X_imputed[j, :]
        missing = np.isnan(col)
        if np.any(missing):
            col_mean = np.nanmean(col)
            col[missing] = col_mean
            X_imputed[j, :] = col
    return X_imputed

def remove_outliers(X, y, threshold):
    d, N = X.shape
    mu = np.mean(X, axis=1)
    sigma = np.std(X, axis=1)
    X_clean = []
    y_clean = []
    for i in range(N):
        sample = X[:, i]
        if np.all(np.abs((sample - mu)/sigma) < threshold):
            X_clean.append(sample)
            y_clean.append(y[i])
    return np.array(X_clean).T, np.array(y_clean)

X_clean = impute_missing(X)
X_clean, y_clean = remove_outliers(X_clean, y, threshold=2)

# 3. Feature Transformation (z-score normalization)
def zscore_normalize(X):
    mu = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, keepdims=True)
    return (X - mu) / sigma, mu, sigma

X_norm, mu_X, sigma_X = zscore_normalize(X_clean)

# 4. Feature Extraction (PCA)
# Transpose X_norm so that observations are rows
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm.T).T  # 2 x samples

# 5. Feature Derivation: polynomial features (degree 2)
def create_polynomial_features(X, degree):
    # X is d x N; for degree 2, we add squares and pairwise products.
    d, N = X.shape
    X_poly = X.copy()
    if degree >= 2:
        # Square terms
        for i in range(d):
            X_poly = np.vstack((X_poly, X[i, :]**2))
        # Pairwise products
        for i in range(d):
            for j in range(i+1, d):
                X_poly = np.vstack((X_poly, X[i, :] * X[j, :]))
    return X_poly

X_poly = create_polynomial_features(X_norm, 2)

# 6. Visualization
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(X[0, :], X[1, :], c=y, cmap='coolwarm', edgecolors='k')
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.title("Original Data")
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(X_clean[0, :], X_clean[1, :], c=y_clean, cmap='coolwarm', edgecolors='k')
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.title("Cleaned Data")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(X_pca[0, :], X_pca[1, :], c=y_clean, cmap='coolwarm', edgecolors='k')
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA Extracted Features")
plt.grid(True)

plt.subplot(1,2,2)
# For visualization, plot the first two rows of polynomial features.
plt.scatter(X_poly[0, :], X_poly[1, :], c=y_clean, cmap='coolwarm', edgecolors='k')
plt.xlabel("Poly Feature 1"); plt.ylabel("Poly Feature 2")
plt.title("Polynomial Derived Features")
plt.grid(True)
plt.tight_layout()
plt.show()
