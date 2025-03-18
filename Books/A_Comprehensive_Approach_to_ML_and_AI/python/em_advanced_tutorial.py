#!/usr/bin/env python3
"""
emTutorial_advanced.py
----------------------
This tutorial demonstrates the EM algorithm for Gaussian mixture estimation.
It generates multidimensional data from several Gaussian components, runs EM,
and plots the log-likelihood vs. iterations as well as the estimated clusters.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

np.random.seed(2)

# Generate dataset from 2 Gaussian mixtures
nRand = 2
X = np.random.randn(500, 2) * 1 + 2
for _ in range(1, nRand):
    X = np.vstack((X, np.random.randn(500, 2) * 1 + 4.5))

def initialize_EM(X, k):
    n, d = X.shape
    kmeans = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X)
    M = kmeans.cluster_centers_.T  # d x k
    W = np.array([np.sum(kmeans.labels_==i)/n for i in range(k)])
    V = np.zeros((d, d, k))
    for i in range(k):
        cluster_points = X[kmeans.labels_==i]
        V[:,:,i] = np.cov(cluster_points, rowvar=False)
    return W, M, V

def expectation(X, k, W, M, V):
    n, d = X.shape
    E = np.zeros((n, k))
    for j in range(k):
        rv = multivariate_normal(mean=M[:,j], cov=V[:,:,j])
        E[:, j] = W[j] * rv.pdf(X)
    E = E / np.sum(E, axis=1, keepdims=True)
    return E

def maximization(X, k, E):
    n, d = X.shape
    W = np.sum(E, axis=0) / n
    M = (X.T @ E) / np.sum(E, axis=0)
    V = np.zeros((d, d, k))
    for j in range(k):
        diff = X - M[:, j]
        V[:,:,j] = (diff.T * E[:, j]) @ diff / np.sum(E[:, j])
    return W, M, V

def log_likelihood(X, k, W, M, V):
    n, d = X.shape
    L = 0
    for j in range(k):
        rv = multivariate_normal(mean=M[:,j], cov=V[:,:,j])
        L += W[j] * rv.pdf(X)
    return np.sum(np.log(L + 1e-10))

def EM_GM(X, k, ltol=1e-5, maxiter=500):
    W, M, V = initialize_EM(X, k)
    L_old = -np.inf
    L_history = []
    for i in range(maxiter):
        L_new = log_likelihood(X, k, W, M, V)
        L_history.append(L_new)
        if np.abs((L_new - L_old)/L_old) < ltol and i > 0:
            break
        L_old = L_new
        E = expectation(X, k, W, M, V)
        W, M, V = maximization(X, k, E)
    return W, M, V, np.array(L_history)

k = nRand
W, M, V, L_history = EM_GM(X, k, ltol=1e-5, maxiter=500)

# k-Means initialization for reference
kmeans = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X)
ctrs = kmeans.cluster_centers_

# Plot raw data and mixture
plt.figure()
plt.scatter(X[:,0], X[:,1], c='r', s=10)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Gaussian Mixture")
plt.show()

# Plot EM estimated Gaussians along with k-means centers
def plot_gaussian_mixture(X, W, M, V, ctrs):
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c='r', s=10)
    for j in range(W.shape[0]):
        mean = M[:,j]
        cov = V[:,:,j]
        # Draw an ellipse for each Gaussian
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
        width, height = 2*np.sqrt(eigvals)
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                          edgecolor='k', fc='None', lw=2)
        plt.gca().add_patch(ellipse)
    ctrs = np.array(ctrs)
    plt.scatter(ctrs[:,0], ctrs[:,1], c='b', marker='*', s=150, label="k-Means")
    plt.title("GMM estimated by EM (black ellipses) and k-Means (blue stars)")
    plt.xlabel("Dimension 1"); plt.ylabel("Dimension 2")
    plt.legend()
    plt.xlim([-2,8]); plt.ylim([-2,8])
    plt.show()

plot_gaussian_mixture(X, W, M, V, ctrs)

plt.figure()
plt.plot(L_history, marker='*')
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.title("Likelihood vs. Iteration")
plt.grid(True)
plt.show()
