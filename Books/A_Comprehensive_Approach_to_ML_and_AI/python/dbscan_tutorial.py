#!/usr/bin/env python3
"""
dbscan_tutorial.py
------------------
This script demonstrates a simple implementation of DBSCAN.
It generates synthetic 2D data, runs DBSCAN, and visualizes the clusters.
"""

import numpy as np
import matplotlib.pyplot as plt

def region_query(X, idx, eps):
    point = X[idx, :]
    distances = np.sqrt(np.sum((X - point)**2, axis=1))
    return np.where(distances <= eps)[0]

def dbscan_clustering(X, eps, minPts):
    n = X.shape[0]
    labels = np.zeros(n, dtype=int)  # 0 means unassigned; noise will be -1
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    for i in range(n):
        if not visited[i]:
            visited[i] = True
            neighbors = region_query(X, i, eps)
            if len(neighbors) < minPts:
                labels[i] = -1  # noise
            else:
                cluster_id += 1
                labels[i] = cluster_id
                seed_set = list(neighbors)
                k = 0
                while k < len(seed_set):
                    j = seed_set[k]
                    if not visited[j]:
                        visited[j] = True
                        neighbors_j = region_query(X, j, eps)
                        if len(neighbors_j) >= minPts:
                            seed_set.extend(neighbors_j.tolist())
                    if labels[j] == 0:
                        labels[j] = cluster_id
                    k += 1
    return labels

# Generate synthetic data: two clusters
np.random.seed(0)
X1 = np.random.randn(50,2)
X2 = np.random.randn(50,2) + 3
X = np.vstack((X1, X2))

eps = 0.8
minPts = 5
labels = dbscan_clustering(X, eps, minPts)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=labels, cmap='jet', edgecolor='k')
plt.title(f"DBSCAN Clustering (eps={eps}, minPts={minPts})")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()

numClusters = len(np.unique(labels[labels > 0]))
noise_count = np.sum(labels == -1)
print(f"Number of clusters found: {numClusters}")
print(f"Number of noise points: {noise_count}")
