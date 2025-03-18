import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def k_means_fast(X, K, max_iters=100):
    """
    Optimized K-Means Clustering Algorithm.

    Args:
        X (np.ndarray): Data matrix of shape (N, D), where N is the number
                        of points and D is the feature dimension.
        K (int): Number of clusters.
        max_iters (int, optional): Maximum number of iterations. Default is 100.

    Returns:
        centroids (np.ndarray): Final cluster centroids of shape (K, D).
        labels (np.ndarray): Cluster assignment for each point (length N).
    """
    N, D = X.shape

    # For reproducibility (similar to MATLAB's rng('default'))
    np.random.seed(0)
    # Initialize K centroids by selecting K random points from X
    indices = np.random.choice(N, K, replace=False)
    centroids = X[indices, :].copy()

    labels = np.zeros(N, dtype=int)
    prev_centroids = centroids.copy()

    for iter in range(max_iters):
        # Compute the Euclidean distances from each point to each centroid
        distances = cdist(X, centroids, metric='euclidean')
        # Assign each point to the nearest centroid
        labels = np.argmin(distances, axis=1)

        # Update centroids by computing the mean of points assigned to each cluster
        for k in range(K):
            if np.any(labels == k):
                centroids[k, :] = np.mean(X[labels == k, :], axis=0)
            else:
                # If a cluster is empty, reinitialize its centroid to a random point
                centroids[k, :] = X[np.random.randint(N), :]

        # Check for convergence (if centroids change less than a small threshold)
        if np.max(np.abs(centroids - prev_centroids)) < 1e-6:
            break
        prev_centroids = centroids.copy()

    return centroids, labels

def k_means(X, K, max_iters=100):
    """
    K-Means Clustering Algorithm using loops.

    Args:
        X (np.ndarray): Data matrix of shape (N, D).
        K (int): Number of clusters.
        max_iters (int, optional): Maximum number of iterations. Default is 100.

    Returns:
        centroids (np.ndarray): Final cluster centroids.
        labels (np.ndarray): Cluster assignments for each point.
    """
    N, D = X.shape

    # For reproducibility
    np.random.seed(0)
    indices = np.random.choice(N, K, replace=False)
    centroids = X[indices, :].copy()
    labels = np.zeros(N, dtype=int)
    prev_centroids = centroids.copy()

    for iter in range(max_iters):
        # Assign each data point to the nearest centroid (non-vectorized)
        for i in range(N):
            distances = np.sum((centroids - X[i, :])**2, axis=1)
            labels[i] = np.argmin(distances)

        # Compute new centroids as the mean of assigned points
        for k in range(K):
            if np.sum(labels == k) == 0:
                centroids[k, :] = X[np.random.randint(N), :]
            else:
                centroids[k, :] = np.mean(X[labels == k, :], axis=0)

        # Check for convergence
        if np.max(np.abs(centroids - prev_centroids)) < 1e-6:
            break
        prev_centroids = centroids.copy()

    return centroids, labels

if __name__ == '__main__':
    # Generate synthetic data: two Gaussian clusters
    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) + 2  # Cluster centered at +2
    cluster2 = np.random.randn(50, 2) - 2  # Cluster centered at -2
    X = np.vstack((cluster1, cluster2))

    # Set number of clusters
    K = 2

    # Run K-means using the fast (vectorized) version
    centroids, labels = k_means_fast(X, K)

    # ----------------------- Plot 1 -----------------------
    # Plot data points colored by cluster (all in blue, as in MATLAB 'b')
    plt.figure()
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = (labels == label)
        # Plot each cluster with blue markers.
        # To mimic MATLAB's gscatter with a single color,
        # we label only once.
        plt.scatter(X[idx, 0], X[idx, 1], c='b', marker='o')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(['Clusters'], loc='lower right')  # 'SE' in MATLAB ~ 'lower right'
    plt.grid(True)
    plt.show()

    # ----------------------- Plot 2 -----------------------
    # Plot data points with different colors and overlay centroids.
    plt.figure()
    colors = ['r', 'b']  # Assign red to cluster 1 and blue to cluster 2
    for label in unique_labels:
        idx = (labels == label)
        plt.scatter(X[idx, 0], X[idx, 1], c=colors[label], marker='o', label=f'Cluster {label+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='k', marker='x', linewidths=2, label='Centroids')
    plt.title('K-Means Clustering - Centroids Identified')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
