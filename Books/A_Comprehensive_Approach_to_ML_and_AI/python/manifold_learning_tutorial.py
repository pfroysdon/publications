import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

def myLLE(X, K, d):
    # X: (N x D) data matrix, K: number of neighbors, d: target dimension.
    N = X.shape[0]
    dist = squareform(pdist(X))
    
    W = np.zeros((N, N))
    for i in range(N):
        # Get indices of K nearest neighbors (excluding self)
        idx = np.argsort(dist[i, :])
        neighbors = idx[1:K+1]
        # Compute local covariance matrix
        Z = (X[i, :] - X[neighbors, :]).T  # shape (D, K)
        C = Z.T @ Z
        # Regularization for numerical stability
        C += np.eye(K) * 1e-3 * np.trace(C)
        # Solve for weights
        w = np.linalg.solve(C, np.ones((K, 1)))
        w = w / np.sum(w)
        W[i, neighbors] = w.flatten()
    
    M = (np.eye(N) - W).T @ (np.eye(N) - W)
    # Compute eigen-decomposition (use eigh for symmetric matrices)
    eigvals, eigvecs = eigh(M)
    # Sort eigenvectors by ascending eigenvalues
    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]
    # Discard the first eigenvector (zero eigenvalue)
    Y = eigvecs[:, 1:d+1]
    
    if d >= 2:
        plt.figure()
        plt.scatter(Y[:, 0], Y[:, 1], c='b', s=25)
        plt.title("LLE Embedding (First Two Dimensions)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.show()
    return Y

def generate_swiss_roll(N=1000):
    # Generate a Swiss Roll dataset.
    t = (3*np.pi/2) * (1 + 2 * np.random.rand(N))
    h = 21 * np.random.rand(N)
    X = np.zeros((N, 3))
    X[:, 0] = t * np.cos(t)
    X[:, 1] = h
    X[:, 2] = t * np.sin(t)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap='viridis', s=20)
    ax.view_init(elev=6, azim=-17)
    plt.title("Swiss Roll Dataset")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.colorbar(sc)
    plt.show()
    
    return X, t

# Main script for manifold learning
if __name__ == '__main__':
    X, t_param = generate_swiss_roll(1000)
    # Run LLE with K=12 neighbors and target dimension 2
    Y_embed = myLLE(X, K=12, d=2)
