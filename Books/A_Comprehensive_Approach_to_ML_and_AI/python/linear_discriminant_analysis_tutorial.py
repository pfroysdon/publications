import numpy as np
import matplotlib.pyplot as plt

def lda(X, y):
    """
    Linear Discriminant Analysis (LDA).
    
    Parameters:
      X : n x d data matrix.
      y : n x 1 vector of class labels.
      
    Returns:
      W : Projection matrix (d x d), with eigenvectors sorted by descending eigenvalues.
      projectedData : Data projected onto the new subspace.
    """
    n, d = X.shape
    classes = np.unique(y)
    C = len(classes)
    
    # Compute overall mean
    mu = np.mean(X, axis=0)
    
    # Initialize scatter matrices
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))
    
    for cl in classes:
        X_i = X[y == cl]
        Ni = X_i.shape[0]
        mu_i = np.mean(X_i, axis=0)
        # Within-class scatter
        Sw += (X_i - mu_i).T @ (X_i - mu_i)
        # Between-class scatter
        diff = (mu_i - mu).reshape(-1, 1)
        Sb += Ni * (diff @ diff.T)
    
    # Solve the generalized eigenvalue problem: Sb * v = lambda * Sw * v
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    
    # Sort eigenvectors by descending eigenvalues
    idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, idx]
    
    # Project the data
    projectedData = X @ W
    return W, projectedData

# Generate synthetic data for two classes
np.random.seed(1)
N = 100
X1 = np.random.randn(N, 2) + 2
X2 = np.random.randn(N, 2) - 2
X = np.vstack([X1, X2])
y = np.concatenate([np.ones(N), 2 * np.ones(N)])

# Perform LDA
W, Z = lda(X, y)

# Plot original data with the first LDA direction
plt.figure(figsize=(12, 5))

# Subplot 1: Original data
plt.subplot(1, 2, 1)
plt.scatter(X1[:, 0], X1[:, 1], color='red', label='Class 1', alpha=0.7)
plt.scatter(X2[:, 0], X2[:, 1], color='blue', label='Class 2', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Data')
plt.grid(True)
# Draw a dashed line representing the first LDA direction
mu = np.mean(X, axis=0)
dirVec = W[:, 0]
t = np.linspace(-80, 80, 100)
linePoints = mu + np.outer(t, dirVec)
plt.plot(linePoints[:, 0], linePoints[:, 1], 'k--', linewidth=2)

# Subplot 2: Projected data (onto the first discriminant)
plt.subplot(1, 2, 2)
# Project data onto first discriminant axis
proj_class1 = Z[:N, 0]
proj_class2 = Z[N:, 0]
plt.scatter(proj_class1, np.zeros_like(proj_class1), color='red', label='Class 1', alpha=0.7)
plt.scatter(proj_class2, np.zeros_like(proj_class2), color='blue', label='Class 2', alpha=0.7)
plt.xlabel('Projection Value')
plt.title('Data Projected onto First Discriminant')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
