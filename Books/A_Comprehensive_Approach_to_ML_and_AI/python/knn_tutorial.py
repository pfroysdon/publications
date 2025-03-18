import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset: Two Gaussian clusters
X = np.vstack([np.random.randn(50, 2) + 2, np.random.randn(50, 2) - 2])
Y = np.concatenate([np.ones(50), np.zeros(50)])  # Class labels: 1 and 0

# Split data into training and testing sets
train_ratio = 0.8
train_size = int(np.floor(train_ratio * X.shape[0]))
X_train = X[:train_size, :]
Y_train = Y[:train_size]
X_test = X[train_size:, :]
Y_test = Y[train_size:]

def knn_classify_fast(X_train, Y_train, X_test, K):
    """Optimized KNN using Euclidean distance and majority voting."""
    num_test = X_test.shape[0]
    Y_pred = np.zeros(num_test)
    for i in range(num_test):
        # Compute Euclidean distances from X_test[i] to all training samples
        distances = np.linalg.norm(X_train - X_test[i, :], axis=1)
        # Find the indices of the K nearest neighbors
        idx = np.argsort(distances)[:K]
        nearest_labels = Y_train[idx]
        # Use mode to get the majority vote (SciPy returns a ModeResult)
        Y_pred[i] = mode(nearest_labels).mode[0]
    return Y_pred

# Perform KNN classification
K = 5
Y_pred = knn_classify_fast(X_train, Y_train, X_test, K)
accuracy = np.mean(Y_pred == Y_test) * 100
print("KNN Accuracy: {:.2f}%".format(accuracy))

# Create a mesh grid for visualization of decision boundaries
x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
x1_grid, x2_grid = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                               np.linspace(x2_min, x2_max, 100))
X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]

# Predict on the grid
Y_grid = knn_classify_fast(X_train, Y_train, X_grid, K)
Y_grid = Y_grid.reshape(x1_grid.shape)

# Plot training data without decision boundary (similar to first figure)
plt.figure()
plt.scatter(X_train[Y_train==1, 0], X_train[Y_train==1, 1],
            color='blue', label='Class 1')
plt.scatter(X_train[Y_train==0, 0], X_train[Y_train==0, 1],
            color='red', label='Class 0')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Clustering')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("results/knn1.png")
plt.show()

# Plot decision boundary using contourf
plt.figure()
plt.scatter(X_train[Y_train==1, 0], X_train[Y_train==1, 1],
            color='blue', label='Class 1')
plt.scatter(X_train[Y_train==0, 0], X_train[Y_train==0, 1],
            color='red', label='Class 0')
plt.contourf(x1_grid, x2_grid, Y_grid, alpha=0.1, linewidths=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Clustering - Decision Boundary')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("results/knn2.png")
plt.show()
