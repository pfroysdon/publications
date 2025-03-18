import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.randn(100, 3)  # 100x3 feature matrix
Y = np.random.randint(0, 2, (100, 1))  # Binary labels

train_ratio = 0.8
N = X.shape[0]
train_size = int(np.floor(train_ratio * N))

# Shuffle data.
indices = np.random.permutation(N)
X_shuffled = X[indices]
Y_shuffled = Y[indices]

# Split data.
X_train = X_shuffled[:train_size]
Y_train = Y_shuffled[:train_size]
X_test = X_shuffled[train_size:]
Y_test = Y_shuffled[train_size:]

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Plot 3D data.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c='b', marker='o', label='Train Data')
ax.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c='r', marker='^', label='Test Data')
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.set_title("Train-Test Split Visualization")
ax.legend()
plt.show()
