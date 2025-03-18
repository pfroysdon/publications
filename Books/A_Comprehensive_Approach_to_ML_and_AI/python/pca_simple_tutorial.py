import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
N = 50
# Generate sample data
x_in = np.zeros((N, 2))
x_in[:, 0] = np.random.randn(N) + 1
x_in[:, 1] = 0.5 * np.random.randn(N) + 0.5

# Rotation matrix
theta = -0.707
R = np.array([[np.cos(theta), np.sin(theta)],
              [-np.sin(theta), np.cos(theta)]])
x_in = x_in @ R.T  # Rotate data

# Step 1: Calculate the mean and center the data
mu_x = np.mean(x_in, axis=0)
x_bar = x_in - mu_x

# Step 2: Calculate covariance matrix
C_x = np.cov(x_bar, rowvar=False)
print("Covariance of input:")
print(C_x)

# Step 3: Eigen-decomposition
eigvals, eigvecs = np.linalg.eig(C_x)
# Sort eigenvalues (and corresponding eigenvectors)
sorted_idx = np.argsort(eigvals)
d1, d2 = eigvals[sorted_idx[0]], eigvals[sorted_idx[1]]
e1 = eigvecs[:, sorted_idx[0]]
e2 = eigvecs[:, sorted_idx[1]]
print("Eigenvector e1:")
print(e1)
print("Eigenvector e2:")
print(e2)
print("Eigenvalue d1:")
print(d1)
print("Eigenvalue d2:")
print(d2)

# Plot centered input vectors
plt.figure()
plt.plot(x_in[:,0], x_in[:,1], 'o')
plt.plot(mu_x[0], mu_x[1], 'r+', markersize=12, markeredgewidth=2)
plt.title("Centered Input Vectors")
plt.axis("equal")
plt.xlim([-3,3]); plt.ylim([-3,3])
plt.show()

# Plot centered data with eigenvectors
plt.figure()
plt.plot(x_bar[:,0], x_bar[:,1], 'bo')
origin = np.array([[0, 0],[0, 0]])
plt.quiver(0, 0, 2*d1*e1[0], 2*d1*e1[1], color='r', angles='xy', scale_units='xy', scale=1, width=0.005)
plt.quiver(0, 0, 2*d2*e2[0], 2*d2*e2[1], color='r', angles='xy', scale_units='xy', scale=1, width=0.005)
plt.title("Centered Input Vectors with Eigenvectors")
plt.axis("equal")
plt.xlim([-3,3]); plt.ylim([-3,3])
plt.show()

# Step 4: Project input data onto principal components
# Compute projections: note that eigenvectors form the new basis.
y = np.vstack([e2, e1]) @ x_bar.T  # shape (2, N)

plt.figure()
plt.plot(y[0, :], y[1, :], 'o')
plt.title("Projections onto Principal Components")
plt.axis("equal")
plt.xlim([-3,3]); plt.ylim([-3,3])
plt.show()

# Step 5: Project data using only one principal component
y1 = e1.T @ x_bar.T  # projection onto first principal component
plt.figure()
plt.plot(y1, np.zeros_like(y1), 'o')
plt.title("Projections onto One Principal Component (e1)")
plt.axis("equal")
plt.xlim([-3,3]); plt.ylim([-3,3])
plt.show()

y2 = e2.T @ x_bar.T  # projection onto second principal component
plt.figure()
plt.plot(y2, np.zeros_like(y2), 'o')
plt.title("Projections onto One Principal Component (e2)")
plt.axis("equal")
plt.xlim([-3,3]); plt.ylim([-3,3])
plt.show()
