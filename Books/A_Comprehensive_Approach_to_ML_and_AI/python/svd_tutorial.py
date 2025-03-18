import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io

# 1. Read the image (ensure the image file is in the 'data' folder)
RGB = io.imread('data/arizona_photo.jpg') / 255.0  # convert to [0,1] float

# 2. Reshape the image: each row is a pixel [R, G, B]
height, width, channels = RGB.shape
X = RGB.reshape(-1, 3)  # shape (N, 3)
N = X.shape[0]

# 3. Plot pixels in RGB space (plot every 100th pixel)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(0, N, 100):
    ax.scatter(X[i, 0], X[i, 1], X[i, 2], color=X[i, :], s=20)
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.set_xlim([0,1]); ax.set_ylim([0,1]); ax.set_zlim([0,1])
plt.title('RGB Color Space Scatter Plot')
plt.show()

# 4. Perform PCA on color vectors
m_x = np.mean(X, axis=0)
X_centered = X - m_x
C_x = np.cov(X_centered, rowvar=False)

# Compute eigenvalues and eigenvectors.
eigvals, eigvecs = np.linalg.eig(C_x)
# Sort eigenvalues in ascending order and reorder eigenvectors.
sorted_idx = np.argsort(eigvals)
e1 = eigvecs[:, sorted_idx[-1]]
e2 = eigvecs[:, sorted_idx[-2]]
e3 = eigvecs[:, sorted_idx[-3]]
d1 = eigvals[sorted_idx[-1]]
d2 = eigvals[sorted_idx[-2]]
d3 = eigvals[sorted_idx[-3]]
print("Eigenvector e1:", e1)
print("Eigenvector e2:", e2)
print("Eigenvector e3:", e3)
print("Eigenvalue d1:", d1)
print("Eigenvalue d2:", d2)
print("Eigenvalue d3:", d3)

# Construct transformation matrix A: rows are eigenvectors ordered by descending eigenvalue.
A = np.vstack([e1, e2, e3])

# 5. Project pixels: Y = A * (x - m_x) for each pixel.
Y = (A @ X_centered.T)  # shape (3, N)

# Display the projections as images.
Y1 = Y[0, :].reshape(height, width)
Y2 = Y[1, :].reshape(height, width)
Y3 = Y[2, :].reshape(height, width)
plt.figure(figsize=(15,4))
plt.subplot(1,3,1); plt.imshow(Y1, cmap='gray'); plt.title('Projection on PC1')
plt.subplot(1,3,2); plt.imshow(Y2, cmap='gray'); plt.title('Projection on PC2')
plt.subplot(1,3,3); plt.imshow(Y3, cmap='gray'); plt.title('Projection on PC3')
plt.show()

# 6. Reconstruct image using only the first principal component.
k = 1
A_k = A[:k, :]  # (1, 3)
# Project and reconstruct: X_r = (A_k^T * Y_k) + m_x, where Y_k = A_k*(X - m_x)
Y_k = A_k @ X_centered.T   # shape (1, N)
X_r = (A_k.T @ Y_k).T + m_x  # shape (N, 3)
X_r = np.clip(X_r, 0, 1)
# Reshape back to image.
reconstructed = X_r.reshape(height, width, 3)

# Plot color-space scatter of reconstructed pixels.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(0, N, 100):
    ax.scatter(X_r[i, 0], X_r[i, 1], X_r[i, 2], color=X_r[i, :], s=20)
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.set_xlim([0,1]); ax.set_ylim([0,1]); ax.set_zlim([0,1])
plt.title('Reconstructed Color Space Scatter (using PC1)')
plt.show()

# Display original and reconstructed images.
plt.figure()
plt.subplot(1,2,1); plt.imshow(RGB); plt.title('Original Image')
plt.subplot(1,2,2); plt.imshow(reconstructed); plt.title('Reconstructed Image (PC1)')
plt.show()
