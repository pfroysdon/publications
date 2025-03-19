# SVD Tutorial in R: PCA for Image Compression
#
# This tutorial reads a color image, reshapes it into a 2D matrix where each row is a pixel's RGB vector,
# performs PCA via eigen decomposition on the covariance matrix, and reconstructs the image using a reduced number of components.

library(jpeg)
library(ggplot2)

# Read the image (ensure the file exists in the specified path)
img <- readJPEG("data/arizona_photo.jpg")
# Convert image (an array of dimensions height x width x 3) to a 2D matrix
dim_img <- dim(img)
X <- matrix(as.vector(img), ncol = 3, byrow = TRUE)  # Each row is an RGB vector
N <- nrow(X)

# Optional: Plot a sample of pixels in RGB space
sample_idx <- seq(1, N, by = 100)
plot3d <- function(data) {
  library(scatterplot3d)
  scatterplot3d(data[,1], data[,2], data[,3], pch = 16, color = rgb(data[,1], data[,2], data[,3]))
}
plot3d(X[sample_idx, ])

# Perform PCA on the RGB vectors
mu_x <- colMeans(X)
X_centered <- sweep(X, 2, mu_x)
C_x <- cov(X_centered)
eig <- eigen(C_x)
# Eigenvectors (columns of V) and eigenvalues
V <- eig$vectors
D <- eig$values

# For demonstration, project onto the top 1 principal component
k <- 1
A_k <- t(V[, order(D, decreasing = TRUE)[1:k]])
Y <- A_k %*% t(X_centered)  # Resulting in a (k x N) matrix

# Reconstruct the image using the top k components
X_recon <- t(t(A_k) %*% Y) + matrix(rep(mu_x, each = N), ncol = 3, byrow = TRUE)
X_recon <- pmin(pmax(X_recon, 0), 1)  # Ensure values are between 0 and 1

# Reshape reconstructed data to original image dimensions
img_recon <- array(X_recon, dim = dim_img)

# Plot original and reconstructed images
par(mfrow = c(1,2))
plot(as.raster(img), main = "Original Image")
plot(as.raster(img_recon), main = paste("Reconstructed Image (k =", k, ")"))
