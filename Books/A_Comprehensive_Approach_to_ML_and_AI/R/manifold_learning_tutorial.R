# Manifold Learning Tutorial in R using LLE
#
# This script generates a Swiss Roll dataset and applies Locally Linear Embedding (LLE)
# to reduce its dimensionality.

# Function to perform LLE
myLLE <- function(X, K, d) {
  # X: Data matrix (N x D) where rows are samples
  # K: Number of nearest neighbors
  # d: Target dimensionality
  N <- nrow(X)
  
  # Compute pairwise distances
  distMat <- as.matrix(dist(X))
  
  # Initialize weight matrix W (N x N)
  W <- matrix(0, nrow = N, ncol = N)
  
  for (i in 1:N) {
    # Find indices of K nearest neighbors (excluding self)
    idx <- order(distMat[i, ])[2:(K + 1)]
    # Compute local covariance matrix
    Z <- t(X[i, ] - X[idx, , drop = FALSE])  # D x K
    C <- t(Z) %*% Z  # K x K
    # Regularization for numerical stability
    C <- C + diag(rep(1e-3 * sum(diag(C)), K))
    # Solve for reconstruction weights
    w <- solve(C, rep(1, K))
    w <- w / sum(w)
    W[i, idx] <- w
  }
  
  # Compute matrix M = (I - W)' %*% (I - W)
  M <- t(diag(N) - W) %*% (diag(N) - W)
  
  # Eigen decomposition
  eig <- eigen(M, symmetric = TRUE)
  eigvec <- eig$vectors
  # Discard the first eigenvector (smallest eigenvalue)
  Y_embed <- eigvec[, 2:(d + 1)]
  
  # Visualization if target dimension >= 2
  if (d >= 2) {
    plot(Y_embed[, 1], Y_embed[, 2], pch = 16,
         main = "LLE Embedding (First Two Dimensions)",
         xlab = "Dimension 1", ylab = "Dimension 2")
    grid()
  }
  return(Y_embed)
}

# Function to generate Swiss Roll dataset
generateSwissRoll <- function(N = 1000) {
  # Generate parameter t uniformly in [1.5*pi, 4.5*pi]
  t <- runif(N, min = 1.5 * pi, max = 4.5 * pi)
  # Generate height component uniformly in [0, 21]
  h <- runif(N, min = 0, max = 21)
  X <- matrix(0, nrow = N, ncol = 3)
  X[, 1] <- t * cos(t)  # X-coordinate
  X[, 2] <- h           # Y-coordinate (height)
  X[, 3] <- t * sin(t)  # Z-coordinate
  
  # 3D scatter plot using scatterplot3d
  if (!requireNamespace("scatterplot3d", quietly = TRUE))
    install.packages("scatterplot3d")
  library(scatterplot3d)
  scatterplot3d(X, color = rainbow(N)[rank(t)], pch = 16,
                main = "Swiss Roll Dataset", xlab = "X", ylab = "Y", zlab = "Z")
  return(list(X = X, t = t))
}

# Generate Swiss Roll data and apply LLE
swiss <- generateSwissRoll(1000)
X_swiss <- swiss$X  # 1000 x 3 matrix
# Apply LLE with K = 12 nearest neighbors and target dimension 2
Y_lle <- myLLE(X_swiss, K = 12, d = 2)
