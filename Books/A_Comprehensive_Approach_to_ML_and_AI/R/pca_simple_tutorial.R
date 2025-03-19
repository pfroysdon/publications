# PCA Simple Tutorial in R
#
# This demo uses Principal Component Analysis (PCA) to compute the principal components of the data.
# The data is first centered, the covariance matrix is computed, eigen decomposition is performed,
# and the results are visualized by plotting the data with overlaid eigenvectors and by projecting onto the principal components.

# For reproducibility
set.seed(0)

# -----------------------------
# Generate sample data
# -----------------------------
N <- 50
x_in <- cbind(rnorm(N) + 1, 0.5 * rnorm(N) + 0.5)
theta <- -0.707
R_mat <- matrix(c(cos(theta), sin(theta), -sin(theta), cos(theta)), nrow = 2)
x_in <- x_in %*% t(R_mat)

# Step 1: Calculate mean of input vectors
mu_x <- colMeans(x_in)

# Step 2: Center the data
x_bar <- sweep(x_in, 2, mu_x)

# Step 3: Calculate covariance matrix
C_x <- cov(x_bar)
cat("Covariance of input:\n"); print(C_x)

# Step 4: Eigen decomposition of covariance matrix
eig <- eigen(C_x)
V <- eig$vectors
D <- eig$values
e1 <- V[,1]
e2 <- V[,2]
cat("Eigenvector e1:\n"); print(e1)
cat("Eigenvector e2:\n"); print(e2)
cat("Eigenvalue d1:\n"); print(D[1])
cat("Eigenvalue d2:\n"); print(D[2])

# Plot centered data with mean marked
plot(x_in[,1], x_in[,2], pch = 16, main = "Centered Input Vectors", xlab = "X1", ylab = "X2", xlim = c(-3,3), ylim = c(-3,3))
points(mu_x[1], mu_x[2], col = "red", pch = 4, lwd = 2)

# Plot centered data with eigenvectors
plot(x_bar[,1], x_bar[,2], pch = 16, col = "blue", main = "Centered Data with Eigenvectors", xlab = "X1", ylab = "X2", xlim = c(-3,3), ylim = c(-3,3))
arrows(0, 0, 2 * sqrt(D[1]) * e1[1], 2 * sqrt(D[1]) * e1[2], col = "red", lwd = 2)
arrows(0, 0, 2 * sqrt(D[2]) * e2[1], 2 * sqrt(D[2]) * e2[2], col = "red", lwd = 2)

# Project data onto principal components
y_proj <- t(V) %*% t(x_bar)
plot(y_proj[1, ], y_proj[2, ], pch = 16, main = "Projection onto Principal Components",
     xlab = "PC1", ylab = "PC2", xlim = c(-3,3), ylim = c(-3,3))

# Project using only one principal component (e.g., PC1)
y_pc1 <- y_proj[1, ]
plot(y_pc1, rep(0, length(y_pc1)), pch = 16, main = "Projection onto PC1", xlab = "PC1", ylab = "", xlim = c(-3,3))
