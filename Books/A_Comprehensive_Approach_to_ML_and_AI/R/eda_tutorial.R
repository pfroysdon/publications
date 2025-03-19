# eda_tutorial.R
# This script performs PCA on a toy dataset (200 samples, 5 features) and plots the first two principal components.

rm(list=ls())
graphics.off()

X <- matrix(rnorm(200 * 5), nrow = 200, ncol = 5)
k <- 2  # Reduce to 2 dimensions

simplePCA <- function(X, k) {
  X_mean <- colMeans(X)
  X_centered <- sweep(X, 2, X_mean)
  S <- cov(X_centered)
  eig_res <- eigen(S)
  idx <- order(eig_res$values, decreasing = TRUE)
  W <- eig_res$vectors[, idx[1:k]]
  Z <- X_centered %*% W
  list(Z = Z, W = W)
}

pca_res <- simplePCA(X, k)
Z <- pca_res$Z

plot(Z[, 1], Z[, 2], pch = 16, col = "blue", 
     xlab = "Principal Component 1", ylab = "Principal Component 2",
     main = "PCA: First Two Principal Components")
grid()
