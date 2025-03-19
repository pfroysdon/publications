# em_advanced_tutorial.R
# Expectation & Maximization demo
#
# Description:
#   EM algorithm for k multidimensional Gaussian mixture estimation.
#
# Notes:
#   See subfunctions for descriptions.
#
# -------------------------------------------------------------------------
# Main Script

rm(list = ls())
graphics.off()
set.seed(2)
library(mvtnorm)  # For dmvnorm

## Generate dataset
# Generate random variables & mixtures
nRand <- 2
X <- matrix(rnorm(500 * 2, mean = 2, sd = 1), ncol = 2)
for (ii in 2:nRand) {
  X <- rbind(X, matrix(rnorm(500 * 2, mean = 4.5, sd = 1), ncol = 2))
}

## Run EM algorithm for Gaussian Mixture estimation
# EM_GM returns estimated weights (W), means (M), covariances (V), and log-likelihood history (L)
EM_GM <- function(X, k, ltol = 1e-5, maxiter = 500, fflag = 1, Init = NULL) {
  t_start <- proc.time()[3]
  if (is.null(Init)) {
    init_res <- init_EM(X, k)
    W <- init_res$W; M <- init_res$M; V <- init_res$V; L <- 0
  } else {
    W <- Init$W; M <- Init$M; V <- Init$V
  }
  Ln <- Likelihood(X, k, W, M, V)
  Lo <- 2 * Ln
  niter <- 0
  L_hist <- c()
  while ((abs(100 * (Ln - Lo) / Lo) > ltol) && (niter <= maxiter)) {
    # E-step: Compute responsibilities
    E <- Expectation(X, k, W, M, V, fflag)
    # M-step: Update parameters
    mm <- Maximization(X, k, E, fflag)
    W <- mm$W; M <- mm$M; V <- mm$V
    Lo <- Ln
    Ln <- Likelihood(X, k, W, M, V)
    niter <- niter + 1
    L_hist[niter] <- Ln
  }
  cat(sprintf("CPU time used for EM_GM: %.2fs\n", proc.time()[3] - t_start))
  cat(sprintf("Number of iterations: %d\n", niter))
  list(W = W, M = M, V = V, L = L_hist)
}

## Subfunction: init_EM
init_EM <- function(X, k) {
  n <- nrow(X); d <- ncol(X)
  # Use kmeans for initialization
  km <- kmeans(X, centers = k, nstart = 5)
  M <- t(km$centers)  # d x k
  # Initialize structure for covariance calculation
  V <- array(0, dim = c(d, d, k))
  W <- rep(0, k)
  for (i in 1:k) {
    idx <- which(km$cluster == i)
    W[i] <- length(idx) / n
    V[,,i] <- cov(X[idx, , drop = FALSE])
  }
  list(W = W, M = M, V = V)
}

## Subfunction: Expectation using vectorization
Expectation <- function(X, k, W, M, V, fflag) {
  n <- nrow(X)
  E <- matrix(0, n, k)
  if (fflag == 1) {
    for (j in 1:k) {
      # Use dmvnorm for multivariate normal pdf
      E[, j] <- W[j] * dmvnorm(X, mean = as.vector(M[, j]), sigma = V[,,j])
    }
    E <- E / rowSums(E)
  } else {
    # Loop version (slower)
    for (i in 1:n) {
      for (j in 1:k) {
        E[i, j] <- W[j] * dmvnorm(X[i, ], mean = as.vector(M[, j]), sigma = V[,,j])
      }
      E[i, ] <- E[i, ] / sum(E[i, ])
    }
  }
  E
}

## Subfunction: Maximization
Maximization <- function(X, k, E, fflag) {
  n <- nrow(X); d <- ncol(X)
  if (fflag == 1) {
    W_new <- colSums(E)
    M_new <- t(t(X) %*% E) / matrix(W_new, nrow = d, ncol = k, byrow = TRUE)
    V_new <- array(0, dim = c(d, d, k))
    for (i in 1:k) {
      X_centered <- X - matrix(rep(M_new[, i], n), nrow = n, byrow = TRUE)
      Wsp <- diag(E[, i])
      V_new[,,i] <- t(X_centered) %*% Wsp %*% X_centered / W_new[i]
    }
    W_new <- W_new / n
  } else {
    # Loop version (omitted for brevity)
    stop("Non-vectorized Maximization not implemented")
  }
  list(W = W_new, M = M_new, V = V_new)
}

## Subfunction: Likelihood
Likelihood <- function(X, k, W, M, V) {
  n <- nrow(X); d <- ncol(X)
  U <- colMeans(X)
  S <- cov(X)
  L_val <- 0
  for (i in 1:k) {
    iV <- solve(V[,,i])
    ll_1 <- -0.5 * n * log(det(2 * pi * V[,,i]))
    ll_2 <- -0.5 * (n - 1) * (sum(diag(iV %*% S)) + t(U - M[, i]) %*% iV %*% (U - M[, i]))
    L_val <- L_val + W[i] * (ll_1 + ll_2)
  }
  L_val
}

## Subfunction: Plot_GM (for 2D data)
Plot_GM <- function(W, M, V, X) {
  d <- nrow(M); k <- ncol(M)
  if (d == 1) {
    R <- seq(min(X), max(X), length.out = 1000)
    Q <- rep(0, length(R))
    for (i in 1:k) {
      P <- W[i] * dnorm(R, mean = M[, i], sd = sqrt(V[,,i]))
      Q <- Q + P
      lines(R, P, col = "red", lwd = 2)
    }
    lines(R, Q, col = "black", lwd = 2)
    xlabel("X"); ylabel("Probability density")
    title("Gaussian Mixture estimated by EM")
  } else if (d == 2) {
    Plot_Std_Ellipse(M, V, X)
  }
}

## Subfunction: Plot_Std_Ellipse for 2D data (simplified version)
Plot_Std_Ellipse <- function(M, V, X) {
  plot(X, col = "red", pch = 16, main = "Gaussian Mixture estimated by EM",
       xlab = "1st dimension", ylab = "2nd dimension")
  k <- ncol(M)
  for (i in 1:k) {
    if (all(V[,,i] == 0)) {
      V[,,i] <- diag(rep(.Machine$double.eps, nrow(V[,,i])))
    }
    ellipse <- ellipse::ellipse(V[,,i], centre = M[, i], level = 0.95, npoints = 100)
    lines(ellipse, col = "black", lwd = 2)
    points(M[1, i], M[2, i], pch = 8, col = "black", cex = 1.5)
  }
}

## Main Execution
em_result <- EM_GM(X, nRand, ltol = 1e-5, maxiter = 500, fflag = 1, Init = NULL)

# Also compute k-Means estimate using R's kmeans
km <- kmeans(X, centers = nRand, nstart = 5)
ctrs <- km$centers

# Plot raw results
plot(X[, 1], X[, 2], col = "red", pch = 16,
     xlab = "1st dimension", ylab = "2nd dimension",
     main = "Gaussian Mixture")
points(ctrs[, 1], ctrs[, 2], col = "blue", pch = 8, cex = 1.5)
# Uncomment below if you have Plot_GM defined for 2D data:
Plot_GM(em_result$W, em_result$M, em_result$V, X)
title("Gaussian Mixture estimated by EM (black) & k-Means (blue)")
xlim(c(-2, 8)); ylim(c(-2, 8))

# Plot likelihood vs iteration
plot(em_result$L, type = "b", pch = 16, xlab = "# of iterations", ylab = "Likelihood",
     main = "Likelihood vs. Iteration")
