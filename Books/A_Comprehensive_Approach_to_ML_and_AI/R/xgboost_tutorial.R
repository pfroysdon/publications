# xgboost_tutorial.R
# This tutorial implements a simplified gradient boosting (XGBoost-like) model
# using regression stumps on a synthetic 2D dataset for binary classification.
# It trains T boosting rounds and plots the decision boundary.
# Note: This implementation is for educational purposes.

set.seed(1)

# 1. Generate Synthetic Data
N <- 100
# Class 1: centered at (2,2)
X1 <- matrix(rnorm(N * 2), ncol = 2) + 2
# Class 0: centered at (-2,-2)
X0 <- matrix(rnorm(N * 2), ncol = 2) - 2
X <- rbind(X1, X0)
y <- c(rep(1, N), rep(0, N))  # Labels in {0,1}

# 2. Train XGBoost-like Model
T_rounds <- 50
eta <- 0.1

# Function: Train a regression stump for a given target residual vector r
stumpRegTrain <- function(X, r) {
  n <- nrow(X); d <- ncol(X)
  bestFeature <- 1
  bestThreshold <- 0
  bestLoss <- Inf
  bestC1 <- 0
  bestC2 <- 0
  
  for (j in 1:d) {
    xj <- X[, j]
    uniqueVals <- sort(unique(xj))
    if (length(uniqueVals) == 1) next
    # Candidate thresholds: midpoints between unique values
    thresholds <- (uniqueVals[-1] + uniqueVals[-length(uniqueVals)]) / 2
    for (thresh in thresholds) {
      leftIdx <- which(xj <= thresh)
      rightIdx <- which(xj > thresh)
      if (length(leftIdx) < 5 || length(rightIdx) < 5) next
      c1 <- mean(r[leftIdx])
      c2 <- mean(r[rightIdx])
      sse_left <- sum((r[leftIdx] - c1)^2)
      sse_right <- sum((r[rightIdx] - c2)^2)
      sse <- sse_left + sse_right
      if (sse < bestLoss) {
        bestLoss <- sse
        bestFeature <- j
        bestThreshold <- thresh
        bestC1 <- c1
        bestC2 <- c2
      }
    }
  }
  list(feature = bestFeature, threshold = bestThreshold, c1 = bestC1, c2 = bestC2)
}

# Function: Predict with regression stump
stumpRegPredict <- function(stump, X) {
  j <- stump$feature
  thr <- stump$threshold
  c1 <- stump$c1; c2 <- stump$c2
  preds <- ifelse(X[, j] < thr, c1, c2)
  preds
}

# XGBoost training function
xgboostTrain <- function(X, y, T, eta) {
  n <- nrow(X)
  F <- rep(0, n)  # Initial additive model
  models <- vector("list", T)
  
  for (t in 1:T) {
    p <- 1 / (1 + exp(-F))  # sigmoid prediction
    # Negative gradient: for logistic loss, residual = y - p
    grad <- y - p
    # Train regression stump on residuals
    stump <- stumpRegTrain(X, grad)
    h <- stumpRegPredict(stump, X)
    # Line search: optimal multiplier alpha = <grad, h> / <h, h>
    alpha <- sum(grad * h) / (sum(h * h) + 1e-12)
    F <- F + eta * alpha * h
    models[[t]] <- list(stump = stump, coef = eta * alpha)
  }
  models
}

# XGBoost prediction function
xgboostPredict <- function(X, models) {
  n <- nrow(X)
  F <- rep(0, n)
  T <- length(models)
  for (t in 1:T) {
    stump <- models[[t]]$stump
    alpha <- models[[t]]$coef
    F <- F + alpha * stumpRegPredict(stump, X)
  }
  p <- 1 / (1 + exp(-F))
  p  # return probabilities (use threshold 0.5 for class 1)
}

models <- xgboostTrain(X, y, T_rounds, eta)

# 3. Visualize Decision Boundary
x1_range <- seq(min(X[,1]) - 1, max(X[,1]) + 1, length.out = 200)
x2_range <- seq(min(X[,2]) - 1, max(X[,2]) + 1, length.out = 200)
grid <- expand.grid(x1 = x1_range, x2 = x2_range)
grid_preds <- xgboostPredict(as.matrix(grid), models)
grid_pred_matrix <- matrix(grid_preds, nrow = 200, ncol = 200)

# Plot training data
plot(X[y==1,1], X[y==1,2], col = "blue", pch = 16, xlab = "Feature 1", ylab = "Feature 2",
     main = "XGBoost Classification")
points(X[y==0,1], X[y==0,2], col = "red", pch = 16)
# Plot decision boundary: contour at probability = 0.5
contour(x1_range, x2_range, grid_pred_matrix, levels = 0.5, add = TRUE, lwd = 2, col = "black")
legend("topright", legend = c("Class 1", "Class 0"), col = c("blue", "red"), pch = 16)
grid()
