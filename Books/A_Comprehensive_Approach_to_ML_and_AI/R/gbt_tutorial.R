# gbt_tutorial.R
# Gradient Boosting Trees Tutorial
#
# This script demonstrates a simplified gradient boosting trees algorithm
# for binary classification using logistic loss.
#
# We:
#   1. Generate synthetic 2D data:
#         - Class 1: centered at (2,2)
#         - Class 0: centered at (-2,-2)
#   2. Train an ensemble of regression stumps sequentially by fitting the negative gradient
#      (residuals) of the logistic loss.
#   3. Compute predictions via an additive model and convert to probabilities using the sigmoid.
#   4. Visualize the decision boundary and report training accuracy.

rm(list = ls())
graphics.off()
set.seed(1)

## 1. Generate Synthetic Data
N <- 100
# Class 1: centered at (2,2)
X1 <- matrix(rnorm(N * 2, mean = 2), ncol = 2)
# Class 0: centered at (-2,-2)
X0 <- matrix(rnorm(N * 2, mean = -2), ncol = 2)
X <- rbind(X1, X0)
y <- c(rep(1, N), rep(0, N))

## Sigmoid Function
sigmoid <- function(x) { 1 / (1 + exp(-x)) }

## 2. Train Gradient Boosting Trees Model
gradientBoostingTrain <- function(X, y, T, eta) {
  n <- nrow(X)
  F <- rep(0, n)  # Initial model output F(x)=0
  models <- vector("list", T)
  gammas <- numeric(T)
  
  for (t in 1:T) {
    p <- sigmoid(F)
    residuals <- y - p
    stump <- stumpRegTrain(X, residuals)
    h <- stumpRegPredict(stump, X)
    gamma <- 1  # optimal for squared error loss
    gammas[t] <- gamma
    F <- F + eta * gamma * h
    models[[t]] <- list(stump = stump, coef = eta * gamma)
  }
  list(models = models)
}

stumpRegTrain <- function(X, r) {
  # Train a regression stump that minimizes squared error.
  n <- nrow(X); d <- ncol(X)
  bestLoss <- Inf
  bestFeature <- 1; bestThreshold <- 0; bestC1 <- 0; bestC2 <- 0
  for (j in 1:d) {
    thresholds <- unique(X[, j])
    for (thresh in thresholds) {
      left <- X[, j] < thresh
      right <- !left
      if (sum(left) == 0 || sum(right) == 0) next
      c1 <- mean(r[left])
      c2 <- mean(r[right])
      loss <- sum((r[left] - c1)^2) + sum((r[right] - c2)^2)
      if (loss < bestLoss) {
        bestLoss <- loss
        bestFeature <- j
        bestThreshold <- thresh
        bestC1 <- c1
        bestC2 <- c2
      }
    }
  }
  list(feature = bestFeature, threshold = bestThreshold, c1 = bestC1, c2 = bestC2)
}

stumpRegPredict <- function(stump, X) {
  # Predict with the regression stump.
  preds <- ifelse(X[, stump$feature] < stump$threshold, stump$c1, stump$c2)
  preds
}

## 3. Prediction Function for Gradient Boosting Model
gradientBoostingPredict <- function(X, modelStruct) {
  n <- nrow(X)
  F <- rep(0, n)
  for (m in modelStruct$models) {
    F <- F + m$coef * stumpRegPredict(m$stump, X)
  }
  sigmoid(F)
}

gbt_model <- gradientBoostingTrain(X, y, T = 50, eta = 0.1)
y_pred <- gradientBoostingPredict(X, gbt_model)
y_pred_class <- ifelse(y_pred >= 0.5, 1, 0)
train_acc <- mean(y_pred_class == y) * 100
cat(sprintf("Gradient Boosting Trees Training Accuracy: %.2f%%\n", train_acc))

## 4. Visualize Decision Boundary
x_min <- min(X[,1]) - 1; x_max <- max(X[,1]) + 1
y_min <- min(X[,2]) - 1; y_max <- max(X[,2]) + 1
xx <- seq(x_min, x_max, length.out = 200)
yy <- seq(y_min, y_max, length.out = 200)
gridPoints <- as.matrix(expand.grid(xx, yy))
preds_grid <- gradientBoostingPredict(gridPoints, gbt_model)
preds_grid_mat <- matrix(preds_grid, nrow = 200, ncol = 200)

par(mfrow = c(2,1))
plot(X[y==1,1], X[y==1,2], col = "blue", pch = 16, xlab = "Feature 1", ylab = "Feature 2",
     main = "Gradient Boosting Trees Classification")
points(X[y==0,1], X[y==0,2], col = "red", pch = 16)
contour(xx, yy, preds_grid_mat, levels = 0.5, add = TRUE, col = "black", lwd = 2)
