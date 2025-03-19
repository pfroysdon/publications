# flexibility_interpretability_classification_tutorial.R
# This script demonstrates the trade-off between flexibility and interpretability
# using a classification example.
#
# We generate synthetic 2D data where the true decision boundary is circular.
# Then, we apply:
#   1. Logistic Regression (a linear, interpretable model).
#   2. k-Nearest Neighbors (k-NN) with k = 5 (a flexible, non-parametric model).
#
# Logistic regression produces a straight-line decision boundary,
# while k-NN is capable of capturing the circular structure.

rm(list = ls())
graphics.off()
set.seed(1)

## 1. Generate Synthetic Data
N <- 200
# Generate points uniformly in [-1,1] x [-1,1]
X <- matrix(runif(N * 2, min = -1, max = 1), ncol = 2)
# Label = 1 if inside a circle of radius 0.5, else 0
y <- ifelse(rowSums(X^2) < 0.5^2, 1, 0)

## 2. Fit Logistic Regression (Interpretable Model)
addIntercept <- function(X) {
  cbind(1, X)
}
X_lr <- addIntercept(X)
logisticRegression <- function(X, y, lr = 0.1, num_iters = 1000) {
  n <- nrow(X)
  d <- ncol(X)
  w <- rep(0, d)
  losses <- numeric(num_iters)
  for (iter in 1:num_iters) {
    logits <- X %*% w
    preds <- 1 / (1 + exp(-logits))
    loss <- -mean(y * log(preds + 1e-8) + (1 - y) * log(1 - preds + 1e-8))
    losses[iter] <- loss
    grad <- t(X) %*% (preds - y) / n
    w <- w - lr * grad
  }
  list(w = w, losses = losses)
}
lr_model <- logisticRegression(X_lr, y)
cat("Logistic Regression Coefficients (Interpretable):\n")
print(lr_model$w)

## 3. Fit k-NN Classifier (Flexible Model)
knnClassifier <- function(X_train, y_train, X_test, k = 5) {
  n_test <- nrow(X_test)
  preds <- numeric(n_test)
  for (i in 1:n_test) {
    dists <- rowSums((X_train - matrix(rep(X_test[i, ], nrow(X_train)), ncol = ncol(X_train), byrow = TRUE))^2)
    idx <- order(dists)[1:k]
    preds[i] <- as.numeric(names(sort(table(y_train[idx]), decreasing = TRUE))[1])
  }
  preds
}
preds_knn <- knnClassifier(X, y, X)
preds_knn <- matrix(preds_knn, ncol = 1)

## 4. Plot Decision Boundaries
# Create grid
xx <- seq(-1, 1, length.out = 100)
yy <- seq(-1, 1, length.out = 100)
grid <- expand.grid(x1 = xx, x2 = yy)

# Logistic Regression predictions
grid_lr <- addIntercept(as.matrix(grid))
preds_lr <- 1 / (1 + exp(-grid_lr %*% lr_model$w))
preds_lr_mat <- matrix(preds_lr, nrow = 100, ncol = 100)

# k-NN predictions
preds_knn_grid <- knnClassifier(X, y, as.matrix(grid), k = 5)
preds_knn_mat <- matrix(as.numeric(preds_knn_grid), nrow = 100, ncol = 100)

par(mfrow = c(1,2))
# Logistic Regression plot
image(xx, yy, preds_lr_mat, col = gray.colors(100), xlab = "x1", ylab = "x2",
      main = "Logistic Regression (Interpretable)")
contour(xx, yy, preds_lr_mat, levels = 0.5, add = TRUE, col = "red", lwd = 2)
points(X[y == 0, 1], X[y == 0, 2], col = "blue", pch = 16)
points(X[y == 1, 1], X[y == 1, 2], col = "green", pch = 16)
legend("topright", legend = c("Decision Boundary", "Class 0", "Class 1"),
       col = c("red", "blue", "green"), pch = c(NA, 16, 16), lty = c(1, NA, NA))

# k-NN plot
image(xx, yy, preds_knn_mat, col = gray.colors(100), xlab = "x1", ylab = "x2",
      main = "k-NN (Flexible, k = 5)")
contour(xx, yy, preds_knn_mat, levels = 0.5, add = TRUE, col = "red", lwd = 2)
points(X[y == 0, 1], X[y == 0, 2], col = "blue", pch = 16)
points(X[y == 1, 1], X[y == 1, 2], col = "green", pch = 16)
legend("topright", legend = c("Decision Boundary", "Class 0", "Class 1"),
       col = c("red", "blue", "green"), pch = c(NA, 16, 16), lty = c(1, NA, NA))

cat("\nNote:\n")
cat(" - Logistic Regression yields a linear decision boundary that is simple and interpretable.\n")
cat(" - k-NN is more flexible and captures the circular boundary, but its decision rule is less transparent.\n")
