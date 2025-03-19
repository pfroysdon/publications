# hyperparam_tutorial.R
# Hyperparameter Tuning Tutorial
#
# This script demonstrates hyperparameter tuning for a simple ridge regression model.
# It includes:
#   - A holdout split into training and test sets.
#   - 5-fold cross-validation over a grid of lambda values.
#   - Training the final model using the best lambda.
#   - Plotting the validation curve and learning curves.
#
rm(list = ls())
graphics.off()
set.seed(1)

## Generate synthetic data for regression
n <- 200
X <- matrix(linspace(0, 10, n), ncol = 1)
y <- sin(X) + 0.5 * rnorm(n)

## Holdout Split Function
holdoutSplit <- function(X, y, testRatio) {
  n <- length(y)
  idx <- sample(1:n)
  nTest <- round(testRatio * n)
  testIdx <- idx[1:nTest]
  trainIdx <- idx[(nTest + 1):n]
  list(X_train = X[trainIdx, , drop = FALSE],
       y_train = y[trainIdx, , drop = FALSE],
       X_test = X[testIdx, , drop = FALSE],
       y_test = y[testIdx, , drop = FALSE])
}

split_res <- holdoutSplit(X, y, 0.2)
X_train <- split_res$X_train; y_train <- split_res$y_train
X_test <- split_res$X_test; y_test <- split_res$y_test

## Define Hyperparameter Grid
lambdaGrid <- 10^seq(-4, 2, length.out = 10)

## k-Fold Cross-Validation Function for Ridge Regression
kFoldCV <- function(X, y, k, lambda) {
  n <- nrow(X)
  # Create fold assignments
  folds <- sample(rep(1:k, length.out = n))
  errors <- numeric(k)
  for (i in 1:k) {
    testIdx <- which(folds == i)
    trainIdx <- setdiff(1:n, testIdx)
    X_train_cv <- X[trainIdx, , drop = FALSE]
    y_train_cv <- y[trainIdx, , drop = FALSE]
    X_val_cv <- X[testIdx, , drop = FALSE]
    y_val_cv <- y[testIdx, , drop = FALSE]
    theta <- ridgeRegression(X_train_cv, y_train_cv, lambda)
    y_pred <- X_val_cv %*% theta
    errors[i] <- mean((y_val_cv - y_pred)^2)
  }
  mean(errors)
}

ridgeRegression <- function(X, y, lambda) {
  p <- ncol(X)
  theta <- solve(t(X) %*% X + lambda * diag(p), t(X) %*% y)
  theta
}

cvErrors <- sapply(lambdaGrid, function(lambda) kFoldCV(X_train, y_train, 5, lambda))
bestIdx <- which.min(cvErrors)
bestLambda <- lambdaGrid[bestIdx]
cat(sprintf("Best lambda from grid search: %.4f\n", bestLambda))

## Train final model using bestLambda
theta <- ridgeRegression(X_train, y_train, bestLambda)
y_pred <- X_test %*% theta
testError <- mean((y_test - y_pred)^2)
cat(sprintf("Test MSE: %.4f\n", testError))

## Plot Validation Curve
par(mfrow = c(1,1))
plot(lambdaGrid, cvErrors, type = "b", log = "x", col = "blue", lwd = 2,
     xlab = "Lambda", ylab = "Cross-Validation MSE", main = "Validation Curve")
grid()

## Learning Curves Function
learningCurves <- function(X, y, k, lambda) {
  n <- nrow(X)
  sizes <- round(seq(0.1 * n, n, length.out = 10))
  trainErrors <- numeric(length(sizes))
  valErrors <- numeric(length(sizes))
  for (i in 1:length(sizes)) {
    sizeTrain <- sizes[i]
    idx <- sample(1:n)
    trainIdx <- idx[1:sizeTrain]
    valIdx <- idx[(sizeTrain + 1):n]
    X_train_subset <- X[trainIdx, , drop = FALSE]
    y_train_subset <- y[trainIdx, , drop = FALSE]
    X_val <- X[valIdx, , drop = FALSE]
    y_val <- y[valIdx, , drop = FALSE]
    theta <- ridgeRegression(X_train_subset, y_train_subset, lambda)
    y_pred_train <- X_train_subset %*% theta
    y_pred_val <- X_val %*% theta
    trainErrors[i] <- mean((y_train_subset - y_pred_train)^2)
    valErrors[i] <- mean((y_val - y_pred_val)^2)
  }
  list(trainErrors = trainErrors, valErrors = valErrors)
}

lc <- learningCurves(X_train, y_train, 5, bestLambda)
par(mfrow = c(1,1))
plot(lc$trainErrors, type = "l", col = "blue", lwd = 2,
     xlab = "Training set size index", ylab = "MSE", main = "Learning Curves")
lines(lc$valErrors, col = "red", lwd = 2)
legend("topright", legend = c("Training Error", "Validation Error"),
       col = c("blue", "red"), lwd = 2)
grid()
